"""DittoDub NVIDIA accelerator for Ming 3.0 TTS 0.5B (official PyTorch inference): exact output, about 2x faster."""
import hashlib
import inspect
from importlib.metadata import version
from importlib.resources import files
import json
from pathlib import Path
import sys
import time


def _source_qualification(model):
    """Qualify the loaded implementations, including instance method bindings."""
    hashes = json.loads(files(__package__).joinpath('source_hashes.json').read_text())
    names = {'modeling_bailingmm.py': type(model).__module__,
             **{path: path[:-3].replace('/', '.') for path in hashes if path.startswith('fm/')}}
    loaded = {}
    paths = {}
    for relative, digest in hashes.items():
        module = sys.modules.get(names[relative])
        filename = getattr(module, '__file__', None)
        if not filename:
            return False, f'Unqualified loaded upstream module: {relative}'
        path = Path(filename).resolve()
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            return False, f'Unqualified loaded upstream source: {relative}'
        loaded[names[relative]] = module
        paths[names[relative]] = path

    def qualified_method(instance, name):
        cls = type(instance)
        module = loaded.get(cls.__module__)
        method = getattr(instance, name, None)
        function = getattr(method, '__func__', None)
        source = inspect.getsourcefile(inspect.unwrap(function)) if function else None
        return (module is not None and getattr(module, cls.__name__, None) is cls
                and function is getattr(cls, name, None) and source is not None
                and Path(source).resolve() == paths[cls.__module__])

    try:
        cfm = model.flowloss.cfm
        instances = ((model, loaded[type(model).__module__].BailingMMNativeForConditionalGeneration, 'sample'),
                     (model.flowloss, loaded['fm.flowloss'].FlowLoss, 'sample'),
                     (cfm, loaded['fm.CFM'].CFM, 'sample'),
                     (cfm.model, loaded['fm.dit'].DiT, 'forward'),
                     (model.linear_proj_audio, loaded['fm.dit'].Aggregator, 'forward'))
        for instance, cls, method in instances:
            if type(instance) is not cls or not qualified_method(instance, method):
                return False, f'Unqualified loaded implementation: {cls.__name__}.{method}'
        for module in model.modules():
            cls = type(module)
            if (cls.__module__.startswith('fm.') or cls.__name__ == 'RMSNorm'):
                if not qualified_method(module, 'forward'):
                    return False, f'Unqualified loaded implementation: {cls.__name__}.forward'
        helper = loaded['fm.CFM'].get_epss_timesteps
        source = inspect.getsourcefile(helper)
        if (inspect.unwrap(cfm.sample).__globals__.get('get_epss_timesteps') is not helper
                or helper.__module__ != 'fm.CFM' or source is None
                or Path(source).resolve() != paths['fm.CFM']):
            return False, 'Unqualified loaded implementation: get_epss_timesteps'
    except (AttributeError, TypeError, ValueError):
        return False, 'Unqualified loaded upstream model structure'
    from .qwen import qualify_qwen
    return qualify_qwen(model)


def qualification(model):
    import torch
    expected = {'torch': '2.11.0+cu128', 'transformers': '4.46.3',
                'x-transformers': '2.9.2', 'diffusers': '0.33.0', 'triton': '3.6.0'}
    for name, pinned in expected.items():
        if version(name) != pinned:
            return False, f'{name} differs from qualified {pinned}'
    if not torch.cuda.is_available() or torch.version.cuda != '12.8':
        return False, 'Requires qualified CUDA 12.8 runtime'
    parameters = list(model.parameters())
    if not parameters or any(p.device != parameters[0].device or p.device.type != 'cuda'
                             or p.dtype != torch.bfloat16 for p in parameters):
        return False, 'Requires the original CUDA BF16 model on one GPU'
    if torch.cuda.get_device_name(parameters[0].device) != 'NVIDIA GeForce RTX 5090':
        return False, 'Exact-output qualification covers NVIDIA GeForce RTX 5090'
    if torch.cuda.current_device() != parameters[0].device.index:
        return False, 'Set the current CUDA device to the model device before configuring'
    if (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.benchmark
            or not torch.backends.cudnn.deterministic):
        return False, 'Requires TF32 disabled, cudnn.benchmark=False, cudnn.deterministic=True'
    if (type(model).__name__ != 'BailingMMNativeForConditionalGeneration'
            or model.model_type != 'dense' or model.patch_size != 1
            or model.history_patch_size != 1 or model.latent_dim != 64
            or model.model.config.hidden_size != 896 or any(m.training for m in model.modules())):
        return False, 'Requires pinned Ming 0.5B dense model in eval mode'
    supported, reason = _source_qualification(model)
    if not supported:
        return supported, reason
    return True, 'RTX 5090 / PyTorch 2.11.0+cu128 / pinned Ming dense source'


def configure(model, profile='optimized'):
    """Configure before adding sampling wrappers. A fresh model restores baseline.

    Optimized inference is synchronous and immutable. Do not train, replace or
    mutate weights, move devices, or share this model between concurrent calls.
    CUDA graphs capture parameter storage; arbitrary mutations are not detected.
    """
    if profile not in {'baseline', 'optimized'}:
        raise ValueError(f'Unknown CUDA profile: {profile}')
    previous = model.__dict__.get('_ming_accel_cuda')
    if previous is not None:
        if previous.get("installation_state") != "complete":
            raise RuntimeError("Previous installation did not complete; load a fresh model")
        if previous['profile'] != profile:
            raise ValueError('Load a fresh model to switch CUDA profiles')
        return dict(previous)
    metadata = dict(backend='cuda', profile=profile, revision=10 if profile=='optimized' else 0,
                    note='Same output as stock Ming 3.0 CUDA inference, faster.')
    if profile == 'baseline':
        return dict(metadata, qualification='Baseline; no optimization installed', preparation_seconds=0.)
    supported, reason = qualification(model)
    if not supported:
        raise RuntimeError(reason + "; use profile='baseline'")
    import torch
    from .graphs import GraphCall, hoisted_solver
    from .normalization import install
    from transformers.modeling_outputs import CausalLMOutputWithPast

    started = time.perf_counter()
    original_sample = model.sample
    original_llm = model.model.forward
    original_solver = model.flowloss.cfm.sample
    original_aggregator = model.linear_proj_audio.forward
    flow = GraphCall(hoisted_solver(model.flowloss.cfm))
    aggregator = GraphCall(original_aggregator)
    stats = {'hidden_calls': 0, 'flow_fallbacks': 0, 'aggregator_fallbacks': 0}
    active = False

    def hidden_forward(*args, **kwargs):
        if not active:
            return original_llm(*args, **kwargs)
        # This private path is entered only while advancing the pinned speech
        # generator. Direct LLM/text calls retain logits and their normal API.
        inputs = dict(kwargs)
        if (args or inputs.pop('audio_mask', None) is not None
                or inputs.pop('image_mask', None) is not None
                or not inputs.get('return_dict') or not inputs.get('output_hidden_states')):
            return original_llm(*args, **kwargs)
        out = model.model.model(**inputs)
        stats['hidden_calls'] += 1
        return CausalLMOutputWithPast(past_key_values=out.past_key_values,
                                     hidden_states=out.hidden_states, attentions=out.attentions)

    def sample(*args, **kwargs):
        nonlocal active
        if torch.is_grad_enabled() or any(m.training for m in model.modules()):
            raise RuntimeError('Configured CUDA model requires eval and inference_mode/no_grad')
        generator = original_sample(*args, **kwargs)
        while True:
            if active:
                raise RuntimeError('Concurrent/reentrant use of a configured CUDA model is unsupported')
            active = True
            try:
                value = next(generator)
            except StopIteration:
                return
            finally:
                active = False
            yield value

    def solver(*args, **kwargs):
        shapes = {'noise': (1,1,64), 'c': (1,1,896),
                  'latent_history': (1,1,64), 'sde_noise': (10,1,1,64)}
        supported = (active and not args
            and all(torch.is_tensor(kwargs.get(k)) and kwargs[k].is_cuda
                    and kwargs[k].dtype == torch.bfloat16 and tuple(kwargs[k].shape) == shape
                    for k, shape in shapes.items())
            and kwargs.get('cfg_scale',2.) == 2. and kwargs.get('patch_size',1) == 1
            and kwargs.get('sigma',.25) == .25 and kwargs.get('temperature',0.) == 0.
            and kwargs.get('sway_sampling_coef',-1.) == -1. and kwargs.get('use_epss',True))
        if not supported:
            stats['flow_fallbacks'] += 1
            return original_solver(*args, **kwargs)
        return flow(**kwargs)

    def aggregate(x, *args, **kwargs):
        if active and not args and not kwargs and x.is_cuda and x.dtype==torch.bfloat16 and tuple(x.shape)==(1,1,64):
            return aggregator(x)
        stats['aggregator_fallbacks'] += 1
        return original_aggregator(x, *args, **kwargs)

    # Roll back every own attribute changed by normalization/base/tail setup.
    # Failed models remain explicitly incomplete; retry requires a fresh model.
    journal = [(obj, obj.__dict__.copy()) for obj in model.modules()]
    model.__dict__["_ming_accel_cuda"] = dict(metadata, installation_state="incomplete")
    try:
        install(model)
        model.model.forward = hidden_forward
        model.flowloss.cfm.sample = solver
        model.linear_proj_audio.forward = aggregate
        model.sample = sample
        model.__dict__['_ming_accel_cuda_runtime'] = dict(flow=flow, aggregator=aggregator, stats=stats)
        model.__dict__['_ming_accel_cuda']['stage'] = 'tail'
        from .tail import install as install_tail
        install_tail(model)
        metadata.update(installation_state='complete', qualification=reason,
                        preparation_seconds=time.perf_counter()-started,
                        graph_capture='Lazy on first supported call; inspect runtime_stats(model)',
                        replay_stream_contract='Serial immutable model; installation-time caller stream; '
                                               'foreign streams rejected; caller owns ready inputs and lifetimes',
                        capture_stream_contract='Separate shared capture stream and pool; no warm synchronization')
        model.__dict__['_ming_accel_cuda'] = dict(metadata)
    except BaseException:
        for obj, attrs in reversed(journal):
            obj.__dict__.clear()
            obj.__dict__.update(attrs)
        model.__dict__['_ming_accel_cuda'] = dict(metadata, installation_state='incomplete')
        raise
    return metadata


def runtime_stats(model):
    runtime = model.__dict__.get('_ming_accel_cuda_runtime')
    if runtime is None:
        return {}
    from .tail import stats as tail_stats
    return {**tail_stats(model), **runtime['stats'], **{f'{name}_{key}': getattr(runtime[name],key)
            for name in ('flow','aggregator') for key in ('replays','capture_seconds')}}
