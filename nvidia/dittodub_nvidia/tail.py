"""Native decoder tails with shared CUDA graph capture resources; immutable model, one ordered caller/replay stream.

Capture uses a separate shared stream and pool. Replay/copy/clone use the
installation-time caller stream. Other streams raise, including fallback calls.
No per-call synchronization: caller owns input readiness/lifetimes on that stream.
Context/shape/grad fallbacks stay native on the same ordered stream. Model use
must be serial; weights/device/dtype are immutable as required by configure.
"""
import time
import torch
from .graphs import GraphCall


def supported(x, module):
    return (not module.training and not torch.is_grad_enabled() and x.is_cuda
            and x.dtype == torch.bfloat16 and tuple(x.shape) == (1, 1, 896))


def _assign(obj, name, value):
    # A single fault-injection seam; rollback bypasses this setter.
    setattr(obj, name, value)


def install(model):
    if '_ming_accel_cuda_tail' in model.__dict__:
        raise RuntimeError('Already installed; use a fresh model (no mutation)')
    metadata = model.__dict__.get('_ming_accel_cuda', {})
    if metadata.get('installation_state') != 'incomplete' or metadata.get('stage') != 'tail' or metadata.get('profile') != 'optimized' or metadata.get('revision') != 10:
        raise RuntimeError('Requires configure to have prepared the revision 10 graph runtime')
    runtime = model.__dict__.get('_ming_accel_cuda_runtime')
    if not isinstance(runtime, dict):
        raise RuntimeError('Missing package runtime')
    package_graphs = [runtime[name] for name in ('flow', 'aggregator')]
    if any(type(g) is not GraphCall or g.graph is not None or g.signature is not None
           or g.replays != 0 for g in package_graphs):
        raise RuntimeError('Install before first package capture; use a fresh model')
    layers = list(model.model.model.layers)
    if len(layers) != 24 or not callable(model.model.forward) or not callable(model.sample):
        raise RuntimeError('Unsupported model structure')
    if any(not callable(layer.forward) or layer.training for layer in layers):
        raise RuntimeError('Requires native eval layers')
    # All resources and wrappers exist before the first mutation.
    replay_stream = torch.cuda.current_stream()
    replay_id = replay_stream.cuda_stream
    device = replay_stream.device
    pool = torch.cuda.graph_pool_handle()
    capture_stream = torch.cuda.Stream(device=device)
    def require_stream():
        if torch.cuda.current_stream(device).cuda_stream != replay_id:
            raise RuntimeError('tail_pool requires the installation-time ordered replay stream')
    counters = {'supported_calls': 0, 'fallbacks': 0}
    graphs = []
    edits = []
    for layer in layers:
        original = layer.forward
        def tail(residual, attention, _layer=layer):
            x = residual + attention
            return x + _layer.mlp(_layer.post_attention_layernorm(x))
        graph = SharedPoolGraphCall(tail)
        graph.pool = pool
        graph.stream = capture_stream
        graph.require_stream = require_stream
        graphs.append(graph)
        def forward(hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                    output_attentions=False, use_cache=False, cache_position=None,
                    position_embeddings=None, _layer=layer, _f=original, _g=graph, **kwargs):
            if not supported(hidden_states, _layer) or kwargs:
                counters['fallbacks'] += 1
                return _f(hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                          past_key_value=past_key_value, output_attentions=output_attentions,
                          use_cache=use_cache, cache_position=cache_position,
                          position_embeddings=position_embeddings, **kwargs)
            counters['supported_calls'] += 1
            residual = hidden_states
            attention, weights, present = _layer.self_attn(
                hidden_states=_layer.input_layernorm(hidden_states), attention_mask=attention_mask,
                position_ids=position_ids, past_key_value=past_key_value,
                output_attentions=output_attentions, use_cache=use_cache,
                cache_position=cache_position, position_embeddings=position_embeddings)
            outputs = (_g(residual, attention),)
            if output_attentions: outputs += (weights,)
            if use_cache: outputs += (present,)
            return outputs
        edits.append((layer, 'forward', forward))
    original_llm = model.model.forward
    def guarded_llm(*args, **kwargs):
        require_stream()
        return original_llm(*args, **kwargs)
    original_sample = model.sample
    def guarded_sample(*args, **kwargs):
        require_stream()
        generator = original_sample(*args, **kwargs)
        while True:
            require_stream()
            try: value = next(generator)
            except StopIteration: return
            yield value
    edits.extend([(model.model, 'forward', guarded_llm), (model, 'sample', guarded_sample)])
    # Restore exact own dictionaries/classes, including absent instance forwards.
    objects = layers + [model.model, model] + package_graphs
    journal = [(obj, type(obj), obj.__dict__.copy()) for obj in objects]
    state = dict(mode='tail_pool', graphs=graphs + package_graphs, counters=counters,
                 replay_stream=replay_stream, capture_stream=capture_stream,
                 stream_contract='installation-time caller stream; foreign stream rejected')
    try:
        for graph in package_graphs:
            _assign(graph, '__class__', SharedPoolGraphCall)
            _assign(graph, 'pool', pool)
            _assign(graph, 'stream', capture_stream)
            _assign(graph, 'require_stream', require_stream)
        for obj, name, value in edits:
            _assign(obj, name, value)
        _assign(model, '_ming_accel_cuda_tail', state)
    except BaseException:
        for obj, cls, attrs in reversed(journal):
            if type(obj) is not cls: object.__setattr__(obj, '__class__', cls)
            obj.__dict__.clear()
            obj.__dict__.update(attrs)
        raise


def stats(model):
    state = model.__dict__.get('_ming_accel_cuda_tail')
    if state is None:
        return {}
    graphs = state['graphs'][:24]
    return dict(tail_graphs=len(graphs),
                tail_captured=sum(g.graph is not None for g in graphs),
                tail_replays=sum(g.replays for g in graphs),
                tail_capture_seconds=sum(g.capture_seconds for g in graphs),
                tail_supported_calls=state['counters']['supported_calls'],
                tail_fallbacks=state['counters']['fallbacks'])


class SharedPoolGraphCall(GraphCall):
    """Shared capture resources, explicitly guarded caller-stream replay."""
    def __call__(self, *args, **kwargs):
        self.require_stream()
        values = list(args) + list(kwargs.values())
        signature = (len(args), tuple(kwargs), tuple(
            (v.shape, v.dtype, v.device) if torch.is_tensor(v) else (type(v), v)
            for v in values), torch.is_autocast_enabled('cuda'),
            torch.get_autocast_dtype('cuda'), torch.is_inference_mode_enabled())
        if self.signature is not None and signature != self.signature:
            return self.function(*args, **kwargs)
        if self.graph is None:
            started = time.perf_counter()
            # Capture preparation is local until it succeeds; retry after an
            # exception cannot publish a partially initialized graph/signature.
            static = [v.clone(memory_format=torch.contiguous_format) if torch.is_tensor(v) else v for v in values]
            static_args = static[:len(args)]
            static_kwargs = dict(zip(kwargs, static[len(args):]))
            stream = self.stream
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3): self.function(*static_args, **static_kwargs)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream, pool=self.pool):
                output = self.function(*static_args, **static_kwargs)
            torch.cuda.synchronize()
            self.values, self.output, self.signature, self.graph = static, output, signature, graph
            self.capture_seconds = time.perf_counter() - started
        for target, value in zip(self.values, values):
            if torch.is_tensor(value): target.copy_(value)
        self.graph.replay()
        self.replays += 1
        return self.output.clone()
