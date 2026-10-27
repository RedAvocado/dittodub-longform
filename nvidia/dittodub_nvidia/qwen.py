"""Fail-closed qualification of the Qwen2 implementations used by decoder tails."""
import hashlib
import inspect
from pathlib import Path
import sys
from types import CodeType

QWEN_MODULE = 'transformers.models.qwen2.modeling_qwen2'
QWEN_SHA256 = '4568d588b45dd28152d97c37300a5e227589bc70b9fef5224af29977ecf6abdc'


def qualify_qwen(model):
    module = sys.modules.get(QWEN_MODULE)
    filename = getattr(module, '__file__', None)
    if filename is None:
        return False, 'Unqualified loaded Qwen2 module'
    path = Path(filename).resolve()
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != QWEN_SHA256:
        return False, 'Unqualified loaded Qwen2 source'
    # Disk hashes alone miss class monkeypatching or replacement __code__.
    # Compile the verified bytes without executing them and compare live code.
    codes = {}
    def collect(code):
        codes[code.co_qualname] = code
        for item in code.co_consts:
            if isinstance(item, CodeType):
                collect(item)
    collect(compile(source, str(path), 'exec', dont_inherit=True, optimize=sys.flags.optimize))

    def native_function(function, name):
        return (inspect.isfunction(function)
                and function.__module__ == QWEN_MODULE
                and function.__globals__ is module.__dict__
                and function.__code__ == codes.get(name)
                and not hasattr(function, '__wrapped__'))

    def native_method(instance, cls):
        method = getattr(instance, 'forward', None)
        return (type(instance) is cls and cls.__module__ == QWEN_MODULE
                and getattr(module, cls.__name__, None) is cls
                and 'forward' not in instance.__dict__
                and getattr(method, '__func__', None) is cls.forward
                and native_function(cls.forward, cls.__name__ + '.forward'))

    try:
        llm = model.model
        core = llm.model
        attention_cls = module.QWEN2_ATTENTION_CLASSES[core.config._attn_implementation]
        expected = [(llm, module.Qwen2ForCausalLM), (core, module.Qwen2Model),
                    (core.norm, module.Qwen2RMSNorm)]
        if len(core.layers) != 24:
            return False, 'Requires all 24 native Qwen2 decoder layers'
        for layer in core.layers:
            expected.extend([(layer, module.Qwen2DecoderLayer),
                             (layer.self_attn, attention_cls),
                             (layer.mlp, module.Qwen2MLP),
                             (layer.input_layernorm, module.Qwen2RMSNorm),
                             (layer.post_attention_layernorm, module.Qwen2RMSNorm)])
        for instance, cls in expected:
            if not native_method(instance, cls):
                return False, f'Unqualified loaded Qwen2 implementation: {cls.__name__}.forward'
        for name in ('apply_rotary_pos_emb', 'rotate_half', 'repeat_kv'):
            if not native_function(getattr(module, name), name):
                return False, f'Unqualified loaded Qwen2 helper: {name}'
        # Hooks on captured components would have different call counts after
        # replay. Configure before instrumentation; no hooks are qualified.
        for instance in llm.modules():
            if instance._forward_hooks or instance._forward_pre_hooks:
                return False, 'Qwen2 forward hooks are not qualified for graph capture'
    except (AttributeError, KeyError, TypeError, ValueError):
        return False, 'Unqualified loaded Qwen2 model structure'
    return True, 'Loaded Ming/flow and native Qwen2 implementations match pinned sources'
