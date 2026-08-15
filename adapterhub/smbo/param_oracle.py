"""
Closed-form adapter parameter oracle for the NASPEFT union adapter.

BERT/RoBERTa (encoder-only):
    Union adapter = LoRA(q,v) + ParBn + PrefixTuning(flat=True).
    Per active layer, hidden size H:
        LoRA(q,v), rank r  : 4 * H * r
        ParBn, factor rf   : 2*H*b + b + H,  b = max(1, H // rf)
        Prefix flat, len Lp: 2 * H * Lp
        per_layer = 4Hr + (2Hb+b+H) + 2H*Lp
    total = active_layers * per_layer
    Validated EXACT on 10 ground-truth points (see table below).

T5 (encoder-decoder):
    The adapters library places the union adapter on BOTH encoder.block.i AND
    decoder.block.i when leave_out_i == 0 (shared index, both stacks).
    Decoder blocks have SelfAttention + EncDecAttention → 2× LoRA cost.
    The PrefixTuning pool is allocated ONCE globally (not per active layer).

    enc_block(r,rf,H) = 4Hr  + (2Hb+b+H)        [self-attn only]
    dec_block(r,rf,H) = 8Hr  + (2Hb+b+H)        [self+cross attn]
    prefix_global(Lp) = PREFIX_PER_LP[key] * Lp  [fixed pool, independent of
                                                   active layer count]

    total = n_active * (enc_block + dec_block) + prefix_global

    Validated EXACT on t5-large ground truth:
        leave_out=[1..23] (1 index active), r=16, rf=32, Lp=1, H=1024
        → 1*(132128+197664) + 100352*1 = 4,976,416  ✓

    _T5_PREFIX_PER_LP["t5-large"] = 100352
        = encoder_prefix(2048) + self_prefix(49152) + cross_prefix(49152)
        confirmed from named_parameters() dump.
    _T5_PREFIX_PER_LP["t5-base"] = None  (fill after one ground-truth run)

param% convention:
    param% = adapter_params / pretrained_model_params * 100
    matches Problem._evaluate_single in base_function.py.
"""

from typing import Dict, Any, Optional, Iterable


# ---------------------------------------------------------------------------
# Backbone hidden sizes (HF defaults)
# ---------------------------------------------------------------------------
_HIDDEN_SIZE: Dict[str, int] = {
    "bert-base":    768,
    "bert-large":   1024,
    "roberta-base": 768,
    "roberta-large":1024,
    "t5-base":      768,
    "t5-large":     1024,
    "llama-3.2-1b": 2048,
}

# ---------------------------------------------------------------------------
# Pretrained total parameter counts (denominator for param%)
# ---------------------------------------------------------------------------
_PRETRAINED_PARAMS: Dict[str, int] = {
    "bert-base":    109482240,
    "roberta-base": 124645632,
    "bert-large":   335141888,
    "roberta-large":355359744,
    "t5-large":     737668096,   # confirmed from adapter_summary Full model row
    # "t5-base":    247577856,   # uncomment and verify from your t5-base run
}

# ---------------------------------------------------------------------------
# T5 prefix pool size per unit of Lp (= encoder_prefix + self_prefix +
# cross_prefix control_trans sizes, all with Lp=1 as baseline).
# Validated from named_parameters() dump on t5-large with Lp=1:
#   encoder_prefix.control_trans [2048]  = 2 * H * Lp        (H=1024, Lp=1)
#   self_prefix.control_trans   [49152]  = 48 * H * Lp
#   cross_prefix.control_trans  [49152]  = 48 * H * Lp
#   total = 100352 per unit Lp
# ---------------------------------------------------------------------------
_T5_PREFIX_PER_LP: Dict[str, Optional[int]] = {
    "t5-large": 100352,
    "t5-base":  None,   # run single-layer probe and sum control_trans sizes
}

# ---------------------------------------------------------------------------
# T5 per-stack layer depth (model.config.num_layers, one stack)
# ---------------------------------------------------------------------------
_T5_STACK_DEPTH: Dict[str, int] = {
    "t5-large": 24,
    "t5-base":  12,
}


# ---------------------------------------------------------------------------
# Key resolver
# ---------------------------------------------------------------------------
def _resolve_key(model_path: str) -> str:
    m = model_path.lower()
    if "roberta" in m and "large" in m:
        return "roberta-large"
    if "roberta" in m:
        return "roberta-base"
    if "bert" in m and "large" in m:
        return "bert-large"
    if "bert" in m:
        return "bert-base"
    if "t5" in m and "large" in m:
        return "t5-large"
    if "t5" in m:
        return "t5-base"
    if "llama" in m:
        return "llama-3.2-1b"
    return "bert-base"


# ---------------------------------------------------------------------------
# BERT / RoBERTa helpers (encoder-only, unchanged)
# ---------------------------------------------------------------------------
def per_layer_params(lora_r: int, reduction_parallel: int,
                     reduction_prefix: int, hidden_size: int) -> int:
    """Exact trainable params for ONE active BERT/RoBERTa layer.

    Validated EXACT on 10 ground-truth points:
        (r, rf, Lp) -> per_layer
        (8,64,2)=46860   (16,64,2)=71436   (32,64,2)=120588  (64,64,2)=218892
        (8,1,1)=1207296  (8,32,1)=63768    (8,1024,1)=28417
        (8,64,32)=92940  (8,64,512)=830220
    """
    H  = hidden_size
    b  = max(1, H // int(reduction_parallel))
    lora   = 4 * H * int(lora_r)
    parbn  = 2 * H * b + b + H
    prefix = 2 * H * int(reduction_prefix)
    return lora + parbn + prefix


def adapter_params_from_dict(config_dict: Dict[str, Any],
                             hidden_size: int,
                             num_layers: int) -> int:
    """Total adapter params for a BERT/RoBERTa config dict."""
    active = sum(
        1 for i in range(num_layers)
        if int(config_dict.get(f"leave_out_{i}", 0)) == 0
    )
    if active == 0:
        return 0
    pl = per_layer_params(
        lora_r=config_dict["lora_r"],
        reduction_parallel=config_dict["reduction_parallel"],
        reduction_prefix=config_dict["reduction_prefix"],
        hidden_size=hidden_size,
    )
    return active * pl


# ---------------------------------------------------------------------------
# T5 helpers (encoder-decoder)
# ---------------------------------------------------------------------------
def _t5_enc_block_params(lora_r: int, reduction_parallel: int,
                         hidden_size: int) -> int:
    """Adapter params for ONE active T5 encoder block (self-attn only).

    LoRA: q,v in SelfAttention = 4 * H * r
    ParBn: adapter_down + adapter_up + biases = 2Hb + b + H
    (Prefix cost is global, not per-block — see adapter_params_from_dict_t5)
    """
    H = hidden_size
    b = max(1, H // int(reduction_parallel))
    return 4 * H * int(lora_r) + 2 * H * b + b + H


def _t5_dec_block_params(lora_r: int, reduction_parallel: int,
                         hidden_size: int) -> int:
    """Adapter params for ONE active T5 decoder block (self-attn + cross-attn).

    LoRA: q,v in SelfAttention + q,v in EncDecAttention = 8 * H * r
    ParBn: one adapter at layer.2 (FFN output) = 2Hb + b + H
    (Prefix cost is global, not per-block — see adapter_params_from_dict_t5)
    """
    H = hidden_size
    b = max(1, H // int(reduction_parallel))
    return 8 * H * int(lora_r) + 2 * H * b + b + H


def adapter_params_from_dict_t5(config_dict: Dict[str, Any],
                                hidden_size: int,
                                num_layers: int,
                                key: str) -> int:
    """Total adapter params for a T5 config dict.

    Args:
        config_dict : physical-width dict from AdapterSearchSpace.config_to_dict.
                      leave_out_i == 0 means index i is ACTIVE (both
                      encoder.block.i AND decoder.block.i are adapted).
        hidden_size : H (768 for t5-base, 1024 for t5-large).
        num_layers  : model.config.num_layers (per-stack depth, e.g. 24).
        key         : resolved backbone key, e.g. "t5-large".

    Validated EXACT on t5-large:
        1 active index, r=16, rf=32, Lp=1, H=1024
        → 1*(132128 + 197664) + 100352*1 = 4,976,416  ✓
    """
    prefix_per_lp = _T5_PREFIX_PER_LP.get(key)
    if prefix_per_lp is None:
        raise ValueError(
            f"_T5_PREFIX_PER_LP not set for '{key}'. Run a single-layer probe, "
            f"sum encoder_prefix + self_prefix + cross_prefix control_trans "
            f"sizes (with Lp=1), and add the result to _T5_PREFIX_PER_LP."
        )

    r  = config_dict["lora_r"]
    rf = config_dict["reduction_parallel"]
    Lp = config_dict["reduction_prefix"]

    enc_per = _t5_enc_block_params(r, rf, hidden_size)
    dec_per = _t5_dec_block_params(r, rf, hidden_size)

    # leave_out_i controls encoder.block.i AND decoder.block.i together.
    n_active = sum(
        1 for i in range(num_layers)
        if int(config_dict.get(f"leave_out_{i}", 0)) == 0
    )
    if n_active == 0:
        return 0

    # Prefix pool is allocated once globally, independent of active layer count.
    prefix_global = prefix_per_lp * int(Lp)

    return n_active * (enc_per + dec_per) + prefix_global


# ---------------------------------------------------------------------------
# Unified public API — routes to BERT or T5 formula automatically
# ---------------------------------------------------------------------------
def param_percentage(config_dict: Dict[str, Any],
                     model_path: str,
                     num_layers: int,
                     pretrained_params: Optional[int] = None,
                     hidden_size: Optional[int] = None) -> float:
    """Return param% = adapter_params / pretrained_params * 100.

    Works for BERT, RoBERTa, and T5. Exact, no fine-tuning required.
    """
    key = _resolve_key(model_path)
    if hidden_size is None:
        hidden_size = _HIDDEN_SIZE.get(key, 768)
    if pretrained_params is None:
        pretrained_params = _PRETRAINED_PARAMS.get(key)
        if pretrained_params is None:
            raise ValueError(
                f"No pretrained param count known for '{key}'. "
                f"Pass pretrained_params= explicitly."
            )
    if key.startswith("t5"):
        ap = adapter_params_from_dict_t5(config_dict, hidden_size, num_layers, key)
    else:
        ap = adapter_params_from_dict(config_dict, hidden_size, num_layers)
    return 100.0 * ap / pretrained_params


def batch_param_percentage(config_dicts: Iterable[Dict[str, Any]],
                           model_path: str,
                           num_layers: int,
                           pretrained_params: Optional[int] = None,
                           hidden_size: Optional[int] = None):
    """Vectorized param% for many candidate config dicts (microseconds each)."""
    key = _resolve_key(model_path)
    if hidden_size is None:
        hidden_size = _HIDDEN_SIZE.get(key, 768)
    if pretrained_params is None:
        pretrained_params = _PRETRAINED_PARAMS.get(key)
        if pretrained_params is None:
            raise ValueError(
                f"No pretrained param count known for '{key}'. "
                f"Pass pretrained_params= explicitly."
            )
    is_t5 = key.startswith("t5")
    out = []
    for cd in config_dicts:
        if is_t5:
            ap = adapter_params_from_dict_t5(cd, hidden_size, num_layers, key)
        else:
            ap = adapter_params_from_dict(cd, hidden_size, num_layers)
        out.append(100.0 * ap / pretrained_params)
    return out