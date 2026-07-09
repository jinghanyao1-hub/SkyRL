"""Serialized FP8 weight-sync helpers.

This module prepares Megatron-exported HF/vLLM weights for rollout engines that
expect an FP8 checkpoint-style payload: FP8 weight tensors plus explicit scale
tensors. Megatron parameters remain in their training dtype; only the sync
payload is quantized.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch


SERIALIZED_BLOCKWISE_FP8 = "serialized_blockwise"


def use_power_2_scales_default() -> bool:
    """Whether serialized rollout weights should use power-of-2 block scales.

    This mirrors Megatron-TE ``Float8BlockScaling``: TE quantizes the training
    forward weights with power-of-2 (UE8M0-representable) block scales unless
    ``NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1`` forces FP32 scales. The rollout
    weight sync must use the *same* scale format as the training forward,
    otherwise the vLLM rollout weights and the Megatron train-forward weights
    disagree by up to 2x per block (rollout/train logprob divergence).
    """

    return os.getenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "0") != "1"


_NON_LINEAR_WEIGHT_SUBSTRINGS = (
    "embed_tokens",
    "embedding",
    "word_embeddings",
    "lm_head",
    "output_layer",
)
_SLIME_QWEN35_FP8_WEIGHT_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
    ".linear_attn.in_proj_qkv.weight",
    ".linear_attn.in_proj_z.weight",
    ".linear_attn.out_proj.weight",
    # Qwen3.5-MoE dense-shaped shared expert (vLLM builds it WITH quant_config, so
    # FP8; the router ``mlp.gate`` and ``shared_expert_gate`` stay BF16 by omission).
    ".mlp.shared_expert.gate_proj.weight",
    ".mlp.shared_expert.up_proj.weight",
    ".mlp.shared_expert.down_proj.weight",
)
# Qwen3.5-MoE routed experts. Megatron-Bridge exports them as batched 3D tensors
# ``mlp.experts.gate_up_proj`` [E, 2*moe_inter, hidden] and ``mlp.experts.down_proj``
# [E, hidden, moe_inter]. vLLM's proven blockwise-FP8 MoE path is per-expert
# (``experts.N.<proj>.weight`` + ``.weight_scale_inv`` -> fused ``w13/w2_weight_scale_inv``
# via make_expert_params_mapping), so we un-batch + un-fuse into per-expert 2D linears.
_QWEN35_MOE_GATE_UP_SUFFIX = ".mlp.experts.gate_up_proj"
_QWEN35_MOE_DOWN_SUFFIX = ".mlp.experts.down_proj"
_SLIME_QWEN35_UNQUANTIZED_LINEAR_SUFFIXES = (
    ".in_proj_b",
    ".in_proj_a",
)
_QWEN35_LINEAR_ATTN_PREFIX_TEMPLATES = (
    "{model_prefix}.layers.{layer_idx}.linear_attn",
    "{model_prefix}.language_model.layers.{layer_idx}.linear_attn",
    "language_model.model.layers.{layer_idx}.linear_attn",
    "{model_prefix}.language_model.model.layers.{layer_idx}.linear_attn",
)
_QWEN35_VISION_ATTN_PROJ_PREFIX_TEMPLATES = (
    "visual.blocks.{block_idx}.attn.proj",
    "{model_prefix}.visual.blocks.{block_idx}.attn.proj",
    "language_model.visual.blocks.{block_idx}.attn.proj",
    "{model_prefix}.language_model.visual.blocks.{block_idx}.attn.proj",
)


@dataclass(frozen=True)
class SerializedFp8Config:
    """Configuration for serialized FP8 rollout weight sync."""

    weight_block_size: tuple[int, int] = (128, 128)
    quantized_weight_dtype: torch.dtype = torch.float8_e4m3fn
    scale_dtype: torch.dtype = torch.float32
    power_2_scale: bool = field(default_factory=use_power_2_scales_default)


def get_serialized_fp8_quantization_config(
    weight_block_size: Sequence[int] = (128, 128),
    modules_to_not_convert: Sequence[str] | None = None,
) -> dict:
    """Return the HF quantization_config needed for vLLM serialized FP8."""

    qconfig = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [int(weight_block_size[0]), int(weight_block_size[1])],
    }
    if modules_to_not_convert:
        ignored = list(modules_to_not_convert)
        qconfig["ignored_layers"] = ignored
        qconfig["modules_to_not_convert"] = ignored
    return qconfig


def get_qwen35_slime_parity_modules_to_not_convert(hf_config: Any, model_prefix: str = "model") -> list[str]:
    """Return vLLM module prefixes that Slime leaves unquantized for Qwen3.5.

    Slime's Qwen3.5 FP8 sync quantizes a narrow Megatron-name allow-list. The
    GDN ``in_proj_b``/``in_proj_a`` projection is intentionally absent from that
    allow-list, while vLLM would otherwise quantize the fused ``in_proj_ba``
    module under a global FP8 config. vLLM's skip logic checks full module
    prefixes and requires every shard of a fused module to have the same scheme,
    so emit both shard prefixes for each linear-attention layer.

    The Qwen3.5 text model and full conditional-generation wrapper use
    different prefix families. vLLM may also apply its HF-to-vLLM mapper to
    ignored layers, so include the HF checkpoint form and the mapped vLLM forms.
    """

    text_config = getattr(hf_config, "text_config", None) or getattr(hf_config, "language_config", None) or hf_config
    model_type = str(getattr(text_config, "model_type", "") or getattr(hf_config, "model_type", ""))
    if "qwen3_5" not in model_type and not hasattr(text_config, "layer_types"):
        return []

    layer_types = list(getattr(text_config, "layer_types", []) or [])
    ignored: list[str] = []
    for layer_idx, layer_type in enumerate(layer_types):
        if layer_type != "linear_attention":
            continue
        layer_prefixes = []
        for template in _QWEN35_LINEAR_ATTN_PREFIX_TEMPLATES:
            prefix = template.format(model_prefix=model_prefix, layer_idx=layer_idx)
            if prefix not in layer_prefixes:
                layer_prefixes.append(prefix)

        for layer_prefix in layer_prefixes:
            for suffix in _SLIME_QWEN35_UNQUANTIZED_LINEAR_SUFFIXES:
                ignored.append(f"{layer_prefix}{suffix}")

    # Qwen3.5 text-only RL runs set language_model_only=true, but vLLM 0.23 can
    # still instantiate the unused vision tower before the multimodal limits take
    # effect. Its row-parallel attention output projection is 1152 wide and fails
    # the 128-wide FP8 block divisibility check under TP2, so keep it BF16. vLLM's
    # FP8 ignore matcher is exact-match by default.
    vision_config = getattr(hf_config, "vision_config", None) or getattr(hf_config, "visual_config", None)
    vision_depth = 0
    if vision_config is not None:
        for attr in ("depth", "num_hidden_layers", "num_layers"):
            value = getattr(vision_config, attr, None)
            if isinstance(value, int) and value > 0:
                vision_depth = value
                break
    for block_idx in range(vision_depth):
        for template in _QWEN35_VISION_ATTN_PROJ_PREFIX_TEMPLATES:
            ignored.append(template.format(model_prefix=model_prefix, block_idx=block_idx))
    return ignored


def should_use_serialized_fp8(mode: str | None) -> bool:
    return mode == SERIALIZED_BLOCKWISE_FP8


def is_quantizable_weight(name: str, tensor: torch.Tensor) -> bool:
    """Return whether an exported HF tensor should be serialized as FP8.

    vLLM's FP8 config applies to Linear modules. HF checkpoints also contain 2D
    embedding/output weights, so keep known non-Linear weight tables unquantized.
    """

    if not name.endswith(".weight") or tensor.ndim != 2:
        return False

    return is_quantizable_weight_shape(name, tensor.shape)


def is_quantizable_weight_shape(name: str, shape: Sequence[int]) -> bool:
    if not name.endswith(".weight") or len(shape) != 2:
        return False
    if any(part in name for part in _NON_LINEAR_WEIGHT_SUBSTRINGS):
        return False
    return name.endswith(_SLIME_QWEN35_FP8_WEIGHT_SUFFIXES)


def scale_name_for_weight(name: str) -> str:
    if not name.endswith(".weight"):
        raise ValueError(f"FP8 scale can only be derived from .weight tensors: {name}")
    return name[: -len(".weight")] + ".weight_scale_inv"


def shared_blockwise_fp8_group(name: str) -> tuple[str, str] | None:
    """Return a shared-scale group key for small fused output shards.

    vLLM represents Qwen3.5 GDN ``in_proj_b`` and ``in_proj_a`` as the fused
    ``in_proj_ba`` module. With 128x128 blockwise FP8, the two 64-row shards
    occupy one shared output block, so their ``weight_scale_inv`` values must be
    computed from the fused [B; A] matrix, not independently per shard.
    """

    for shard in ("b", "a"):
        suffix = f".in_proj_{shard}.weight"
        if name.endswith(suffix):
            return name[: -len(suffix)], shard
    return None


def blockwise_fp8_scale_shape(weight_shape: Sequence[int], block_size: Sequence[int]) -> list[int]:
    if len(weight_shape) != 2:
        raise ValueError(f"Blockwise FP8 expects a 2D weight, got shape={list(weight_shape)}")
    block_m, block_n = int(block_size[0]), int(block_size[1])
    return [(int(weight_shape[0]) + block_m - 1) // block_m, (int(weight_shape[1]) + block_n - 1) // block_n]


def blockwise_cast_to_fp8(
    weight: torch.Tensor,
    block_size: Sequence[int],
    power_2_scale: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D tensor to FP8 with FP32 block scales.

    The returned scale is the dequant scale, matching Slime's
    ``weight_scale_inv`` convention for blockwise FP8:

    ``dequantized_weight ~= qweight.float() * scale``.

    When ``power_2_scale`` is True (the default, matching Megatron-TE
    ``Float8BlockScaling`` with ``NVTE_FP8_BLOCK_SCALING_FP32_SCALES != 1``),
    the dequant scale is rounded up to the next power of two. This is the same
    ``2**ceil(log2(amax/fp8_max))`` rule that SGLang/DeepGEMM ``ceil_to_ue8m0``
    applies, so the rollout weights are quantized against the exact scale the
    Megatron training forward uses. Exact FP32 scales instead make the rollout
    and train effective weights disagree by up to 2x per block.
    """

    if weight.ndim != 2:
        raise ValueError(f"Blockwise FP8 expects a 2D tensor, got shape={tuple(weight.shape)}")

    block_m, block_n = int(block_size[0]), int(block_size[1])
    rows, cols = weight.shape
    padded_rows = ((rows + block_m - 1) // block_m) * block_m
    padded_cols = ((cols + block_n - 1) // block_n) * block_n

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    weight_fp32 = weight.detach().to(torch.float32)
    if padded_rows != rows or padded_cols != cols:
        padded = weight_fp32.new_zeros((padded_rows, padded_cols))
        padded[:rows, :cols].copy_(weight_fp32)
    else:
        padded = weight_fp32

    blocks = padded.view(padded_rows // block_m, block_m, padded_cols // block_n, block_n)
    blocks = blocks.permute(0, 2, 1, 3)
    scale = blocks.abs().amax(dim=(2, 3)).clamp(min=1e-10) / fp8_info.max
    if power_2_scale:
        # Round the dequant scale up to the next power of two so the rollout
        # engine and the Megatron-TE training forward quantize weights against
        # identical block scales. Rounding up keeps amax / scale <= fp8_max.
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    q_blocks = (blocks / scale[:, :, None, None]).clamp(min=fp8_info.min, max=fp8_info.max)
    q_blocks = q_blocks.to(torch.float8_e4m3fn)
    q_padded = q_blocks.permute(0, 2, 1, 3).contiguous().view(padded_rows, padded_cols)
    q_weight = q_padded[:rows, :cols].contiguous()
    return q_weight, scale.to(torch.float32).contiguous()


def batched_moe_expert_spec(name: str) -> tuple[str, tuple[str, ...], bool] | None:
    """Recognize a batched Qwen3.5-MoE expert tensor exported by Megatron-Bridge.

    Returns ``(experts_base, proj_names, is_gate_up)`` where ``experts_base`` is the
    ``...mlp.experts`` prefix and per-expert HF names are
    ``f"{experts_base}.{expert_idx}.{proj}.weight"``. ``gate_up_proj`` is split along
    the output dim into ``gate_proj`` (first half) and ``up_proj`` (second half),
    matching vLLM's ``stacked``/``expert`` mappings. Returns ``None`` for non-experts.
    """

    if name.endswith(_QWEN35_MOE_GATE_UP_SUFFIX):
        return name[: -len(".gate_up_proj")], ("gate_proj", "up_proj"), True
    if name.endswith(_QWEN35_MOE_DOWN_SUFFIX):
        return name[: -len(".down_proj")], ("down_proj",), False
    return None


def _per_expert_2d_slices(mat: torch.Tensor, is_gate_up: bool) -> list[torch.Tensor]:
    """Split one expert's 2D matrix into the per-projection 2D linears."""
    if is_gate_up:
        half = mat.shape[0] // 2
        return [mat[:half], mat[half:]]
    return [mat]


def iter_batched_moe_expert_fp8_tensors(
    name: str,
    tensor: torch.Tensor,
    config: SerializedFp8Config,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Un-batch a 3D expert tensor into per-expert FP8 blockwise weights + scales."""
    spec = batched_moe_expert_spec(name)
    if spec is None:
        raise ValueError(f"Not a batched MoE expert tensor: {name}")
    experts_base, proj_names, is_gate_up = spec
    num_experts = tensor.shape[0]
    for expert_idx in range(num_experts):
        mat = tensor[expert_idx]
        for proj, sub in zip(proj_names, _per_expert_2d_slices(mat, is_gate_up)):
            weight_name = f"{experts_base}.{expert_idx}.{proj}.weight"
            q_weight, scale = blockwise_cast_to_fp8(
                sub.contiguous(), config.weight_block_size, config.power_2_scale
            )
            yield weight_name, q_weight
            yield scale_name_for_weight(weight_name), scale.to(config.scale_dtype)


def iter_batched_moe_expert_metadata(
    name: str,
    shape: Sequence[int],
    config: SerializedFp8Config,
) -> Iterable[tuple[str, torch.dtype, list[int]]]:
    """Metadata twin of ``iter_batched_moe_expert_fp8_tensors`` (no quantization)."""
    spec = batched_moe_expert_spec(name)
    if spec is None:
        raise ValueError(f"Not a batched MoE expert tensor: {name}")
    experts_base, proj_names, is_gate_up = spec
    num_experts, out_dim, in_dim = int(shape[0]), int(shape[1]), int(shape[2])
    per_out = out_dim // 2 if is_gate_up else out_dim
    for expert_idx in range(num_experts):
        for proj in proj_names:
            weight_name = f"{experts_base}.{expert_idx}.{proj}.weight"
            weight_shape = [per_out, in_dim]
            yield weight_name, config.quantized_weight_dtype, weight_shape
            yield (
                scale_name_for_weight(weight_name),
                config.scale_dtype,
                blockwise_fp8_scale_shape(weight_shape, config.weight_block_size),
            )


def iter_serialized_fp8_tensors(
    name: str,
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
    config: SerializedFp8Config,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield one or more tensors for a single exported HF weight."""

    if tensor.ndim == 3 and batched_moe_expert_spec(name) is not None:
        yield from iter_batched_moe_expert_fp8_tensors(name, tensor, config)
        return

    if is_quantizable_weight(name, tensor):
        q_weight, scale = blockwise_cast_to_fp8(tensor, config.weight_block_size, config.power_2_scale)
        yield name, q_weight
        yield scale_name_for_weight(name), scale.to(config.scale_dtype)
        return

    yield name, tensor.to(dtype=target_dtype)


def iter_shared_blockwise_fp8_tensors(
    grouped_tensors: Mapping[str, tuple[str, torch.Tensor]],
    target_dtype: torch.dtype,
    config: SerializedFp8Config,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield serialized FP8 tensors for a small fused output group.

    The input mapping currently handles Qwen3.5 GDN ``in_proj_b``/``in_proj_a``.
    If either shard is not quantizable, the function falls back to target dtype
    for the whole group.
    """

    ordered = [grouped_tensors[shard] for shard in ("b", "a")]
    if not all(is_quantizable_weight(name, tensor) for name, tensor in ordered):
        for name, tensor in ordered:
            yield name, tensor.to(dtype=target_dtype)
        return

    fused = torch.cat([tensor.detach() for _name, tensor in ordered], dim=0)
    fused_qweight, fused_scale = blockwise_cast_to_fp8(fused, config.weight_block_size, config.power_2_scale)

    offset = 0
    for name, tensor in ordered:
        rows = tensor.shape[0]
        yield name, fused_qweight[offset : offset + rows].contiguous()
        yield scale_name_for_weight(name), fused_scale.to(config.scale_dtype)
        offset += rows


def iter_serialized_fp8_metadata(
    name: str,
    shape: Sequence[int],
    dtype: torch.dtype,
    config: SerializedFp8Config,
) -> Iterable[tuple[str, torch.dtype, list[int]]]:
    """Return metadata matching ``iter_serialized_fp8_tensors`` without quantizing."""

    if len(shape) == 3 and batched_moe_expert_spec(name) is not None:
        yield from iter_batched_moe_expert_metadata(name, shape, config)
        return

    if is_quantizable_weight_shape(name, shape):
        yield name, config.quantized_weight_dtype, list(shape)
        yield (
            scale_name_for_weight(name),
            config.scale_dtype,
            blockwise_fp8_scale_shape(shape, config.weight_block_size),
        )
    else:
        yield name, dtype, list(shape)
