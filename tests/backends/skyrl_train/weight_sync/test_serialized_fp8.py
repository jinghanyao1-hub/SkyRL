from types import SimpleNamespace

import torch

from skyrl.backends.skyrl_train.weight_sync.base import WeightChunk
from skyrl.backends.skyrl_train.weight_sync.cuda_ipc_strategy import _iter_single_dtype_chunks
from skyrl.backends.skyrl_train.weight_sync.serialized_fp8 import (
    SerializedFp8Config,
    batched_moe_expert_spec,
    blockwise_cast_to_fp8,
    blockwise_fp8_scale_shape,
    get_qwen35_slime_parity_modules_to_not_convert,
    get_serialized_fp8_quantization_config,
    is_quantizable_weight,
    is_quantizable_weight_shape,
    iter_shared_blockwise_fp8_tensors,
    iter_serialized_fp8_metadata,
    iter_serialized_fp8_tensors,
    scale_name_for_weight,
    shared_blockwise_fp8_group,
)


def test_blockwise_fp8_scale_shape_uses_ceil_blocks():
    assert blockwise_fp8_scale_shape([257, 513], [128, 128]) == [3, 5]


def test_blockwise_cast_to_fp8_emits_weight_and_fp32_scale():
    weight = torch.arange(257 * 129, dtype=torch.float32).reshape(257, 129) / 1000

    q_weight, scale = blockwise_cast_to_fp8(weight, [128, 128])

    assert q_weight.shape == weight.shape
    assert q_weight.dtype == torch.float8_e4m3fn
    assert scale.shape == (3, 2)
    assert scale.dtype == torch.float32


def test_blockwise_cast_pow2_scales_match_te_and_sglang_ue8m0_rule():
    # Megatron-TE Float8BlockScaling (power_2_scale=True) and SGLang/DeepGEMM
    # ceil_to_ue8m0 both round the dequant scale up to the next power of two.
    # The serialized rollout scale must use the same rule so the vLLM rollout
    # weights match the Megatron train-forward weights block-for-block.
    torch.manual_seed(0)
    weight = torch.randn(256, 384, dtype=torch.float32)

    _, pow2_scale = blockwise_cast_to_fp8(weight, [128, 128], power_2_scale=True)
    _, exact_scale = blockwise_cast_to_fp8(weight, [128, 128], power_2_scale=False)

    # Every pow2 scale is an exact power of two.
    log2 = torch.log2(pow2_scale)
    assert torch.allclose(log2, log2.round(), atol=0.0)
    # And equals ceil-to-pow2 of the exact FP32 scale.
    expected = torch.pow(2.0, torch.ceil(torch.log2(exact_scale)))
    assert torch.allclose(pow2_scale, expected)
    # Rounding is upward, so it never causes FP8 overflow (scale >= exact scale).
    assert torch.all(pow2_scale >= exact_scale)


def test_blockwise_cast_pow2_rollout_matches_pow2_train_bitwise():
    # If both the rollout sync and the training forward quantize the same master
    # weight with power-of-2 scales, the effective (dequantized) weights are
    # identical. This is the parity the fix restores versus exact FP32 scales.
    torch.manual_seed(1)
    weight = torch.randn(256, 256, dtype=torch.bfloat16)

    q_rollout, s_rollout = blockwise_cast_to_fp8(weight, [128, 128], power_2_scale=True)
    q_train, s_train = blockwise_cast_to_fp8(weight, [128, 128], power_2_scale=True)

    assert torch.equal(q_rollout.view(torch.uint8), q_train.view(torch.uint8))
    assert torch.equal(s_rollout, s_train)


def test_serialized_config_power_2_scale_follows_te_env(monkeypatch):
    monkeypatch.delenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", raising=False)
    assert SerializedFp8Config().power_2_scale is True

    monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")
    assert SerializedFp8Config().power_2_scale is False

    monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "0")
    assert SerializedFp8Config().power_2_scale is True


def test_quantizable_weight_filter_keeps_embeddings_in_target_dtype():
    config = SerializedFp8Config()
    linear = torch.ones((256, 256), dtype=torch.bfloat16)
    embedding = torch.ones((32000, 256), dtype=torch.bfloat16)

    assert is_quantizable_weight("model.layers.0.mlp.down_proj.weight", linear)
    assert is_quantizable_weight("model.layers.0.linear_attn.in_proj_qkv.weight", linear)
    assert is_quantizable_weight("model.layers.0.linear_attn.in_proj_z.weight", linear)
    assert is_quantizable_weight("model.layers.0.linear_attn.out_proj.weight", linear)
    assert not is_quantizable_weight("model.layers.0.linear_attn.conv1d.weight", linear)
    assert not is_quantizable_weight("model.layers.0.linear_attn.in_proj_b.weight", linear)
    assert not is_quantizable_weight("model.layers.0.linear_attn.in_proj_a.weight", linear)
    assert not is_quantizable_weight("model.embed_tokens.weight", embedding)

    tensors = list(
        iter_serialized_fp8_tensors(
            "model.embed_tokens.weight",
            embedding,
            torch.bfloat16,
            config,
        )
    )
    assert [(name, tensor.dtype) for name, tensor in tensors] == [("model.embed_tokens.weight", torch.bfloat16)]


def test_serialized_fp8_metadata_matches_tensor_emission():
    config = SerializedFp8Config(weight_block_size=(128, 128))
    name = "model.layers.0.self_attn.q_proj.weight"
    tensor = torch.ones((257, 129), dtype=torch.bfloat16)

    metadata = list(iter_serialized_fp8_metadata(name, tensor.shape, torch.bfloat16, config))
    emitted = list(iter_serialized_fp8_tensors(name, tensor, torch.bfloat16, config))

    assert metadata == [
        (name, torch.float8_e4m3fn, [257, 129]),
        (scale_name_for_weight(name), torch.float32, [3, 2]),
    ]
    assert [(out_name, out_tensor.dtype, list(out_tensor.shape)) for out_name, out_tensor in emitted] == [
        (name, torch.float8_e4m3fn, [257, 129]),
        (scale_name_for_weight(name), torch.float32, [3, 2]),
    ]


def test_vllm_serialized_fp8_quantization_config():
    assert get_serialized_fp8_quantization_config() == {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }
    assert get_serialized_fp8_quantization_config(
        modules_to_not_convert=["model.layers.0.linear_attn.in_proj_b"]
    ) == {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "ignored_layers": ["model.layers.0.linear_attn.in_proj_b"],
        "modules_to_not_convert": ["model.layers.0.linear_attn.in_proj_b"],
    }


def test_qwen35_gdn_ba_stays_unquantized_to_match_slime_policy():
    config = SerializedFp8Config(weight_block_size=(128, 128))
    b_name = "model.layers.1.linear_attn.in_proj_b.weight"
    a_name = "model.layers.1.linear_attn.in_proj_a.weight"
    b_weight = torch.full((64, 256), 1.0, dtype=torch.bfloat16)
    a_weight = torch.full((64, 256), 10.0, dtype=torch.bfloat16)

    assert shared_blockwise_fp8_group(b_name) == ("model.layers.1.linear_attn", "b")
    assert shared_blockwise_fp8_group(a_name) == ("model.layers.1.linear_attn", "a")

    emitted = list(
        iter_shared_blockwise_fp8_tensors(
            {
                "b": (b_name, b_weight),
                "a": (a_name, a_weight),
            },
            torch.bfloat16,
            config,
        )
    )

    emitted_by_name = {name: tensor for name, tensor in emitted}

    assert emitted_by_name[b_name].dtype == torch.bfloat16
    assert emitted_by_name[a_name].dtype == torch.bfloat16
    assert scale_name_for_weight(b_name) not in emitted_by_name
    assert scale_name_for_weight(a_name) not in emitted_by_name


def test_qwen35_slime_parity_modules_to_not_convert_uses_linear_attention_layers():
    hf_config = SimpleNamespace(
        model_type="qwen3_5_text",
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
        ],
    )

    assert get_qwen35_slime_parity_modules_to_not_convert(hf_config) == [
        "model.layers.0.linear_attn.in_proj_b",
        "model.layers.0.linear_attn.in_proj_a",
        "model.language_model.layers.0.linear_attn.in_proj_b",
        "model.language_model.layers.0.linear_attn.in_proj_a",
        "language_model.model.layers.0.linear_attn.in_proj_b",
        "language_model.model.layers.0.linear_attn.in_proj_a",
        "model.language_model.model.layers.0.linear_attn.in_proj_b",
        "model.language_model.model.layers.0.linear_attn.in_proj_a",
        "model.layers.2.linear_attn.in_proj_b",
        "model.layers.2.linear_attn.in_proj_a",
        "model.language_model.layers.2.linear_attn.in_proj_b",
        "model.language_model.layers.2.linear_attn.in_proj_a",
        "language_model.model.layers.2.linear_attn.in_proj_b",
        "language_model.model.layers.2.linear_attn.in_proj_a",
        "model.language_model.model.layers.2.linear_attn.in_proj_b",
        "model.language_model.model.layers.2.linear_attn.in_proj_a",
    ]


def test_moe_batched_expert_spec_recognizes_and_splits_gate_up():
    base = "model.language_model.layers.5.mlp.experts"
    assert batched_moe_expert_spec(f"{base}.gate_up_proj") == (base, ("gate_proj", "up_proj"), True)
    assert batched_moe_expert_spec(f"{base}.down_proj") == (base, ("down_proj",), False)
    assert batched_moe_expert_spec("model.layers.5.mlp.gate.weight") is None
    assert batched_moe_expert_spec("model.layers.5.self_attn.q_proj.weight") is None


def test_moe_shared_expert_quantized_router_and_gate_bf16():
    lin = (256, 256)
    # shared expert linears are FP8 (vLLM builds them with quant_config)
    assert is_quantizable_weight_shape("model.layers.3.mlp.shared_expert.gate_proj.weight", lin)
    assert is_quantizable_weight_shape("model.layers.3.mlp.shared_expert.up_proj.weight", lin)
    assert is_quantizable_weight_shape("model.layers.3.mlp.shared_expert.down_proj.weight", lin)
    # router + shared_expert_gate stay BF16 (vLLM builds them with quant_config=None)
    assert not is_quantizable_weight_shape("model.layers.3.mlp.gate.weight", (256, 8))
    assert not is_quantizable_weight_shape("model.layers.3.mlp.shared_expert_gate.weight", (256, 1))


def test_batched_moe_experts_unbatch_to_per_expert_fp8_with_pow2_scales():
    # gate_up_proj: [E, 2*moe_inter, hidden]; down_proj: [E, hidden, moe_inter]
    E, moe_inter, hidden = 3, 128, 256
    config = SerializedFp8Config(weight_block_size=(128, 128))
    base = "model.language_model.layers.7.mlp.experts"

    torch.manual_seed(0)
    gate_up = torch.randn(E, 2 * moe_inter, hidden, dtype=torch.bfloat16)
    emitted = list(iter_serialized_fp8_tensors(f"{base}.gate_up_proj", gate_up, torch.bfloat16, config))
    by_name = dict(emitted)
    # per-expert gate_proj + up_proj weights and scales, with the names vLLM's
    # make_expert_params_mapping matches ("experts.N.gate_proj." prefix).
    for e in range(E):
        for proj in ("gate_proj", "up_proj"):
            w = by_name[f"{base}.{e}.{proj}.weight"]
            s = by_name[f"{base}.{e}.{proj}.weight_scale_inv"]
            assert w.dtype == torch.float8_e4m3fn and tuple(w.shape) == (moe_inter, hidden)
            assert s.dtype == torch.float32 and tuple(s.shape) == (1, 2)  # ceil(128/128), ceil(256/128)
            log2 = torch.log2(s)
            assert torch.allclose(log2, log2.round(), atol=0.0)  # power-of-2 scales

    down = torch.randn(E, hidden, moe_inter, dtype=torch.bfloat16)
    emitted_d = dict(iter_serialized_fp8_tensors(f"{base}.down_proj", down, torch.bfloat16, config))
    for e in range(E):
        w = emitted_d[f"{base}.{e}.down_proj.weight"]
        assert w.dtype == torch.float8_e4m3fn and tuple(w.shape) == (hidden, moe_inter)
        assert f"{base}.{e}.down_proj.weight_scale_inv" in emitted_d


def test_batched_moe_metadata_matches_tensor_emission():
    E, moe_inter, hidden = 4, 128, 256
    config = SerializedFp8Config(weight_block_size=(128, 128))
    base = "model.language_model.layers.2.mlp.experts"
    for suffix, shape in (
        ("gate_up_proj", (E, 2 * moe_inter, hidden)),
        ("down_proj", (E, hidden, moe_inter)),
    ):
        name = f"{base}.{suffix}"
        tensor = torch.randn(*shape, dtype=torch.bfloat16)
        meta = list(iter_serialized_fp8_metadata(name, list(shape), torch.bfloat16, config))
        emitted = [(n, t.dtype, list(t.shape)) for n, t in iter_serialized_fp8_tensors(name, tensor, torch.bfloat16, config)]
        assert meta == emitted


def test_cuda_ipc_chunks_are_split_by_actual_dtype_in_first_seen_order():
    chunk = WeightChunk(
        names=["w0", "s0", "w1", "b0"],
        dtypes=["ignored", "ignored", "ignored", "ignored"],
        shapes=[[4], [1], [8], [2]],
        tensors=[
            torch.empty((4,), dtype=torch.float8_e4m3fn),
            torch.ones((1,), dtype=torch.float32),
            torch.empty((8,), dtype=torch.float8_e4m3fn),
            torch.ones((2,), dtype=torch.bfloat16),
        ],
    )

    chunks = list(_iter_single_dtype_chunks(chunk))

    assert [subchunk.names for subchunk in chunks] == [["w0", "w1"], ["s0"], ["b0"]]
    assert [[tensor.dtype for tensor in subchunk.tensors] for subchunk in chunks] == [
        [torch.float8_e4m3fn, torch.float8_e4m3fn],
        [torch.float32],
        [torch.bfloat16],
    ]
