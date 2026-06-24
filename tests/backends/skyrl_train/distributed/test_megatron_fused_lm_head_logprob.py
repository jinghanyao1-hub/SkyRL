import os

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("megatron.core.parallel_state")

from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (  # noqa: E402
    from_fused_lm_head_to_logprobs_packed_sequences,
    from_parallel_logits_to_logprobs_packed_sequences,
)


@pytest.fixture(scope="module", autouse=True)
def single_rank_dist(tmp_path_factory):
    initialized_here = False
    if not dist.is_initialized():
        init_dir = tmp_path_factory.mktemp("dist_init")
        init_file = os.path.join(init_dir, "shared_init")
        dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=0, world_size=1)
        initialized_here = True
    yield
    if initialized_here:
        dist.destroy_process_group()


def _packed_inputs():
    torch.manual_seed(1234)
    hidden = torch.randn(1, 7, 5, dtype=torch.float32)
    weight = torch.randn(11, 5, dtype=torch.float32)
    packed_targets = torch.tensor([[2, 4, 3, 1, 8, 6, 5]], dtype=torch.long)
    cu_seqlens_padded = torch.tensor([0, 4, 7], dtype=torch.long)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ],
        dtype=torch.bool,
    )
    return hidden, weight, packed_targets, cu_seqlens_padded, attention_mask


def _packed_logprobs_from_logits(hidden, weight, packed_targets, cu_seqlens_padded, attention_mask):
    logits = torch.matmul(hidden, weight.t())
    return from_parallel_logits_to_logprobs_packed_sequences(
        logits,
        packed_targets,
        cu_seqlens_padded,
        unpacked_seqlen=4,
        vocab_start_index=0,
        vocab_end_index=weight.shape[0],
        group=dist.group.WORLD,
        inference_only=False,
        cp_group=None,
        chunk_size=2,
        attention_mask=attention_mask,
    )


def _packed_logprobs_fused(hidden, weight, packed_targets, cu_seqlens_padded, attention_mask):
    return from_fused_lm_head_to_logprobs_packed_sequences(
        hidden,
        weight,
        packed_targets,
        cu_seqlens_padded,
        unpacked_seqlen=4,
        vocab_start_index=0,
        vocab_end_index=weight.shape[0],
        group=dist.group.WORLD,
        chunk_size=2,
        temperature=1.0,
        cp_group=None,
        attention_mask=attention_mask,
    )


def test_fused_lm_head_packed_logprobs_match_logits_path():
    hidden, weight, packed_targets, cu_seqlens_padded, attention_mask = _packed_inputs()

    ref = _packed_logprobs_from_logits(hidden, weight, packed_targets, cu_seqlens_padded, attention_mask)
    fused = _packed_logprobs_fused(hidden, weight, packed_targets, cu_seqlens_padded, attention_mask)

    torch.testing.assert_close(fused, ref, rtol=1e-6, atol=1e-6)


def test_fused_lm_head_packed_backward_matches_logits_path():
    hidden, weight, packed_targets, cu_seqlens_padded, attention_mask = _packed_inputs()
    upstream = torch.randn(2, 3, dtype=torch.float32)

    hidden_ref = hidden.clone().requires_grad_(True)
    weight_ref = weight.clone().requires_grad_(True)
    ref = _packed_logprobs_from_logits(hidden_ref, weight_ref, packed_targets, cu_seqlens_padded, attention_mask)
    (ref * upstream).sum().backward()

    hidden_fused = hidden.clone().requires_grad_(True)
    weight_fused = weight.clone().requires_grad_(True)
    fused = _packed_logprobs_fused(hidden_fused, weight_fused, packed_targets, cu_seqlens_padded, attention_mask)
    (fused * upstream).sum().backward()

    torch.testing.assert_close(fused, ref, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(hidden_fused.grad, hidden_ref.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(weight_fused.grad, weight_ref.grad, rtol=1e-5, atol=1e-6)


def test_fused_lm_head_accumulates_megatron_main_grad():
    hidden, weight, packed_targets, cu_seqlens_padded, attention_mask = _packed_inputs()
    upstream = torch.randn(2, 3, dtype=torch.float32)

    hidden_ref = hidden.clone().requires_grad_(True)
    weight_ref = weight.clone().requires_grad_(True)
    ref = _packed_logprobs_from_logits(hidden_ref, weight_ref, packed_targets, cu_seqlens_padded, attention_mask)
    (ref * upstream).sum().backward()

    hidden_fused = hidden.clone().requires_grad_(True)
    weight_fused = weight.clone().requires_grad_(True)
    weight_fused.main_grad = torch.zeros_like(weight_fused)
    weight_fused.grad_added_to_main_grad = False
    weight_fused.zero_out_wgrad = True

    fused = _packed_logprobs_fused(hidden_fused, weight_fused, packed_targets, cu_seqlens_padded, attention_mask)
    (fused * upstream).sum().backward()

    torch.testing.assert_close(hidden_fused.grad, hidden_ref.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(weight_fused.main_grad, weight_ref.grad, rtol=1e-5, atol=1e-6)
    assert weight_fused.grad_added_to_main_grad
