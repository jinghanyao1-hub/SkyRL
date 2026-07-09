# Local FP8 Patch Port Conflict Analysis

Date: 2026-07-09

## Scope

This document describes the expected conflicts when porting the local SkyRL FP8/H100 patch stack from:

```text
/home/ray/default/model_quality_measure/weight_sync_for_fp8_pr/SkyRL
```

onto the latest open-source SkyRL main clone at:

```text
/home/ray/default/model_quality_measure/weight_sync_for_fp8_pr/PR_ready/SkyRL
```

The upstream target inspected here is:

```text
NovaSky-AI/SkyRL@1ab51f1b965c0a3a6fa7dfd4e965758525bfd89b
```

The local patch stack is described by:

```text
/home/ray/default/model_quality_measure/weight_sync_for_fp8_pr/docs/big_pr.md
```

## Summary

The local stack cannot be applied wholesale to latest upstream main. A dry-run `git apply --3way --check` showed conflicts in the files where upstream has added SFT/VLM packing support, profiler support, Triton fused LM-head support, and an upstream FP8-safe packing helper.

The patch should be ported feature by feature:

1. Serialized blockwise FP8 weight sync.
2. Transformer Engine FP8 amax floor.
3. H100 colocated memory relief.
4. TP greater than 1 local-128 FP8 token alignment.
5. Tests and docs for the above.

Several local changes are not documented in `big_pr.md` and should not be included in this PR unless explicitly intended.

## Dry-Run Conflict Set

The following files conflicted in a dry-run three-way patch apply:

```text
skyrl/backends/skyrl_train/distributed/megatron/megatron_utils.py
skyrl/backends/skyrl_train/distributed/megatron/packing_utils.py
skyrl/backends/skyrl_train/utils/replay_utils.py
skyrl/backends/skyrl_train/workers/megatron/megatron_model_wrapper.py
skyrl/backends/skyrl_train/workers/megatron/megatron_worker.py
skyrl/train/config/config.py
skyrl/train/dataset/collators.py
skyrl/train/sft_trainer.py
skyrl/train/trainer.py
tests/train/test_config.py
tests/train/test_sft_packing_collate.py
```

Many other files apply cleanly, but they still need review because they interact with the conflicted files.

## PR 1 and PR 3: Serialized Blockwise FP8 Weight Sync

### Local Intent

The local stack adds checkpoint-format blockwise FP8 weight sync:

- Megatron exports HF/vLLM weights.
- SkyRL quantizes selected Qwen3.5 linear weights to FP8 e4m3.
- SkyRL emits `.weight` plus `.weight_scale_inv` tensors.
- vLLM is initialized with FP8 quantization and dummy load format.
- CUDA IPC handles mixed dtype chunks by splitting one logical checkpoint update into same-dtype packed buffers.

### Expected Clean Areas

These pieces are mostly clean to port:

```text
skyrl/backends/skyrl_train/weight_sync/serialized_fp8.py
skyrl/backends/skyrl_train/weight_sync/cuda_ipc_strategy.py
skyrl/backends/skyrl_train/inference_servers/utils.py
tests/backends/skyrl_train/weight_sync/test_serialized_fp8.py
tests/backends/skyrl_train/inference_servers/test_build_vllm_cli_args.py
```

### Conflict Areas

`megatron_worker.py` conflicts because upstream has changed the Megatron worker around profiler, VLM, and MoE LoRA sync. The local FP8 extractor changes must be added without discarding upstream behavior.

Porting requirements:

- Preserve upstream VLM support.
- Preserve upstream fused-MoE expert LoRA adapter sync.
- Add local `MegatronWeightExtractor(..., fp8_weight_sync_mode=...)`.
- Add serialized FP8 metadata and tensor emission.
- Thread `inference_engine_cfg.fp8_weight_sync_mode` into the extractor.

## PR 2: Transformer Engine FP8 Amax Floor

### Local Intent

The local stack adds:

```text
skyrl/backends/skyrl_train/workers/megatron/_fp8_block_amax_epsilon_patch.py
```

and applies it early in Megatron worker processes when:

```bash
NVTE_FP8_BLOCK_AMAX_EPSILON=1e-4
```

This prevents zero-amax pow-2 FP8 blocks from producing huge gradients and `grad_norm=inf` in the 35B-A3B MoE runs.

### Conflict Risk

The patch file itself is low-risk because upstream has no equivalent file. The integration point is `megatron_worker.py`, which is already conflicted for other reasons.

Porting requirements:

- Import `apply_fp8_block_amax_epsilon_patch`.
- Call it at module import time.
- Re-apply it before Megatron model construction in `init_model`.
- Forward `NVTE_FP8_BLOCK_AMAX_EPSILON` to Ray workers and vLLM engine runtime envs where needed.

## PR 4: H100 Colocated Memory Relief

### Local Intent

The local H100 memory patch reduces HBM pressure when policy, ref, and vLLM are colocated on 80GB H100 GPUs.

Important local surfaces:

```text
skyrl/train/config/config.py
skyrl/train/trainer.py
skyrl/backends/skyrl_train/workers/worker.py
skyrl/backends/skyrl_train/workers/worker_dispatch.py
skyrl/backends/skyrl_train/workers/megatron/megatron_worker.py
skyrl/backends/skyrl_train/distributed/megatron/model_utils.py
skyrl/backends/skyrl_train/inference_servers/remote_inference_client.py
skyrl/backends/skyrl_train/inference_servers/vllm_server_actor.py
```

Key behaviors:

- Sleep colocated vLLM engines before training.
- Reset prefix cache before sleep when configured.
- Add vLLM `/cuda_memory_stats` and `/release_cuda_memory` endpoints.
- Return worker CUDA/NVML memory stats from `empty_cache`.
- Enforce inactive worker residual-HBM barriers.
- Hard-evict inactive ref workers when configured.
- Avoid moving the full policy DP shard to HBM when `SKYRL_CPU_RESIDENT_POLICY_MICROBATCH=1`.
- Add vocab entropy chunking with `vocab_entropy_chunk_size` and `vocab_entropy_chunk_memory_mb`.

### Conflict With Upstream Profiler

Latest upstream added torch profiler support in:

```text
skyrl/train/trainer.py
skyrl/backends/skyrl_train/workers/worker.py
skyrl/backends/skyrl_train/workers/worker_dispatch.py
skyrl/backends/skyrl_train/workers/megatron/megatron_worker.py
skyrl/backends/skyrl_train/utils/profiler.py
```

Porting requirements:

- Preserve upstream profiler start, step, stop, and dispatch methods.
- Preserve upstream vLLM metrics scraper lifecycle.
- Replace plain `inference_engine_client.sleep()` with the local `_sleep_inference_engine_for_training()` helper.
- Keep local worker memory stat returns compatible with upstream profiler RPCs.
- Keep upstream VLM data handling in Megatron worker while adding CPU-resident microbatch behavior.

### Conflict With Upstream Triton Fused LM-Head

Latest upstream added:

```text
skyrl/backends/skyrl_train/distributed/megatron/fused_linear_logprob_triton.py
trainer.fused_lm_head_logprob_backend
```

The local stack adds vocab entropy chunking near the same config and model utility code.

Porting requirements:

- Keep upstream `fused_lm_head_logprob_backend`.
- Add local `vocab_entropy_chunk_size`.
- Add local `vocab_entropy_chunk_memory_mb`.
- Thread chunk settings into entropy paths without removing Triton backend support.

## PR 5: TP Greater Than 1 FP8 Local-128 Token Alignment

### Local Intent

The local H100 TP2 FP8 failure was:

```text
AssertionError: All-gather requires quantizable tensor for quantizer Float8BlockQuantizer
```

Local root cause:

- TP greater than 1 enables Megatron sequence parallelism.
- TE blockwise FP8 validates the local all-gather source tensor.
- `Float8BlockQuantizer` requires the local flattened token dimension to be divisible by 128.
- Aligning only the global sequence length to 128 is insufficient for TP2; global alignment must include the TP factor.

Local rule:

```python
if fp8_enabled:
    fp8_token_align = 128 * tp_size * cp_size if tp_size > 1 else 16 * cp_size
    layout_align = math.lcm(layout_align, fp8_token_align)
```

### Conflict With Upstream PR #1828

Latest upstream already added:

```text
skyrl/backends/skyrl_train/distributed/megatron/packing_utils.py
```

but upstream's FP8 alignment is still:

```python
math.lcm(layout_align, 16 * cp_size)
```

and upstream adds a separate:

```python
get_unpacked_seq_align_size(tp_size, fp8_enabled=False)
```

Porting requirements:

- Do not replace upstream `packing_utils.py` wholesale.
- Keep upstream `get_unpacked_seq_align_size`.
- Change upstream FP8 packed alignment so TP greater than 1 uses local-128 alignment.
- Change unpacked FP8 alignment so TP greater than 1 pads enough for local 128-token sequence-parallel shards.
- Preserve upstream SFT/VLM packing behavior and tests.
- Add local H100 regression tests for `8552 -> 8704` and `8594 -> 8704` under TP2 FP8.

## Changes Not Covered By `big_pr.md`

The following local changes are not part of the five documented PRs. They should not be ported as part of the FP8/H100 PR unless explicitly requested.

### `use_current_policy_logprobs_as_old`

Files:

```text
skyrl/train/config/config.py
skyrl/backends/skyrl_train/utils/ppo_utils.py
skyrl/backends/skyrl_train/workers/worker.py
skyrl/backends/skyrl_train/workers/megatron/megatron_model_wrapper.py
skyrl/train/trainer.py
tests/train/algorithms/test_skip_fwd_logprobs.py
tests/train/test_config.py
```

This changes PPO denominator logprob behavior by using the current train-forward logprobs as detached old logprobs. It is a Slime-parity behavior change, not part of the FP8 weight-sync, memory, or TP alignment stack.

### Ray Node Resource Placement Pinning

Files:

```text
skyrl/train/utils/utils.py
skyrl/train/entrypoints/main_base.py
skyrl/train/trainer.py
skyrl/backends/skyrl_train/workers/worker.py
skyrl/backends/skyrl_train/inference_servers/server_group.py
skyrl/backends/skyrl_train/inference_servers/setup.py
```

This adds `SKYRL_NODE_RESOURCE` placement-group pinning. It is operational cluster placement plumbing, not documented in `big_pr.md`.

### Generate Endpoint Forcing And Logprob Sanitization

Files:

```text
skyrl/backends/skyrl_train/inference_servers/remote_inference_client.py
skyrl/backends/skyrl_train/inference_servers/vllm_server_actor.py
```

This adds `SKYRL_USE_SKYRL_GENERATE_ENDPOINT` and clamps non-finite logprobs to `-9999.0`. It may be useful, but it is not documented in `big_pr.md`.

### Diagnostic Grad And HBM Profiling

Files:

```text
skyrl/backends/skyrl_train/workers/megatron/megatron_worker.py
skyrl/train/utils/utils.py
```

This includes `SKYRL_DEBUG_GRAD_SCAN` and `SKYRL_HBM_PROFILE`. These are investigation aids and should be separated from the core PR unless desired.

### Broad Runtime Environment Forwarding

Files:

```text
skyrl/train/utils/utils.py
skyrl/backends/skyrl_train/inference_servers/engine_utils.py
```

This forwards `PATH`, CUDA library paths, `PYTORCH_CUDA_ALLOC_CONF`, and skip-peer-access knobs. Some env forwarding is needed for H100 operation, but the broad set is not described in `big_pr.md`.

## Recommended Port Order

1. Port serialized FP8 support and its tests.
2. Port amax epsilon patch and env forwarding.
3. Port H100 memory relief, preserving upstream profiler and metrics code.
4. Port TP greater than 1 local-128 alignment by modifying upstream packing helpers rather than replacing them.
5. Add only tests corresponding to the five documented PRs.
6. Keep unrelated local changes out unless a separate PR/document explicitly covers them.

## Validation Targets

After the port, run at minimum:

```bash
uv run --extra dev --extra ray -- pytest -q \
  tests/backends/skyrl_train/weight_sync/test_serialized_fp8.py \
  tests/backends/skyrl_train/inference_servers/test_build_vllm_cli_args.py \
  tests/backends/skyrl_train/distributed/test_preprocess_packed_seqs_cp.py \
  tests/train/test_sft_packing_collate.py
```

Then validate on H100:

- Qwen3.5-4B BF16 and FP8 colocated runs with the memory stack.
- Qwen3.5-9B FP8 TP2/DP4/vLLM TP1x8 run for the local-128 alignment.
- Matching BF16 TP2/DP4/vLLM TP1x8 run for comparison.
