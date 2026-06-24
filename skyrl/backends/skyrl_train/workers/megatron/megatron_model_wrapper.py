from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import megatron.core.parallel_state as mpu
import torch
import torch.nn as nn
from megatron.core.distributed import finalize_model_grads
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.utils import get_attr_wrapped_model
from omegaconf import OmegaConf

from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
    get_model_config,
    make_batch_generator,
    model_packs_sequences_internally,
    preprocess_packed_seqs,
    recover_left_padding,
    remove_left_padding,
)
from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
    from_fused_lm_head_to_logprobs_packed_sequences,
    from_parallel_logits_to_logprobs,
    from_parallel_logits_to_logprobs_packed_sequences,
    vocab_parallel_entropy_from_fused_lm_head_packed_sequences,
    vocab_parallel_entropy,
    vocab_parallel_entropy_packed_sequences,
)
from skyrl.backends.skyrl_train.utils.ppo_utils import (
    PolicyLossRegistry,
    compute_approx_kl,
)
from skyrl.backends.skyrl_train.utils.replay_utils import (
    setup_per_microbatch_replay_backward,
    setup_per_microbatch_replay_forward,
)
from skyrl.backends.skyrl_train.utils.torch_utils import masked_mean
from skyrl.backends.skyrl_train.workers.worker_utils import (
    compute_minibatch_rollout_logprob_diff_metrics,
)
from skyrl.train.config import TrainerConfig


def _build_packed_targets(
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    packed_seq_params,
    sub_seq_lengths: Optional[list[list[int]]] = None,
) -> torch.Tensor:
    """Pack full target token IDs without context-parallel sharding."""
    cu_padded = packed_seq_params.cu_seqlens_q_padded.to(device=sequences.device, dtype=torch.long)
    total_padded_tokens = int(cu_padded[-1].item())

    targets = torch.zeros((total_padded_tokens,), dtype=sequences.dtype, device=sequences.device)
    if sub_seq_lengths is not None:
        cu_padded_cpu = cu_padded.detach().cpu().tolist()
        seg_idx = 0
        for row_idx, row_lens in enumerate(sub_seq_lengths):
            row_offset = 0
            for seq_len in row_lens:
                seq_len = int(seq_len)
                if seg_idx + 1 >= len(cu_padded_cpu):
                    raise ValueError("sub_seq_lengths contains more sub-sequences than packed_seq_params")
                packed_start = cu_padded_cpu[seg_idx]
                targets[packed_start : packed_start + seq_len] = sequences[row_idx, row_offset : row_offset + seq_len]
                row_offset += cu_padded_cpu[seg_idx + 1] - cu_padded_cpu[seg_idx]
                seg_idx += 1
        if seg_idx != len(cu_padded_cpu) - 1:
            raise ValueError(
                f"sub_seq_lengths describes {seg_idx} sub-sequences, "
                f"but packed_seq_params describes {len(cu_padded_cpu) - 1}"
            )
        return targets.unsqueeze(0)

    attention_mask = attention_mask.to(device=sequences.device, dtype=torch.bool)
    token_offsets = attention_mask.to(torch.long).cumsum(dim=1) - 1
    packed_indices = cu_padded[:-1].unsqueeze(1) + token_offsets
    targets[packed_indices[attention_mask]] = sequences[attention_mask]
    return targets.unsqueeze(0)


def _copy_tensor_dict_to_device(batch: Dict[str, Any], device: int) -> Dict[str, Any]:
    """Copy tensor values to device without mutating the CPU microbatch cache."""
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _unwrap_model(model: nn.Module) -> nn.Module:
    while hasattr(model, "module"):
        model = model.module
    return model


def _iter_module_chain(model: nn.Module):
    seen = set()
    stack = [model]
    while stack:
        module = stack.pop()
        if id(module) in seen:
            continue
        seen.add(id(module))
        yield module

        child = getattr(module, "module", None)
        if isinstance(child, nn.Module):
            stack.append(child)
        elif isinstance(child, (list, tuple)):
            stack.extend(item for item in child if isinstance(item, nn.Module))

        stack.extend(module.children())


def _get_wrapped_attr(model: nn.Module, attr: str, allow_none: bool = True) -> Any:
    try:
        return get_attr_wrapped_model(model, attr, allow_none=allow_none)
    except Exception:
        if allow_none:
            return None
        raise


@contextmanager
def _temporary_post_process(model: nn.Module, post_process: bool):
    targets = []
    for module in _iter_module_chain(model):
        if hasattr(module, "post_process"):
            targets.append((module, module.post_process))

    if not targets:
        yield
        return

    for module, _old_post_process in targets:
        module.post_process = post_process
    try:
        yield
    finally:
        for module, old_post_process in targets:
            module.post_process = old_post_process


class MegatronModelWrapper:
    def __init__(
        self,
        config: TrainerConfig,
        actor_module: List[nn.Module],
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
        policy_loss_fn: Optional[Callable] = None,
    ):
        self.cfg = config
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.policy_loss_fn = policy_loss_fn
        self.remove_microbatch_padding = self.cfg.remove_microbatch_padding
        # Some models (e.g. Qwen3.5 via the VL bridge -> Qwen3VLModel) pack
        # sequences inside their own forward; SkyRL sample packing would then
        # double-pack and corrupt the GDN cu_seqlens, so refuse it. For Qwen3.5,
        # use language_model_only=True (native GPTModel GDN path) to pack.
        if self.remove_microbatch_padding and model_packs_sequences_internally(self.actor_module):
            raise ValueError(
                "remove_microbatch_padding=True (sample packing) is not supported for models that "
                "pack sequences inside their own forward (e.g. the Qwen3.5 VL Qwen3VLModel): it "
                "double-packs and corrupts the GatedDeltaNet cu_seqlens. Set "
                "trainer.policy.language_model_only=True to route Qwen3.5 to the native GPTModel GDN "
                "packing path, or set trainer.remove_microbatch_padding=False."
            )

        config = get_model_config(self.actor_module[0])
        # This is set to None by default: https://github.com/NVIDIA/Megatron-LM/blob/07b22a05136a3cb08ece05f7de38cf6aeeb165fb/megatron/core/model_parallel_config.py#L95
        # use the build in finalize_model_grads function to all reduce gradients across parallelism dimensions
        config.finalize_model_grads_func = finalize_model_grads
        # Wire up the optimizer's loss scaler so Megatron's pipeline schedule can scale
        # the loss before backward (critical for fp16 dynamic loss scaling, MoE aux loss
        # scaling, and any explicit loss_scale configuration).
        if actor_optimizer is not None:
            config.grad_scale_func = actor_optimizer.scale_loss
        self._fused_lm_head_status_logged = False
        print(
            f"[skyrl] MegatronModelWrapper init fused_lm_head_logprob={self.cfg.fused_lm_head_logprob}",
            flush=True,
        )

    def _fused_lm_head_fallback_reason(
        self,
        model: nn.Module,
        loss_name: str,
        loss_config: Any,
        forward_only: bool,
    ) -> Optional[str]:
        if not self.cfg.fused_lm_head_logprob:
            return "disabled"
        if forward_only:
            return "forward_only"
        if loss_name == "cross_entropy":
            return "cross_entropy_loss"
        if not self.remove_microbatch_padding:
            return "requires_remove_microbatch_padding"
        if loss_config.use_entropy_loss:
            return "entropy_loss_requires_logits"
        if self.cfg.logprobs_chunk_size is None:
            return "requires_logprobs_chunk_size"
        if not mpu.is_pipeline_last_stage(ignore_virtual=True):
            return None
        if mpu.get_tensor_model_parallel_world_size() != 1:
            return "tp_world_size_not_1"

        if _get_wrapped_attr(model, "post_process", allow_none=True) is None:
            return "missing_post_process"
        model_config = get_model_config(model)
        if getattr(model_config, "use_mup", False):
            return "mup_not_supported"

        try:
            lm_head_weight = self._get_lm_head_weight(model)
        except RuntimeError as exc:
            return f"missing_lm_head_weight:{exc}"
        if lm_head_weight is None or lm_head_weight.dim() != 2:
            return "invalid_lm_head_weight"

        output_layer = _get_wrapped_attr(model, "output_layer", allow_none=True)
        if output_layer is not None and getattr(output_layer, "bias", None) is not None:
            return "output_layer_bias_not_supported"
        return None

    def _get_lm_head_weight(self, model: nn.Module) -> torch.Tensor:
        model_config = get_model_config(model)
        hidden_size = getattr(model_config, "hidden_size", None)

        share_embeddings = bool(
            _get_wrapped_attr(model, "share_embeddings_and_output_weights", allow_none=True)
        )
        if share_embeddings:
            shared_weight_getter = _get_wrapped_attr(
                model, "shared_embedding_or_output_weight", allow_none=True
            )
            if shared_weight_getter is not None:
                shared_weight = shared_weight_getter()
                if shared_weight is not None:
                    return shared_weight

        output_layer = _get_wrapped_attr(model, "output_layer", allow_none=True)
        if output_layer is None or getattr(output_layer, "weight", None) is None:
            for name, param in model.named_parameters():
                if param.dim() != 2 or "output_layer" not in name:
                    continue
                if hidden_size is not None and int(param.shape[-1]) != int(hidden_size):
                    continue
                print(
                    f"[skyrl] fused_lm_head_logprob using named parameter {name} "
                    f"shape={tuple(param.shape)}",
                    flush=True,
                )
                return param
            candidates = [
                f"{name}:{tuple(param.shape)}"
                for name, param in model.named_parameters()
                if param.dim() == 2 and ("output" in name or "embedding" in name)
            ][:12]
            raise RuntimeError(
                "Cannot locate Megatron LM-head output weight for fused logprob path. "
                f"candidate_2d_params={candidates}"
            )
        return output_layer.weight

    def train(self):
        [module.train() for module in self.actor_module]

    def eval(self):
        [module.eval() for module in self.actor_module]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        micro_batches: List[dict],
        seq_len: int,
        micro_batch_size: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward-only inference to compute log-probs over a full mini-batch consisting of multiple micro-batches.

        Args:
            micro_batches: List of micro-batch dicts with keys: "sequences", "attention_mask", "position_ids",
                           and "num_actions".
            seq_len: Padded sequence length per sample.
            micro_batch_size: Per-micro-batch size.
            temperature: Optional temperature scaling for logits.

        Returns:
            torch.Tensor of concatenated log-probs across micro-batches (valid on pipeline last stage only).
        """
        forward_backward_func = get_forward_backward_func()

        def collection_func(logits, data):
            sequences = data["sequences"]
            packed_seq_params = data.get("packed_seq_params")
            packed_targets = data.get("packed_targets")
            tp_grp = mpu.get_tensor_model_parallel_group()
            tp_rank = mpu.get_tensor_model_parallel_rank()

            if temperature != 1.0:
                logits.div_(temperature)

            if packed_seq_params is not None and packed_targets is not None:
                token_logprobs = from_parallel_logits_to_logprobs_packed_sequences(
                    logits,
                    packed_targets,
                    packed_seq_params.cu_seqlens_q_padded,
                    sequences.shape[1],
                    vocab_start_index=tp_rank * logits.shape[-1],
                    vocab_end_index=(tp_rank + 1) * logits.shape[-1],
                    group=tp_grp,
                    inference_only=True,
                    cp_group=mpu.get_context_parallel_group(),
                    chunk_size=self.cfg.logprobs_chunk_size,
                    attention_mask=data["attention_mask"],
                    sub_seq_lengths=data.get("sub_seq_lengths_list"),
                )
            else:
                token_logprobs = from_parallel_logits_to_logprobs(
                    logits,
                    sequences,
                    vocab_start_index=tp_rank * logits.shape[-1],
                    vocab_end_index=(tp_rank + 1) * logits.shape[-1],
                    tp_group=tp_grp,
                    inference_only=True,
                    cp_group=None,
                    chunk_size=self.cfg.logprobs_chunk_size,  # chunk seq dim to bound peak memory
                )
            return torch.tensor(0.0, device=token_logprobs.device), {"log_probs": token_logprobs}

        def forward_step(batch_iter, model):
            batch = _copy_tensor_dict_to_device(next(batch_iter), torch.cuda.current_device())

            rollout_expert_indices = batch.pop("rollout_expert_indices", None)
            if rollout_expert_indices is not None:
                setup_per_microbatch_replay_forward(
                    rollout_expert_indices,
                    batch["attention_mask"],
                    model_config=get_model_config(model),
                    remove_microbatch_padding=self.remove_microbatch_padding,
                )

            sequences = batch["sequences"]
            attention_mask = batch["attention_mask"].to(bool)
            position_ids = batch["position_ids"]
            sub_seq_lengths_field = batch.get("sub_seq_lengths")
            sub_seq_lengths = [t.tolist() for t in sub_seq_lengths_field] if sub_seq_lengths_field is not None else None
            batch["sub_seq_lengths_list"] = sub_seq_lengths

            if self.remove_microbatch_padding:
                new_sequences, packed_seq_params = preprocess_packed_seqs(
                    sequences,
                    attention_mask,
                    pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
                    sub_seq_lengths=sub_seq_lengths,
                )
                batch["packed_seq_params"] = packed_seq_params
                batch["packed_targets"] = _build_packed_targets(
                    sequences, attention_mask, packed_seq_params, sub_seq_lengths=sub_seq_lengths
                )
                new_attention_mask = None
                new_position_ids = None
            else:
                new_sequences, new_attention_mask, new_position_ids = remove_left_padding(
                    sequences,
                    attention_mask,
                    position_ids,
                    pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
                )
                packed_seq_params = None

            outputs = model(
                new_sequences,
                new_position_ids,
                new_attention_mask,
                packed_seq_params=packed_seq_params,
            )

            if not self.remove_microbatch_padding:
                outputs = recover_left_padding(
                    outputs,
                    new_attention_mask,
                    attention_mask,
                    seq_len,
                    post_process=mpu.is_pipeline_last_stage(ignore_virtual=True),
                )

            return outputs, partial(collection_func, data=batch)

        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.actor_module))

        output = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.actor_module,
            num_microbatches=len(micro_batches),
            seq_length=seq_len,
            micro_batch_size=micro_batch_size,
            forward_only=True,
        )

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            log_probs = [o["log_probs"] for o in output]
            log_probs = torch.cat(log_probs, dim=0)
            # take last num_actions tokens per micro; concatenate later
            # Assume all micros have same num_actions
            num_actions = micro_batches[0]["num_actions"]
            log_probs = log_probs[:, -num_actions:]
        else:
            # return dummy tensor for non-last pp stages
            device = micro_batches[0]["sequences"].device
            log_probs = torch.zeros(size=(1, 1), dtype=torch.bfloat16, device=device)
        return log_probs

    def forward_backward_mini_batch(
        self,
        micro_batches: List[dict],
        seq_len: int,
        micro_batch_size: int,
        temperature: float = 1.0,
        loss_fn: Optional[str] = None,
        loss_fn_config: Optional[Dict[str, Any]] = None,
        forward_only: bool = False,
    ) -> List[dict]:
        """
        Run forward-backward over a full mini-batch consisting of multiple micro-batches.

        Args:
            micro_batches: A list of micro-batch dicts. Each dict must contain keys:
                "sequences", "attention_mask", "position_ids", "num_actions",
                "old_action_log_probs", "base_action_log_probs", "advantages",
                "loss_mask", "rollout_action_logprobs".
            seq_len: Sequence length (tokens) per sample (assumed same across micros after padding).
            micro_batch_size: Micro-batch size per forward pass.
            temperature: Optional temperature for logits scaling.
            loss_fn: Optional loss function name (e.g., "cross_entropy", "ppo").
                     If provided, overrides the config's policy_loss_type.
            loss_fn_config: Optional config overrides for the loss function.
            forward_only: If True, run the forward pass without backward (no gradients).
                          Useful for evaluation / loss-only inference paths (e.g., SFT
                          ``forward(loss_fn=...)`` codepath).

        Returns:
            List[dict]: one metrics dict per micro-batch in order.
        """
        forward_backward_func = get_forward_backward_func()

        # Resolve loss function
        resolved_loss_name = loss_fn if loss_fn is not None else self.cfg.algorithm.policy_loss_type
        if loss_fn is not None:
            current_loss_fn = PolicyLossRegistry.get(loss_fn)
        else:
            current_loss_fn = self.policy_loss_fn

        # Build config for loss function, applying any overrides
        loss_config = self.cfg.algorithm
        if loss_fn_config is not None:

            new_loss_config = OmegaConf.merge(OmegaConf.create(asdict(loss_config)), OmegaConf.create(loss_fn_config))
            # NOTE: users can provide a custom loss config class, so we need to use the same class after applying overrides
            loss_config = type(loss_config).from_dict_config(new_loss_config)

        def loss_func(logits, data):
            sequences = data["sequences"]
            packed_seq_params = data.get("packed_seq_params")
            packed_targets = data.get("packed_targets")
            num_actions = data["num_actions"]
            old_action_log_probs = data["old_action_log_probs"]
            base_action_log_probs = data["base_action_log_probs"]
            advantages = data["advantages"]
            loss_mask = data["loss_mask"]
            rollout_action_logprobs = data["rollout_action_logprobs"]
            action_mask = data.get("action_mask")
            num_microbatches = data.get("num_microbatches")
            # Number of microbatches carrying real samples (excludes fully-padding
            # microbatches added by token-based batching). Used to normalize the
            # KL/entropy terms over real microbatches only. Falls back to
            # num_microbatches when not provided (no padding microbatches).
            num_real_microbatches = data.get("num_real_microbatches", num_microbatches)

            dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
            tp_grp = mpu.get_tensor_model_parallel_group()
            tp_rank = mpu.get_tensor_model_parallel_rank()
            fused_lm_head_logprob = bool(data.get("fused_lm_head_logprob", False))
            fused_hidden_states = None
            fused_lm_head_weight = data.get("fused_lm_head_weight")

            if fused_lm_head_logprob:
                # Megatron returns hidden states as [sequence, batch, hidden]
                # when GPTModel.post_process is disabled.
                mtp_num_layers = int(data.get("fused_lm_head_mtp_num_layers") or 0)
                if mtp_num_layers > 0:
                    logits = torch.chunk(logits, 1 + mtp_num_layers, dim=0)[0]
                fused_hidden_states = logits.transpose(0, 1).contiguous()
                if fused_lm_head_weight is not None and fused_hidden_states.shape[-1] != fused_lm_head_weight.shape[-1]:
                    raise RuntimeError(
                        "Fused LM-head path expected decoder hidden states with last dim "
                        f"{fused_lm_head_weight.shape[-1]}, got {tuple(fused_hidden_states.shape)}. "
                        "Megatron post_process was not disabled for this forward."
                    )
            elif temperature != 1.0:
                logits.div_(temperature)

            if fused_lm_head_logprob:
                if packed_seq_params is None or packed_targets is None or fused_lm_head_weight is None:
                    raise RuntimeError("Fused LM-head logprob path requires packed targets and LM-head weight.")
                token_logprobs = from_fused_lm_head_to_logprobs_packed_sequences(
                    fused_hidden_states,
                    fused_lm_head_weight,
                    packed_targets,
                    packed_seq_params.cu_seqlens_q_padded,
                    sequences.shape[1],
                    vocab_start_index=tp_rank * fused_lm_head_weight.shape[0],
                    vocab_end_index=(tp_rank + 1) * fused_lm_head_weight.shape[0],
                    group=tp_grp,
                    chunk_size=self.cfg.logprobs_chunk_size,
                    temperature=temperature,
                    cp_group=mpu.get_context_parallel_group(),
                    attention_mask=data["attention_mask"],
                    sub_seq_lengths=data.get("sub_seq_lengths_list"),
                )
            elif packed_seq_params is not None and packed_targets is not None:
                token_logprobs = from_parallel_logits_to_logprobs_packed_sequences(
                    logits,
                    packed_targets,
                    packed_seq_params.cu_seqlens_q_padded,
                    sequences.shape[1],
                    vocab_start_index=tp_rank * logits.shape[-1],
                    vocab_end_index=(tp_rank + 1) * logits.shape[-1],
                    group=tp_grp,
                    inference_only=False,
                    cp_group=mpu.get_context_parallel_group(),
                    chunk_size=self.cfg.logprobs_chunk_size,
                    attention_mask=data["attention_mask"],
                    sub_seq_lengths=data.get("sub_seq_lengths_list"),
                )
            else:
                token_logprobs = from_parallel_logits_to_logprobs(
                    logits,
                    sequences,
                    vocab_start_index=tp_rank * logits.shape[-1],
                    vocab_end_index=(tp_rank + 1) * logits.shape[-1],
                    tp_group=tp_grp,
                    inference_only=False,
                    cp_group=None,
                    chunk_size=self.cfg.logprobs_chunk_size,  # chunk seq dim to bound peak memory
                )

            action_log_probs = token_logprobs[:, -num_actions:]

            # policy loss should be calculated based on the selected token logprobs
            policy_loss, loss_metrics = current_loss_fn(
                action_log_probs,
                old_action_log_probs,
                advantages,
                config=loss_config,
                loss_mask=loss_mask,
                rollout_logprobs=rollout_action_logprobs,
            )

            # SFT path: cross_entropy loss (negative log likelihood)
            if resolved_loss_name == "cross_entropy":
                loss = policy_loss

                # Compute elementwise loss for Tinker API (per-token NLL)
                with torch.no_grad():
                    elementwise_loss = -action_log_probs
                    if loss_mask is not None:
                        elementwise_loss = elementwise_loss * loss_mask

                # Build per-sequence loss_fn_outputs.
                # Compute valid_lens vectorized on GPU, then move tensors to CPU
                # exactly once before iterating in Python — avoids ~3N GPU->CPU
                # syncs per micro-batch (item()/cpu()/tolist() inside the loop).
                batch_size = action_log_probs.shape[0]
                seq_len = action_log_probs.shape[1]
                if action_mask is not None:
                    valid_lens_t = action_mask.sum(dim=-1).long()
                elif loss_mask is not None:
                    valid_lens_t = loss_mask.sum(dim=-1).long()
                else:
                    valid_lens_t = torch.full((batch_size,), seq_len, device=action_log_probs.device, dtype=torch.long)

                # Bulk GPU->CPU sync: one transfer for logprobs, elementwise_loss, and valid_lens.
                action_log_probs_cpu = action_log_probs.detach().cpu()
                elementwise_loss_cpu = elementwise_loss.detach().cpu()
                valid_lens = valid_lens_t.cpu().tolist()

                loss_fn_outputs = []
                for i in range(batch_size):
                    valid_len = valid_lens[i]
                    loss_fn_outputs.append(
                        {
                            "logprobs": (action_log_probs_cpu[i, -valid_len:].tolist() if valid_len > 0 else []),
                            "elementwise_loss": (
                                elementwise_loss_cpu[i, -valid_len:].tolist() if valid_len > 0 else []
                            ),
                        }
                    )

                metrics = {
                    "loss": loss.item(),
                    "response_length": num_actions,
                    "loss_fn_outputs": loss_fn_outputs,
                }
                return loss, metrics

            # RL path: add optional KL/entropy terms
            with torch.set_grad_enabled(loss_config.use_entropy_loss):
                if fused_lm_head_logprob:
                    entropy, entropy_for_loss = vocab_parallel_entropy_from_fused_lm_head_packed_sequences(
                        fused_hidden_states,
                        fused_lm_head_weight,
                        packed_seq_params.cu_seqlens_q_padded,
                        sequences.shape[1],
                        num_actions,
                        data["attention_mask"],
                        loss_mask,
                        mpu.get_context_parallel_group(),
                        sub_seq_lengths=data.get("sub_seq_lengths_list"),
                        chunk_size=self.cfg.vocab_entropy_chunk_size,
                        chunk_memory_mb=self.cfg.vocab_entropy_chunk_memory_mb,
                        temperature=temperature,
                    )
                elif packed_seq_params is not None and packed_targets is not None:
                    entropy, entropy_for_loss = vocab_parallel_entropy_packed_sequences(
                        logits,
                        packed_seq_params.cu_seqlens_q_padded,
                        sequences.shape[1],
                        num_actions,
                        data["attention_mask"],
                        loss_mask,
                        mpu.get_context_parallel_group(),
                        sub_seq_lengths=data.get("sub_seq_lengths_list"),
                        chunk_size=self.cfg.vocab_entropy_chunk_size,
                        chunk_memory_mb=self.cfg.vocab_entropy_chunk_memory_mb,
                    )
                else:
                    action_logits = logits[:, -num_actions - 1 : -1, :]
                    entropy_BS = vocab_parallel_entropy(
                        action_logits,
                        chunk_size=self.cfg.vocab_entropy_chunk_size,
                        chunk_memory_mb=self.cfg.vocab_entropy_chunk_memory_mb,
                    )
                    entropy = masked_mean(entropy_BS, loss_mask)
                    entropy_for_loss = entropy

            if loss_config.use_entropy_loss:
                entropy_loss_term = entropy_for_loss * loss_config.entropy_loss_coef
            else:
                entropy_loss_term = torch.tensor(0.0, device=action_log_probs.device)

            if loss_config.use_kl_loss:
                kl_loss = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    loss_mask=loss_mask,
                    kl_estimator_type=loss_config.kl_estimator_type,
                )
                kl_loss = masked_mean(kl_loss, loss_mask, dim=-1).mean()
            else:
                kl_loss = torch.tensor(0.0, device=action_log_probs.device)
            kl_loss_term = kl_loss * loss_config.kl_loss_coef

            # Policy losses are pre-scaled to achieve the correct loss_reduction
            # when summing across the entire minibatch (see `apply_loss_reduction_to_advantages_minibatch`).
            # Megatron divides loss by num_microbatches
            # (https://github.com/NVIDIA/Megatron-LM/blob/core_v0.15.2/megatron/core/pipeline_parallel/schedules.py#L248)
            # and the data parallel all-reduce averages gradients across dp_size.
            # Megatron's schedule separately multiplies loss by the CP size for two-output loss funcs,
            # so CP ranks are not included in this correction factor.
            # (https://github.com/NVIDIA/Megatron-LM/blob/core_v0.15.2/megatron/core/distributed/distributed_data_parallel.py#L285)
            # so we multiply by both factors to recover the correct sum reduction.
            grad_sum_correction_factor = num_microbatches * dp_size

            # NOTE: The KL and entropy loss terms are not pre-scaled,
            # so we just average them across microbatches and DP workers.
            # KL and entropy use Megatron's existing microbatch and CP schedule scaling.
            # Megatron divides by num_microbatches (which includes fully-padding microbatches
            # added by token-based batching). Those padding microbatches contribute 0 to
            # KL/entropy, so dividing by the full count would dilute the regularization by
            # num_real/num_total. Scale up by num_microbatches/num_real_microbatches so the
            # terms are averaged over real microbatches only (no-op when there is no padding).
            kl_entropy_microbatch_scale = num_microbatches / max(1, num_real_microbatches)
            loss = (
                policy_loss * grad_sum_correction_factor
                + (kl_loss_term - entropy_loss_term) * kl_entropy_microbatch_scale
            )
            unscaled_loss = loss / grad_sum_correction_factor

            # Build per-sequence loss_fn_outputs with logprobs.
            batch_size = action_log_probs.shape[0]
            seq_len = action_log_probs.shape[1]

            if action_mask is not None:
                valid_lens = action_mask.sum(dim=1).int().tolist()
            elif loss_mask is not None:
                valid_lens = loss_mask.sum(dim=1).int().tolist()
            else:
                valid_lens = [seq_len] * batch_size

            detached_log_probs = action_log_probs.detach().cpu()
            loss_fn_outputs = []
            for i, valid_len in enumerate(valid_lens):
                loss_fn_outputs.append(
                    {
                        "logprobs": detached_log_probs[i, -valid_len:].tolist() if valid_len > 0 else [],
                    }
                )

            metrics = {
                "final_loss": unscaled_loss.detach().item(),
                "policy_loss": policy_loss.detach().item(),
                "policy_entropy": entropy.detach().item(),
                "policy_kl": kl_loss.detach().item(),
                "loss_fn_outputs": loss_fn_outputs,
            }
            for k, v in loss_metrics.items():
                metrics["loss_metrics/" + k] = v
            metrics.update(
                compute_minibatch_rollout_logprob_diff_metrics(action_log_probs, rollout_action_logprobs, loss_mask)
            )
            return loss, metrics

        def forward_step(batch_iter, model):
            # NOTE(Charlie): despite the name, methods like `remove_left_padding()` are padding-agnostic
            # (can be left, or right) as it uses attention_mask to locate real tokens. Same thing
            # for recover_left_padding and setup_per_microbatch_replay_forward. Especially relevant
            # after this PR https://github.com/NovaSky-AI/SkyRL/pull/1285.
            batch = _copy_tensor_dict_to_device(next(batch_iter), torch.cuda.current_device())

            rollout_expert_indices = batch.pop("rollout_expert_indices", None)
            if rollout_expert_indices is not None:
                setup_per_microbatch_replay_forward(
                    rollout_expert_indices,
                    batch["attention_mask"],
                    model_config=get_model_config(model),
                    remove_microbatch_padding=self.remove_microbatch_padding,
                )

            sequences = batch["sequences"]
            attention_mask = batch["attention_mask"].to(bool)
            position_ids = batch["position_ids"]
            # When present, sub_seq_lengths enumerates every sub-sequence
            # inside every row of the micro-batch (controller-side mini-batch
            # packing). preprocess_packed_seqs uses it to emit cu_seqlens
            # entries covering all sub-seqs, not one per row.
            #
            # It arrives as a ``TensorList`` data field.
            # ``preprocess_packed_seqs`` and the packed-logprob scatter use
            # ``list[list[int]]``, so convert tensors -> python lists here.
            sub_seq_lengths_field = batch.get("sub_seq_lengths")
            sub_seq_lengths = [t.tolist() for t in sub_seq_lengths_field] if sub_seq_lengths_field is not None else None
            batch["sub_seq_lengths_list"] = sub_seq_lengths
            fused_fallback_reason = self._fused_lm_head_fallback_reason(
                model,
                resolved_loss_name,
                loss_config,
                forward_only,
            )
            use_fused_lm_head_logprob = (
                self.cfg.fused_lm_head_logprob
                and mpu.is_pipeline_last_stage(ignore_virtual=True)
                and fused_fallback_reason is None
            )
            if (
                self.cfg.fused_lm_head_logprob
                and mpu.is_pipeline_last_stage(ignore_virtual=True)
                and not self._fused_lm_head_status_logged
            ):
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                status = "enabled" if use_fused_lm_head_logprob else f"fallback:{fused_fallback_reason}"
                print(f"[skyrl] fused_lm_head_logprob={status} rank={rank}", flush=True)
                self._fused_lm_head_status_logged = True
                if fused_fallback_reason is not None:
                    raise RuntimeError(
                        "trainer.fused_lm_head_logprob=true but fused path is unavailable: "
                        f"{fused_fallback_reason}"
                    )

            if self.remove_microbatch_padding:
                new_sequences, packed_seq_params = preprocess_packed_seqs(
                    sequences,
                    attention_mask,
                    pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
                    sub_seq_lengths=sub_seq_lengths,
                )
                batch["packed_seq_params"] = packed_seq_params
                batch["packed_targets"] = _build_packed_targets(
                    sequences, attention_mask, packed_seq_params, sub_seq_lengths=sub_seq_lengths
                )
                new_attention_mask = None
                new_position_ids = None
            else:
                new_sequences, new_attention_mask, new_position_ids = remove_left_padding(
                    sequences,
                    attention_mask,
                    position_ids,
                    pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
                )
                packed_seq_params = None

            if use_fused_lm_head_logprob:
                batch["fused_lm_head_logprob"] = True
                batch["fused_lm_head_weight"] = self._get_lm_head_weight(model)
                model_config = get_model_config(_unwrap_model(model))
                batch["fused_lm_head_mtp_num_layers"] = getattr(model_config, "mtp_num_layers", None)

            post_process_context = (
                _temporary_post_process(model, post_process=False) if use_fused_lm_head_logprob else nullcontext()
            )
            with post_process_context:
                outputs = model(
                    new_sequences,
                    new_position_ids,
                    new_attention_mask,
                    packed_seq_params=packed_seq_params,
                )

            if not self.remove_microbatch_padding:
                outputs = recover_left_padding(
                    outputs,
                    new_attention_mask,
                    attention_mask,
                    seq_len,
                    post_process=mpu.is_pipeline_last_stage(ignore_virtual=True),
                )

            if rollout_expert_indices is not None:
                setup_per_microbatch_replay_backward()

            return outputs, partial(loss_func, data=batch)

        # batch should be a list of micro-batches
        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.actor_module))

        metrics_list = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.actor_module,
            num_microbatches=len(micro_batches),
            seq_length=seq_len,
            micro_batch_size=micro_batch_size,
            forward_only=forward_only,
        )

        # broadcast metrics to all pp ranks
        if not mpu.is_pipeline_last_stage(ignore_virtual=True):
            metrics_list = [None] * len(micro_batches)
        with torch.no_grad():
            torch.distributed.broadcast_object_list(
                metrics_list,
                src=mpu.get_pipeline_model_parallel_last_rank(),
                group=mpu.get_pipeline_model_parallel_group(),
            )

        return metrics_list
