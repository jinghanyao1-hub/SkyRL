"""Tests for build_vllm_cli_args on GPU-less hosts."""

import pytest

from skyrl.backends.skyrl_train.inference_servers.utils import (
    _apply_serialized_fp8_weight_sync_defaults,
    build_vllm_cli_args,
    resolve_policy_model_name,
)
from skyrl.train.config import SkyRLTrainConfig


def test_serialized_fp8_weight_sync_defaults_configure_vllm_checkpoint_fp8():
    cfg = SkyRLTrainConfig()
    ie_cfg = cfg.generator.inference_engine
    ie_cfg.fp8_weight_sync_mode = "serialized_blockwise"
    engine_kwargs = {"hf_overrides": {"rope_theta": 10000.0}}

    _apply_serialized_fp8_weight_sync_defaults(ie_cfg, engine_kwargs)

    assert engine_kwargs["quantization"] == "fp8"
    assert engine_kwargs["load_format"] == "dummy"
    assert engine_kwargs["hf_overrides"]["rope_theta"] == 10000.0
    assert engine_kwargs["hf_overrides"]["quantization_config"] == {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }


@pytest.mark.vllm
def test_build_vllm_cli_args_succeeds_on_gpu_less_host(monkeypatch):
    import vllm.platforms
    from vllm.platforms.interface import UnspecifiedPlatform

    # Simulate the GPU-less Ray head-node case: vLLM resolves current_platform
    # to UnspecifiedPlatform (device_type == ""), so AsyncEngineArgs.add_cli_args
    # walks VllmConfig defaults, instantiates DeviceConfig() and its
    # __post_init__ raises "Failed to infer device type" during arg parsing.
    # With the fix in build_vllm_cli_args, current_platform.device_type is
    # pinned to "cuda" before add_cli_args runs.
    monkeypatch.setattr(vllm.platforms, "_current_platform", UnspecifiedPlatform())

    cfg = SkyRLTrainConfig()
    cfg.generator.inference_engine.served_model_name = "served-alias"
    cfg.generator.inference_engine.engine_init_kwargs = {
        "hf_overrides": {"rope_parameters": {"rope_type": "linear", "factor": 2.0, "rope_theta": 10000.0}}
    }
    args = build_vllm_cli_args(cfg)

    assert args is not None
    assert args.model == cfg.trainer.policy.model.path
    assert args.served_model_name == ["served-alias"]
    assert args.tensor_parallel_size == cfg.generator.inference_engine.tensor_parallel_size
    assert args.hf_overrides["rope_parameters"] == {"rope_type": "linear", "factor": 2.0, "rope_theta": 10000.0}
    assert vllm.platforms.current_platform.device_type == "cuda"


def test_resolve_policy_model_name_uses_served_model_name():
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = "base-model"
    cfg.generator.inference_engine.served_model_name = "served-alias"

    assert resolve_policy_model_name(cfg) == "served-alias"
