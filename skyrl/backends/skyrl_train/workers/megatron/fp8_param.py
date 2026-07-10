"""Utilities for training with persistent Transformer Engine FP8 parameters."""

from collections.abc import Mapping
from typing import Any


def is_fp8_param_enabled(transformer_config_kwargs: Mapping[str, Any]) -> bool:
    """Read persistent-FP8 enablement from SkyRL's dictionary config."""
    return bool(transformer_config_kwargs.get("fp8_param", False))


def initialize_fp8_param_optimizer_masters(
    optimizer: Any,
    *,
    fp8_param: bool,
    fp8_param_gather: bool,
) -> int:
    """Seed FP32 optimizer masters from the loaded FP8 compute parameters.

    Transformer Engine creates persistent FP8 GEMM weights during model
    construction. Megatron-Bridge intentionally skips random initialization
    before importing Hugging Face weights, so TE's optional high-precision
    initialization snapshot is not a valid source for the optimizer master
    parameters. Rebuild the masters after the import by dequantizing the
    compute parameters through Megatron's supported reload path.
    """
    if not fp8_param:
        return 0
    if not fp8_param_gather:
        raise ValueError(
            "Persistent FP8 parameters require ddp_config.fp8_param_gather=true "
            "so updated FP32 master weights are requantized into FP8 compute weights."
        )

    optimizers = getattr(optimizer, "chained_optimizers", None)
    if optimizers is None:
        optimizers = [optimizer]

    initialized = 0
    for megatron_optimizer in optimizers:
        reload_main_params = getattr(megatron_optimizer, "_copy_model_params_to_main_params", None)
        if not callable(reload_main_params):
            raise TypeError(
                "Persistent FP8 parameter training requires a Megatron optimizer "
                "with _copy_model_params_to_main_params()."
            )
        reload_main_params()
        initialized += 1
    return initialized
