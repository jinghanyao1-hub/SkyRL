"""Monkeypatch Transformer-Engine's ``Float8BlockScaling`` recipe to apply a finite
``amax_epsilon`` floor to its blockwise FP8 quantizers.

Background
----------
Qwen3.5-35B-A3B full-FP8 RL training on Blackwell (B200) uses Megatron + TE
``Float8BlockScaling`` (1x128 blockwise, e4m3). The forward pass is fine, but the
grouped-expert weight-gradient produces ``grad_norm=inf`` every step. Root cause:
TE quantizes the grouped-GEMM operands with ``amax_epsilon=0.0`` (the numerical
floor is DISABLED) and forced power-of-2 scales. Zero-amax blocks -- created by
``Fp8Padding`` zero-rows (expert token counts padded up to a multiple of 16)
and/or zero/masked gradient rows -- underflow the scale computation and produce
``inf`` in the accumulated expert weight gradient.

Fix
---
Give the blockwise quantizers a small positive ``amax_epsilon`` so zero-amax
blocks get a finite floor instead of underflowing. Megatron builds the recipe as
``Float8BlockScaling(fp8_format=...)`` (see ``megatron/core/fp8_utils.py``
``get_fp8_recipe``) and does NOT expose ``amax_epsilon``, so we patch the recipe
CLASS. The recipe is rebuilt on every forward pass inside
``megatron.core.fp8_utils.get_fp8_context``; every instance it constructs -- and
thus every ``Float8BlockQuantizer`` used by the Linear AND
GroupedLinear/TEGroupedMLP expert paths -- then picks up the floor.

Driven by env var ``NVTE_FP8_BLOCK_AMAX_EPSILON`` (a float). Unset / empty / <= 0
=> no-op (returns immediately, does not even import TE).

TE-verified facts (checked against the running container, TE 2.10.0)
--------------------------------------------------------------------
* ``Float8BlockScaling`` is a (pydantic) dataclass and HAS ``__post_init__``.
* Its three quantizer configs -- ``fp8_quant_fwd_inp``, ``fp8_quant_fwd_weight``,
  ``fp8_quant_bwd_grad`` -- are ``QParams`` (a frozen dataclass) held as SHARED
  CLASS attributes, NOT per-instance dataclass fields. Replacing the class
  attributes therefore propagates to every instance (verified:
  ``instance.fp8_quant_fwd_inp is Float8BlockScaling.fp8_quant_fwd_inp``).
* ``QParams`` fields: ``power_2_scale, amax_epsilon, random_hadamard_transform,
  stochastic_rounding, fp4_2d_quantization``. Default ``amax_epsilon=0.0``.
* Semantics of ``amax_epsilon``: the C++ setter (transformer_engine.h) is
  documented as "small value to add to amax"; the Python ``QParams`` docstring
  as "optional minimum value of abs max". Either way a small positive value
  floors zero-amax blocks; ``power_2_scale`` (required on Blackwell) is preserved.
"""

from __future__ import annotations

import os

try:  # match the surrounding module's loguru usage; fall back to stdlib logging.
    from loguru import logger as _logger

    def _log_info(msg: str) -> None:
        _logger.info(msg)

    def _log_warning(msg: str) -> None:
        _logger.warning(msg)

except Exception:  # pragma: no cover - loguru should always be present in this env
    import logging

    _std_logger = logging.getLogger(__name__)

    def _log_info(msg: str) -> None:
        _std_logger.info(msg)

    def _log_warning(msg: str) -> None:
        _std_logger.warning(msg)


_ENV_VAR = "NVTE_FP8_BLOCK_AMAX_EPSILON"
_PATCH_FLAG = "_skyrl_amax_epsilon_patched"
_ORIG_POST_INIT_ATTR = "_skyrl_orig_post_init"
_QPARAM_FIELDS = ("fp8_quant_fwd_inp", "fp8_quant_fwd_weight", "fp8_quant_bwd_grad")


def apply_fp8_block_amax_epsilon_patch() -> None:
    """Idempotently patch TE ``Float8BlockScaling`` to floor ``amax_epsilon``.

    No-op unless env ``NVTE_FP8_BLOCK_AMAX_EPSILON`` is set to a positive float.
    Never raises: any failure is logged and swallowed so the worker keeps running
    even on a TE-version mismatch.
    """
    raw = os.getenv(_ENV_VAR)
    if raw is None or raw.strip() == "":
        return  # feature disabled -> silent no-op (does not import TE).

    try:
        epsilon = float(raw)
    except (TypeError, ValueError):
        _log_warning(
            f"[fp8-amax-eps] {_ENV_VAR}={raw!r} is not a valid float; "
            f"leaving TE Float8BlockScaling unchanged."
        )
        return

    if epsilon <= 0.0:
        _log_warning(
            f"[fp8-amax-eps] {_ENV_VAR}={epsilon} is <= 0; leaving TE "
            f"Float8BlockScaling unchanged (0.0 is the TE default = floor disabled)."
        )
        return

    try:
        from transformer_engine.common.recipe import Float8BlockScaling, QParams
    except Exception as exc:  # TE missing / renamed / version mismatch -> never crash.
        _log_warning(
            f"[fp8-amax-eps] could not import Float8BlockScaling/QParams from "
            f"transformer_engine ({exc!r}); patch skipped."
        )
        return

    # Idempotency: skip if already patched with the same epsilon.
    if getattr(Float8BlockScaling, _PATCH_FLAG, None) == epsilon:
        return

    def _with_epsilon(qp: "QParams") -> "QParams":
        # QParams is a frozen dataclass -> build a new one, preserving every field
        # except amax_epsilon. Critically preserves power_2_scale (Blackwell requires
        # power-of-2 scales; NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1 asserts on B200).
        return QParams(
            power_2_scale=qp.power_2_scale,
            amax_epsilon=epsilon,
            random_hadamard_transform=qp.random_hadamard_transform,
            stochastic_rounding=qp.stochastic_rounding,
            fp4_2d_quantization=qp.fp4_2d_quantization,
        )

    try:
        # (1) PRIMARY: replace the SHARED CLASS-LEVEL QParams. In TE 2.10 these are
        # plain class attributes (not dataclass fields), so this propagates to every
        # Float8BlockScaling instance Megatron builds, hence to every quantizer
        # (Linear + GroupedLinear/TEGroupedMLP expert path).
        for name in _QPARAM_FIELDS:
            setattr(Float8BlockScaling, name, _with_epsilon(getattr(Float8BlockScaling, name)))

        # (2) DEFENSE-IN-DEPTH: also override per-instance inside __post_init__, so the
        # floor still applies if a future TE version turns these into real per-instance
        # dataclass fields (where class-attr replacement would be shadowed). Wrap the
        # original __post_init__ (keeps its asserts) exactly once.
        if not hasattr(Float8BlockScaling, _ORIG_POST_INIT_ATTR):
            _orig_post_init = getattr(Float8BlockScaling, "__post_init__", None)
            setattr(Float8BlockScaling, _ORIG_POST_INIT_ATTR, _orig_post_init)

            def _patched_post_init(self) -> None:
                orig = getattr(type(self), _ORIG_POST_INIT_ATTR, None)
                if orig is not None:
                    orig(self)
                eps = getattr(type(self), _PATCH_FLAG, None)
                if not eps or eps <= 0.0:
                    return
                for fname in _QPARAM_FIELDS:
                    cur = getattr(self, fname)
                    if getattr(cur, "amax_epsilon", None) == eps:
                        continue
                    # QParams is frozen and Float8BlockScaling is a pydantic dataclass;
                    # object.__setattr__ installs the per-instance override safely.
                    object.__setattr__(
                        self,
                        fname,
                        QParams(
                            power_2_scale=cur.power_2_scale,
                            amax_epsilon=eps,
                            random_hadamard_transform=cur.random_hadamard_transform,
                            stochastic_rounding=cur.stochastic_rounding,
                            fp4_2d_quantization=cur.fp4_2d_quantization,
                        ),
                    )

            Float8BlockScaling.__post_init__ = _patched_post_init

        setattr(Float8BlockScaling, _PATCH_FLAG, epsilon)
    except Exception as exc:
        _log_warning(
            f"[fp8-amax-eps] failed to patch Float8BlockScaling ({exc!r}); "
            f"TE recipe left unchanged."
        )
        return

    # (3) VERIFY on a freshly constructed instance so the infra log proves the patch
    # took effect (this is the exact construction Megatron uses).
    try:
        from transformer_engine.common.recipe import Format

        probe = Float8BlockScaling(fp8_format=Format.E4M3)
        observed = {name: getattr(probe, name).amax_epsilon for name in _QPARAM_FIELDS}
        pow2 = {name: getattr(probe, name).power_2_scale for name in _QPARAM_FIELDS}
        all_ok = all(v == epsilon for v in observed.values())
        _log_info(
            f"[fp8-amax-eps] patched TE Float8BlockScaling: set amax_epsilon={epsilon} "
            f"(env {_ENV_VAR}={raw!r}) on {_QPARAM_FIELDS}. "
            f"Verified on a fresh Float8BlockScaling(fp8_format=E4M3): "
            f"amax_epsilon={observed}, power_2_scale(preserved)={pow2} -> "
            f"{'OK' if all_ok else 'WARNING: MISMATCH'}."
        )
        if not all_ok:
            _log_warning(
                f"[fp8-amax-eps] verification MISMATCH: expected {epsilon}, got {observed}."
            )
    except Exception as exc:
        _log_warning(
            f"[fp8-amax-eps] patch applied but post-patch verification failed ({exc!r})."
        )
