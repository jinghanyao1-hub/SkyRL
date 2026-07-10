import pytest

from skyrl.backends.skyrl_train.workers.megatron.fp8_param import (
    initialize_fp8_param_optimizer_masters,
    is_fp8_param_enabled,
)


class _FakeOptimizer:
    def __init__(self):
        self.calls = 0

    def _copy_model_params_to_main_params(self):
        self.calls += 1


def test_fp8_param_enablement_reads_skyrl_transformer_config_mapping():
    assert is_fp8_param_enabled({"fp8_param": True})
    assert not is_fp8_param_enabled({"fp8_param": False})
    assert not is_fp8_param_enabled({})


def test_fp8_param_master_initialization_reloads_each_chained_optimizer():
    first = _FakeOptimizer()
    second = _FakeOptimizer()
    chained = type("FakeChainedOptimizer", (), {"chained_optimizers": [first, second]})()

    initialized = initialize_fp8_param_optimizer_masters(
        chained,
        fp8_param=True,
        fp8_param_gather=True,
    )

    assert initialized == 2
    assert first.calls == 1
    assert second.calls == 1


def test_fp8_param_master_initialization_requires_fp8_param_gather():
    optimizer = _FakeOptimizer()

    with pytest.raises(ValueError, match="fp8_param_gather=true"):
        initialize_fp8_param_optimizer_masters(
            optimizer,
            fp8_param=True,
            fp8_param_gather=False,
        )

    assert optimizer.calls == 0


def test_non_persistent_fp8_does_not_touch_optimizer_masters():
    optimizer = _FakeOptimizer()

    initialized = initialize_fp8_param_optimizer_masters(
        optimizer,
        fp8_param=False,
        fp8_param_gather=False,
    )

    assert initialized == 0
    assert optimizer.calls == 0
