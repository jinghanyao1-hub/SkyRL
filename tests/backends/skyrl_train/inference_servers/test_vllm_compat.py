import torch

from skyrl.backends.skyrl_train.inference_servers.vllm_compat import (
    patch_vllm_fp8_kv_cache_sleep_wake,
)


def test_fp8_kv_cache_sleep_wake_patch_flattens_nested_mamba_states():
    class FakeRunner:
        def __init__(self, kv_caches):
            self.kv_caches = kv_caches

        def init_fp8_kv_scales(self):
            for cache in self.kv_caches:
                cache.zero_()

    flat_cache = torch.ones(2)
    mamba_state_a = torch.ones(3)
    mamba_state_b = torch.ones(4)
    kv_caches = [flat_cache, [mamba_state_a, {"state": mamba_state_b}], None]
    runner = FakeRunner(kv_caches)

    assert patch_vllm_fp8_kv_cache_sleep_wake(FakeRunner)
    assert not patch_vllm_fp8_kv_cache_sleep_wake(FakeRunner)

    runner.init_fp8_kv_scales()

    assert runner.kv_caches is kv_caches
    assert torch.count_nonzero(flat_cache) == 0
    assert torch.count_nonzero(mamba_state_a) == 0
    assert torch.count_nonzero(mamba_state_b) == 0
