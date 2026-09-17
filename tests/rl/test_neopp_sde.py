import os
import tempfile
import unittest
from pathlib import Path

# These tests exercise the rollout math and trace store on CPU.  Keep the
# production CUDA check intact while making the test entry point self-contained.
os.environ.setdefault("SKIP_PLATFORM_CHECK", "True")

import torch
from safetensors.torch import load_file

from lightx2v.rl.sde import HybridSdeState, SdeRolloutConfig, recompute_log_prob, transition
from lightx2v.rl.trace_store import TraceStore


class NeoPPSdeTest(unittest.TestCase):
    def test_invalid_t_eps_is_rejected(self):
        with self.assertRaises(ValueError):
            SdeRolloutConfig(t_eps=0.0).resolve_indices(4)

    def test_zero_noise_mean_is_official_euler_step(self):
        sample = torch.randn(2, 3, 4)
        velocity = torch.randn_like(sample)
        mean, scale = transition(velocity, sample, 0.25, 0.5, 0.75, 0.0)
        expected = sample.float() + 0.25 * velocity.float()
        torch.testing.assert_close(mean, expected)
        self.assertEqual(float(scale.max()), 0.0)

    def test_trace_recomputes_saved_log_prob(self):
        timesteps = torch.linspace(0, 1, 5)
        state = HybridSdeState(
            SdeRolloutConfig(noise_level=0.7, indices=(0, 2)),
            timesteps,
            seed=7,
            device=torch.device("cpu"),
        )
        sample = torch.randn(2, 3, 4)
        for index in range(4):
            velocity = torch.full_like(sample, 0.1 * (index + 1))
            sample = state.advance(sample, velocity, timesteps[index], timesteps[index + 1], index)
        trace = state.finish(sample)
        for next_sample, mean, scale, old_log_prob in zip(trace.next_samples, trace.old_means, trace.scales, trace.old_log_probs):
            torch.testing.assert_close(recompute_log_prob(next_sample, mean, scale), old_log_prob)

    def test_trace_store_ttl_and_delete(self):
        timesteps = torch.linspace(0, 1, 3)
        state = HybridSdeState(SdeRolloutConfig(indices=(0,)), timesteps, seed=1, device=torch.device("cpu"))
        sample = torch.randn(1, 2, 3)
        sample = state.advance(sample, torch.zeros_like(sample), timesteps[0], timesteps[1], 0)
        sample = state.advance(sample, torch.zeros_like(sample), timesteps[1], timesteps[2], 1)
        trace = state.finish(sample)
        with tempfile.TemporaryDirectory() as directory:
            store = TraceStore(directory, ttl_seconds=10)
            bundle_id = store.put(trace, {"policy_version": "test"})
            path = store.get_path(bundle_id)
            self.assertTrue(path.is_file())
            self.assertIn("old_log_probs", load_file(str(path)))
            self.assertTrue(store.delete(bundle_id))
            self.assertFalse(Path(path).exists())


if __name__ == "__main__":
    unittest.main()
