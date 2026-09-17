import torch

from lightx2v.rl.weights import model_state, update_weights
from types import SimpleNamespace


class _StateNode:
    def __init__(self, values):
        self.values = values

    def state_dict(self, destination):
        destination.update(self.values)


class _Model:
    def __init__(self):
        self.pre_weight = _StateNode(
            {
                "model.weight": torch.ones(4),
                "diffusion_model.blocks.norm.diff": torch.tensor(0.0),
                "diffusion_model.blocks.active_norm.diff": torch.ones(4),
            }
        )
        self.transformer_weights = _StateNode({})


def test_model_state_skips_only_inactive_scalar_diff_placeholders():
    state = model_state(_Model())

    assert set(state) == {
        "model.weight",
        "diffusion_model.blocks.active_norm.diff",
    }


def test_online_update_refreshes_fused_experts_from_checkpoint_tensors():
    model = _Model()
    fused = torch.zeros(4)
    block = SimpleNamespace(mlp_mot_gen=SimpleNamespace(
        rebuild_fused_weights=lambda: fused.copy_(model.pre_weight.values["model.weight"] * 2),
    ))
    model.transformer_weights.blocks = [block]
    incoming = {name: torch.full_like(value, 3) for name, value in model_state(model).items()}
    receipt = update_weights(model, incoming, strict=True)
    torch.testing.assert_close(fused, torch.full((4,), 6.0))
    assert receipt["updated"] == sorted(incoming)
