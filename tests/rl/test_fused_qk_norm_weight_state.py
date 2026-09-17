import ast
from pathlib import Path

import torch

# Test the real checkpoint-state methods without importing CUDA kernels.
source = Path(__file__).resolve().parents[2] / "lightx2v/common/ops/norm/rms_norm_weight.py"
cls = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "RMSWeightFusedQKNorm3DRope")
cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in {"__init__", "load", "state_dict"}]
namespace = {}
exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
RMSWeightFusedQKNorm3DRope = namespace["RMSWeightFusedQKNorm3DRope"]


def test_fused_qk_norm_exposes_all_source_tensors():
    names = ["q_t", "q_hw", "k_t", "k_hw"]
    module = RMSWeightFusedQKNorm3DRope(*names)
    tensors = {name: torch.randn(8) for name in names}
    module.load(tensors)

    state = module.state_dict()

    assert list(state) == names
    assert all(state[name] is tensors[name] for name in names)
