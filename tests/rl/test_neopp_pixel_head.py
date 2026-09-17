from types import SimpleNamespace

import torch
import torch.nn.functional as F

import ast
from pathlib import Path

# The decoder math needs Torch, but does not need the CUDA attention/MoE imports.
source = Path(__file__).resolve().parents[2] / "lightx2v/models/networks/neopp/infer/transformer_infer.py"
cls = next(node for node in ast.parse(source.read_text()).body
           if isinstance(node, ast.ClassDef) and node.name == "NeoppTransformerInfer")
cls.bases = []
cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef)
            and node.name in {"_fm_head", "_fm_head_pixel", "_patchify_pixels"}]
namespace = {"F": F, "torch": torch}
exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
NeoppTransformerInfer = namespace["NeoppTransformerInfer"]


class _Conv:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def apply(self, value):
        return F.conv2d(value, self.weight, self.bias, padding=1)


def _patchify_pixels(value, token_h, token_w, output_patch_size):
    value = value.reshape(1, 3, token_h, output_patch_size, token_w, output_patch_size)
    value = torch.einsum("bchpwq->bhwpqc", value)
    return value.contiguous().reshape(token_h * token_w, output_patch_size**2 * 3)


def test_u15_pixel_head_matches_conv_decoder_and_patchify():
    torch.manual_seed(7)
    token_h, token_w = 2, 3
    hidden_size = 16
    output_patch_size = 32
    hidden = torch.randn(token_h * token_w, hidden_size)
    weights = SimpleNamespace(
        conv1=_Conv(torch.randn(16, hidden_size // 4, 3, 3), torch.randn(16)),
        conv2=_Conv(torch.randn(192, 16 // 4, 3, 3), torch.randn(192)),
    )

    infer = NeoppTransformerInfer.__new__(NeoppTransformerInfer)
    infer.use_pixel_head = True
    infer.patch_size = 16
    infer.merge_size = 2
    infer.scheduler = SimpleNamespace(
        image_prediction=torch.zeros(1, 3, token_h * output_patch_size, token_w * output_patch_size)
    )

    actual = infer._fm_head(weights, hidden, token_h, token_w)

    image = hidden.reshape(1, token_h, token_w, hidden_size).permute(0, 3, 1, 2).contiguous()
    image = F.pixel_shuffle(image, 2)
    image = F.gelu(weights.conv1.apply(image))
    image = F.pixel_shuffle(image, 2)
    image = weights.conv2.apply(image)
    image = F.pixel_shuffle(image, 8)
    expected = _patchify_pixels(image, token_h, token_w, output_patch_size)

    torch.testing.assert_close(actual, expected)
    assert actual.shape == (token_h * token_w, output_patch_size**2 * 3)
