"""分割の正しさのテスト: どの分割点で切っても tail(*head(x)) が元のモデルの出力と一致すること。"""

import pytest
import torch
import torchvision

from splitcomp.models import build_model
from splitcomp.split import Splitter, split_sequential


def _check_all_points(model, x, trace_depth=None, kind="all"):
    model.eval()
    with torch.no_grad():
        ref = model(x)
    splitter = Splitter(model, trace_depth)
    points = splitter.split_points(kind)
    assert points, "分割点が 1 つもない"
    for p in points:
        head, tail = splitter.split(p)
        with torch.no_grad():
            z = head(x)
            assert isinstance(z, tuple) and len(z) == p.num_tensors
            out = tail(*z)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4, msg=f"分割点 {p.name} で出力が一致しない")
    return splitter


def test_toy_cnn_all_points():
    splitter = _check_all_points(build_model("toy_cnn"), torch.randn(2, 3, 16, 16))
    # 残差接続の途中では 2 つのテンソルを送る必要がある
    assert any(p.num_tensors == 2 for p in splitter.split_points("all"))


def test_toy_mlp_all_points():
    _check_all_points(build_model("toy_mlp"), torch.randn(4, 32))


@pytest.mark.parametrize("name", ["resnet18", "mobilenet_v3_small"])
def test_torchvision_clean_points(name):
    model = torchvision.models.get_model(name)
    _check_all_points(model, torch.randn(1, 3, 64, 64), kind="clean")


def test_trace_depth_gives_coarse_points():
    model = torchvision.models.get_model("resnet18")
    splitter = _check_all_points(model, torch.randn(1, 3, 64, 64), trace_depth=1)
    names = [p.name for p in splitter.split_points("clean")]
    for layer in ["layer1", "layer2", "layer3", "layer4"]:
        assert layer in names


def test_aliases_and_unknown_point():
    splitter = Splitter(build_model("toy_cnn"))
    x = torch.randn(1, 3, 8, 8)
    head, tail = splitter.split("input")
    (z,) = head(x)
    torch.testing.assert_close(z, x)  # 入力をそのまま送る
    head, tail = splitter.split("output")
    assert head(x)[0].shape == (1, 10)  # 出力を送る
    with pytest.raises(KeyError):
        splitter.get_point("no_such_layer")


def test_split_sequential():
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)).eval()
    x = torch.randn(3, 4)
    with pytest.warns(UserWarning):
        head, tail = split_sequential(model, 2)
    torch.testing.assert_close(tail(*head(x)), model(x))
