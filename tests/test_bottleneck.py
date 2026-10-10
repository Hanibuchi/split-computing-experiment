"""ボトルネックのテスト: 形の整合性、学習で損失が下がること、チェックポイントの保存・読込。"""

import copy

import pytest
import torch

from splitcomp.builder import build_split, save_bottleneck
from splitcomp.compression import attach_bottleneck, build_bottleneck, infer_feature_shape
from splitcomp.config import BottleneckConfig, ModelConfig, TrainConfig
from splitcomp.models import build_model
from splitcomp.split import Splitter
from splitcomp.training import BottleneckTrainer


@pytest.mark.parametrize("downsample", [1, 2, 4])
def test_conv_bottleneck_shapes(downsample):
    enc, dec = build_bottleneck([32, 15, 15], BottleneckConfig(channels=4, downsample=downsample))
    x = torch.randn(2, 32, 15, 15)
    z = enc(x)
    assert z.shape[1] == 4 and z.shape[2] == -(-15 // downsample)
    assert dec(z).shape == x.shape  # 奇数サイズでも元の形に戻る


def test_linear_bottleneck_shapes():
    enc, dec = build_bottleneck([64], BottleneckConfig(channels=8))
    x = torch.randn(3, 64)
    assert enc(x).shape == (3, 8) and dec(enc(x)).shape == x.shape


def test_non_clean_point_rejected():
    splitter = Splitter(build_model("toy_cnn"))
    point = next(p for p in splitter.split_points("all") if p.num_tensors == 2)
    head, _ = splitter.split(point)
    with pytest.raises(ValueError):
        infer_feature_shape(head, torch.randn(1, 3, 8, 8))


def test_training_reduces_loss_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model_cfg = ModelConfig(name="toy_cnn")
    teacher = build_model(model_cfg.name)
    head, tail = Splitter(copy.deepcopy(teacher)).split("add")
    x = torch.randn(64, 3, 16, 16)
    feature_shape = infer_feature_shape(head, x[:1])
    bcfg = BottleneckConfig(channels=4)
    enc, dec = build_bottleneck(feature_shape, bcfg)

    cfg = TrainConfig(model=model_cfg, split_point="add", bottleneck=bcfg, lr=1e-2, distill="feature")
    trainer = BottleneckTrainer(cfg, teacher, head, tail, enc, dec, torch.device("cpu"))
    loader = [(x[i : i + 16], torch.zeros(16, dtype=torch.long)) for i in range(0, 64, 16)]
    first = trainer.train_epoch(loader, 1)["loss"]
    for e in range(2, 8):
        last = trainer.train_epoch(loader, e)["loss"]
    assert last < first * 0.7

    # head/tail は固定されているので重みが変わっていないこと
    for p_student, p_teacher in zip(trainer.head.parameters(), trainer.teacher_head.parameters()):
        torch.testing.assert_close(p_student, p_teacher)

    ckpt = tmp_path / "bn.pt"
    save_bottleneck(ckpt, model_cfg, "add", bcfg, feature_shape, trainer.encoder, trainer.decoder)
    # 保存したチェックポイントからは、学習時と同じ構成の head/tail が復元される。
    # （toy_cnn は乱数初期化なので、ベースの重みを学習時の教師と揃えてから比較する）
    bundle = build_split(ModelConfig(name="toy_cnn"), "", str(ckpt))
    assert bundle.has_bottleneck and bundle.point.name == "add"
    bundle.head.head.load_state_dict(trainer.head.state_dict())
    bundle.tail.tail.load_state_dict(trainer.tail.state_dict())
    h2, t2 = attach_bottleneck(trainer.head, trainer.tail, trainer.encoder, trainer.decoder)
    h2.eval(), t2.eval()
    with torch.no_grad():
        torch.testing.assert_close(bundle.tail(*bundle.head(x[:4])), t2(*h2(x[:4])))
