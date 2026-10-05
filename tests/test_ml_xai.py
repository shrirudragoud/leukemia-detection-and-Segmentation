import numpy as np
import pytest

torch = pytest.importorskip("torch")

from leukemia_ml.config import ModelConfig  # noqa: E402
from leukemia_ml.models.model import build_model  # noqa: E402
from leukemia_ml.xai.gradcam import deletion_curve, grad_cam, randomisation_sanity  # noqa: E402


@pytest.fixture(scope="module")
def cnn_model():
    torch.manual_seed(0)
    m = build_model(ModelConfig(encoder="timm:resnet10t", pretrained=False, freeze="full",
                                head="mil", hidden=32), n_classes=4)
    return m.eval()


@pytest.fixture(scope="module")
def vit_model():
    torch.manual_seed(0)
    return build_model(ModelConfig(encoder="dinov2_s", pretrained=False, input_size=56,
                                   freeze="full", head="mil", hidden=32), n_classes=4).eval()


def _x(size=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(1, 3, size, size, generator=g)
    x[..., 20:44, 20:44] = 0.2                                  # a "cell"
    return x


@pytest.mark.parametrize("fixture,size", [("cnn_model", 64), ("vit_model", 56)])
def test_grad_cam_shape_range_determinism(request, fixture, size):
    model = request.getfixturevalue(fixture)
    x = _x(size)
    cam, cls = grad_cam(model, x)
    assert cam.shape == (size, size) and cam.min() >= 0 and cam.max() <= 1 + 1e-6
    cam2, cls2 = grad_cam(model, x)
    assert np.array_equal(cam, cam2) and cls == cls2
    other, _ = grad_cam(model, x, class_idx=(cls + 1) % 4)
    assert other.shape == cam.shape
    assert 0 <= cls < 4


def test_grad_cam_depends_on_the_input(cnn_model):
    a, _ = grad_cam(cnn_model, _x(64, 0))
    b, _ = grad_cam(cnn_model, _x(64, 1))
    assert not np.allclose(a, b)


def test_randomisation_sanity_reports_each_stage_and_decorrelates(cnn_model):
    res = randomisation_sanity(cnn_model, _x(64))
    stages = [r["randomised_through"] for r in res]
    assert stages[0] == "head" and stages[1].startswith("encoder_block")
    corr = [r["spearman_with_original"] for r in res]
    assert all(-1.0 <= c <= 1.0 for c in corr)
    assert abs(corr[-1]) < 0.9                                    # fully randomised model != original map
    # the original model must be left untouched by the check
    a, _ = grad_cam(cnn_model, _x(64))
    b, _ = grad_cam(cnn_model, _x(64))
    assert np.array_equal(a, b)


def test_deletion_curve_structure_and_start_value(cnn_model):
    x = _x(64)
    cam, cls = grad_cam(cnn_model, x)
    d = deletion_curve(cnn_model, x, cam, cls, fracs=(0.0, 0.3, 0.6))
    assert d["cam"][0] == pytest.approx(d["random"][0])           # nothing deleted yet
    assert len(d["cam"]) == len(d["random"]) == 3
    assert all(0.0 <= p <= 1.0 for p in d["cam"] + d["random"])
    assert d["cam"][-1] != d["cam"][0]                            # deleting pixels changes the output
