import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

import torch.nn as nn  # noqa: E402

from leukemia_ml.config import ModelConfig  # noqa: E402
from leukemia_ml.models.encoders import build_encoder, load_dinobloom_state, set_trainable  # noqa: E402
from leukemia_ml.models.heads import GatedAttentionMIL, MeanPoolHead, masked_softmax  # noqa: E402
from leukemia_ml.models.hybrid import FeatureBranch  # noqa: E402
from leukemia_ml.models.lora import LoRALinear, apply_lora, lora_parameters  # noqa: E402
from leukemia_ml.models.model import LeukemiaModel, build_model, pad_bags  # noqa: E402

ARCH = "vit_small_patch14_dinov2.lvd142m"


# ------------------------------------------------------------------------------- LoRA
def test_lora_is_identity_at_init_and_merge_roundtrips():
    torch.manual_seed(0)
    base = nn.Linear(8, 6)
    x = torch.randn(5, 8)
    lora = LoRALinear(base, rank=2, alpha=4)
    assert torch.allclose(lora(x), base(x))
    nn.init.normal_(lora.B)
    out = lora(x)
    assert not torch.allclose(out, base(x))
    lora.merge()
    assert torch.allclose(lora(x), out, atol=1e-5)
    lora.unmerge()
    assert torch.allclose(lora(x), out, atol=1e-5)


def test_lora_trains_adapters_only():
    torch.manual_seed(0)
    base = nn.Linear(8, 6)
    w0 = base.weight.detach().clone()
    lora = LoRALinear(base, 2, 4)
    opt = torch.optim.SGD([p for p in lora.parameters() if p.requires_grad], lr=0.1)
    for _ in range(3):
        opt.zero_grad()
        lora(torch.randn(4, 8)).pow(2).sum().backward()
        opt.step()
    assert torch.equal(base.weight, w0) and base.weight.grad is None
    assert lora.B.abs().sum() > 0


def test_apply_lora_wraps_vit_attention_and_counts_params():
    vit = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0, img_size=32)
    n_blocks = len(vit.blocks)
    names = apply_lora(vit, ("attn.qkv", "attn.proj"), rank=4, alpha=8)
    assert len(names) == 2 * n_blocks
    for p in vit.parameters():
        p.requires_grad = False
    for n, p in vit.named_parameters():
        p.requires_grad = n.endswith((".A", ".B"))
    assert len(lora_parameters(vit)) == 4 * n_blocks
    with pytest.raises(ValueError):
        apply_lora(vit, ("does.not.exist",), 2, 4)


# ------------------------------------------------------------------------- encoders
def _write_fake_dinobloom(path, img_size=224):
    arch = timm.create_model(ARCH, pretrained=False, num_classes=0, img_size=img_size)
    sd = {"backbone." + k: v for k, v in arch.state_dict().items()}
    sd["backbone.mask_token"] = torch.zeros(1, 384)
    sd["ibot_head.mlp.0.weight"] = torch.zeros(2, 2)
    sd["dino_head.last_layer.weight_g"] = torch.zeros(2, 1)
    torch.save({"teacher": sd}, path)
    return arch


def test_dinobloom_checkpoint_loads_exactly(tmp_path):
    p = tmp_path / "DinoBloom-S.pth"
    arch = _write_fake_dinobloom(p)
    sd = load_dinobloom_state(p)
    assert not any(k.startswith(("ibot", "dino", "mask")) for k in sd)
    enc = build_encoder("dinobloom_s", 224, True, str(p)).eval()
    x = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        ref = arch.eval()((x - enc.mean) / enc.std)
        assert torch.allclose(enc(x), ref, atol=1e-5)
    assert enc.out_dim == 384


def test_dinobloom_resamples_position_embedding(tmp_path):
    p = tmp_path / "w.pth"
    _write_fake_dinobloom(p)
    enc = build_encoder("dinobloom_s", 112, True, str(p)).eval()
    assert enc.backbone.pos_embed.shape[1] == 1 + 8 * 8
    with torch.no_grad():
        e = enc(torch.rand(2, 3, 112, 112))
    assert e.shape == (2, 384) and torch.isfinite(e).all()


def test_dinobloom_errors(tmp_path):
    with pytest.raises(ValueError, match="weights_path"):
        build_encoder("dinobloom_s", 112, True, None)
    p = tmp_path / "bad.pth"
    arch = timm.create_model(ARCH, pretrained=False, num_classes=0, img_size=224)
    sd = {"backbone." + k: v for k, v in arch.state_dict().items() if "blocks.3.attn.qkv" not in k}
    torch.save({"teacher": sd}, p)
    with pytest.raises(RuntimeError, match="mismatch"):
        build_encoder("dinobloom_s", 224, True, str(p))
    with pytest.raises(ValueError, match="multiple of 14"):
        build_encoder("dinov2_s", 100, False)
    with pytest.raises(ValueError):
        build_encoder("nonsense", 112, False)


def test_cnn_encoder_and_trainable_modes():
    enc = build_encoder("timm:resnet10t", pretrained=False)
    assert enc(torch.rand(2, 3, 64, 64)).shape == (2, enc.out_dim)
    set_trainable(enc, "frozen")
    assert not any(p.requires_grad for p in enc.parameters())
    set_trainable(enc, "full")
    assert all(p.requires_grad for p in enc.parameters())
    vit = build_encoder("dinov2_s", 28, False)
    set_trainable(vit, "last_k", last_k=2)
    flags = [any(p.requires_grad for p in b.parameters()) for b in vit.blocks]
    assert flags == [False] * (len(flags) - 2) + [True, True]
    with pytest.raises(ValueError):
        set_trainable(enc, "last_k")


# --------------------------------------------------------------------------- heads
def _bag(b=3, k=5, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(b, k, d, generator=g)
    mask = torch.zeros(b, k, dtype=torch.bool)
    for i, n in enumerate([5, 3, 1][:b]):
        mask[i, :n] = True
    return h, mask


def test_masked_softmax_normalises_valid_cells_only():
    h, mask = _bag()
    a = masked_softmax(torch.randn(3, 5), mask)
    assert torch.allclose(a.sum(1), torch.ones(3))
    assert (a[~mask] == 0).all()
    assert torch.isfinite(masked_softmax(torch.randn(1, 4), torch.zeros(1, 4, dtype=torch.bool))).all()


def test_mil_is_permutation_invariant_and_padding_independent():
    torch.manual_seed(0)
    head = GatedAttentionMIL(16, 4, hidden=32).eval()
    h, mask = _bag()
    logits, attn = head(h, mask)
    perm = torch.tensor([4, 2, 0, 3, 1])
    h2, m2 = h.clone(), mask.clone()
    h2[0], m2[0] = h[0][perm], mask[0][perm]
    assert torch.allclose(head(h2, m2)[0][0], logits[0], atol=1e-5)
    # extra padding columns change nothing
    h3 = torch.cat([h, torch.randn(3, 4, 16)], dim=1)
    m3 = torch.cat([mask, torch.zeros(3, 4, dtype=torch.bool)], dim=1)
    assert torch.allclose(head(h3, m3)[0], logits, atol=1e-5)
    assert torch.allclose(attn.sum(1), torch.ones(3), atol=1e-5)


def test_mil_ignores_cell_count_duplication():
    """Softmax-normalised attention: repeating every cell must not change the prediction."""
    torch.manual_seed(1)
    head = GatedAttentionMIL(16, 4, hidden=32).eval()
    h = torch.randn(1, 4, 16)
    mask = torch.ones(1, 4, dtype=torch.bool)
    a = head(h, mask)[0]
    b = head(torch.cat([h, h], dim=1), torch.ones(1, 8, dtype=torch.bool))[0]
    assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("cls", [GatedAttentionMIL, MeanPoolHead])
def test_heads_backprop_everywhere_and_no_nan_on_empty_bag(cls):
    head = cls(16, 4, hidden=32)
    h, mask = _bag()
    logits, _ = head(h, mask)
    logits.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    out, attn = head(torch.zeros(1, 3, 16), torch.zeros(1, 3, dtype=torch.bool))
    assert torch.isfinite(out).all() and torch.isfinite(attn).all()


# -------------------------------------------------------------------------- hybrid
def test_feature_branch_group_dropout_and_ablation():
    torch.manual_seed(0)
    br = FeatureBranch(6, 8, [slice(0, 3), slice(3, 6)], group_dropout=0.5)
    f = torch.ones(4000, 6)
    br.train()
    # group dropout zeros WHOLE groups: inspect through a linear probe of the input
    dropped = []
    orig = br.net.forward
    br.net.forward = lambda x: (dropped.append(x.clone()), orig(x))[1]
    br(f)
    x = dropped[0]
    assert ((x[:, :3] == 0).all(1) | (x[:, :3] == 1).all(1)).all()
    assert 0.4 < (x[:, 0] == 0).float().mean() < 0.6
    br.net.forward = orig
    br.eval()
    a, b = br(f[:5]), br(f[:5], zero_groups=(1,))
    assert not torch.allclose(a, b) and torch.allclose(br(f[:5]), a)


# ---------------------------------------------------------------------------- model
def test_model_forward_backward_and_param_groups():
    cfg = ModelConfig(encoder="timm:resnet10t", pretrained=False, freeze="full", head="mil",
                      use_features=True, feature_hidden=8, hidden=32)
    model = build_model(cfg, n_classes=4, n_features=5)
    crops = torch.rand(6, 3, 32, 32)
    n = torch.tensor([4, 2])
    bag = torch.repeat_interleave(torch.arange(2), n)
    pos = torch.cat([torch.arange(4), torch.arange(2)])
    logits, attn = model(crops, torch.randn(6, 5), bag, pos, n)
    assert logits.shape == (2, 4) and attn.shape == (2, 4)
    assert attn[1, 2:].abs().sum() == 0
    logits.sum().backward()
    from leukemia_ml.config import TrainConfig
    groups = model.trainable_groups(TrainConfig(lr_head=1e-3, lr_encoder=1e-5))
    assert {g["name"] for g in groups} == {"head", "branch", "encoder"}
    assert [g["lr"] for g in groups if g["name"] == "encoder"] == [1e-5]


def test_frozen_encoder_builds_no_graph_and_lora_model_has_adapters():
    frozen = build_model(ModelConfig(encoder="timm:resnet10t", pretrained=False, freeze="frozen"), 4)
    assert not any(p.requires_grad for p in frozen.encoder.parameters())
    out = frozen.embed_cells(torch.rand(3, 3, 32, 32))
    assert not out.requires_grad
    lora = build_model(ModelConfig(encoder="dinov2_s", pretrained=False, input_size=28,
                                   freeze="lora", lora_rank=4), 4)
    names = [n for n, p in lora.encoder.named_parameters() if p.requires_grad]
    assert names and all(n.endswith((".A", ".B")) for n in names)


def test_pad_bags_layout():
    h = torch.arange(10, dtype=torch.float32).view(5, 2)
    padded, mask = pad_bags(h, torch.tensor([0, 0, 0, 1, 1]), torch.tensor([0, 1, 2, 0, 1]), 2, 3)
    assert padded.shape == (2, 3, 2) and mask.tolist() == [[True] * 3, [True, True, False]]
    assert torch.equal(padded[1, 1], h[4]) and padded[1, 2].abs().sum() == 0


def test_model_config_validation():
    with pytest.raises(ValueError):
        ModelConfig(freeze="nope")
    with pytest.raises(ValueError):
        ModelConfig(head="nope")
    with pytest.raises(ValueError):
        ModelConfig(lora_rank=0)
