import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from test_ml_data import dcfg, synth  # noqa: E402,F401

from leukemia_ml.cli import main  # noqa: E402
from leukemia_ml.config import ExperimentConfig  # noqa: E402
from leukemia_ml.experiments import _apply, render_markdown, run_probe_ablation  # noqa: E402


def base(synth):
    return ExperimentConfig(data=dcfg(synth), name="abl")


VARIANTS = [
    {"name": "shape (tabular)", "tabular": True, "data": {"feature_groups": ["cell_shape"]}},
    {"name": "texture (tabular)", "tabular": True, "data": {"feature_groups": ["cell_texture"]}},
    {"name": "shape+texture (tabular)", "tabular": True,
     "data": {"feature_groups": ["cell_shape", "cell_texture"]}},
]


def test_apply_overrides_do_not_mutate_base(synth):
    b = base(synth)
    v = _apply(b, {"data": {"task": "binary"}, "model": {"input_size": 56}})
    assert v.data.task == "binary" and v.model.input_size == 56
    assert b.data.task == "4way" and b.model.input_size == 112


def test_probe_ablation_structure_ci_and_determinism(synth, tmp_path):
    r1 = run_probe_ablation(base(synth), VARIANTS, tmp_path / "a", seeds=(0, 1), n_boot=200, scheme="grouped", embargo=0)
    run_probe_ablation(base(synth), VARIANTS, tmp_path / "b", seeds=(0, 1), n_boot=200, scheme="grouped", embargo=0)
    assert [r["variant"] for r in r1["rows"]] == [v["name"] for v in VARIANTS]
    assert r1["reference"] == "shape (tabular)"
    for row in r1["rows"]:
        ci = row["balanced_accuracy_ci"]
        assert ci["lo"] <= ci["point"] <= ci["hi"] and row["n_images"] == 48
    assert "vs_reference" not in r1["rows"][0]
    for row in r1["rows"][1:]:
        cmp = row["vs_reference"]
        assert cmp["lo"] <= cmp["hi"] and 0 < cmp["p"] <= 1 and cmp["p_holm"] >= cmp["p"]
    # cell size encodes the class in the synthetic data, so shape features must be informative
    assert r1["rows"][0]["balanced_accuracy"] > 0.6
    assert (tmp_path / "a" / "ablation.md").exists()
    assert json.loads((tmp_path / "a" / "ablation.json").read_text())["rows"] == \
        json.loads((tmp_path / "b" / "ablation.json").read_text())["rows"]


def test_markdown_mentions_reference_and_ci(synth, tmp_path):
    rep = run_probe_ablation(base(synth), VARIANTS[:2], tmp_path, seeds=(0,), n_boot=100, scheme="grouped", embargo=0)
    md = render_markdown(rep)
    assert "reference: shape (tabular)" in md and "Holm" in md and "texture (tabular)" in md


def test_cli_ablate(synth, tmp_path, capsys):
    spec = {"base": {"data": json.loads(json.dumps(dcfg(synth).__dict__, default=list))},
            "variants": VARIANTS[:2]}
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    assert main(["ablate", "--spec", str(p), "--out", str(tmp_path / "o"), "--seeds", "1",
                 "--boot", "50", "--scheme", "grouped", "--embargo", "0"]) == 0
    assert "Balanced acc." in capsys.readouterr().out


def test_contiguous_is_the_default_scheme_and_is_recorded(synth, tmp_path):
    rep = run_probe_ablation(base(synth), VARIANTS[:2], tmp_path, seeds=(0,), n_boot=50, embargo=0)
    assert rep["scheme"] == "contiguous" and rep["embargo"] == 0
    assert "contiguous 5-fold CV" in rep["protocol"]
    assert "contiguous folds, embargo 0" in (tmp_path / "ablation.md").read_text()
    assert all(r["n_images"] == 48 for r in rep["rows"])
