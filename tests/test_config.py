import json

import pytest

from leukemia_pp.config import PipelineConfig


def test_roundtrip_and_hash_stable():
    cfg = PipelineConfig()
    again = PipelineConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert again == cfg
    assert again.hash() == cfg.hash()


def test_partial_override_and_hash_changes():
    cfg = PipelineConfig.from_dict({"segmentation": {"min_cell_area": 50},
                                    "features": {"glcm_distances": [1, 2]}})
    assert cfg.segmentation.min_cell_area == 50
    assert cfg.features.glcm_distances == (1, 2)
    assert cfg.hash() != PipelineConfig().hash()


@pytest.mark.parametrize("bad", [
    {"nope": 1},
    {"segmentation": {"nope": 1}},
    {"split": {"train": 0.5, "val": 0.5, "test": 0.5}},
    {"denoise": {"method": "magic"}},
])
def test_invalid_config_rejected(bad):
    with pytest.raises(ValueError):
        PipelineConfig.from_dict(bad)
