from __future__ import annotations

import pandas as pd
import yaml

from scripts.generate_experiment_configs import generate_round1, generate_round2, generate_round2_gated, generate_round3
from scripts.generate_sabc_configs import generate_sabc_configs


def test_generate_experiment_config_counts(tmp_path) -> None:
    out = tmp_path / "configs"
    meta = tmp_path / "station_meta.csv"
    pd.DataFrame(
        [
            {"station_id": "a", "split": "heldout"},
            {"station_id": "b", "split": "heldout"},
            {"station_id": "c", "split": "heldout"},
            {"station_id": "d", "split": "heldout"},
            {"station_id": "e", "split": "heldout"},
        ]
    ).to_csv(meta, index=False)

    assert generate_round1(out) == 162
    assert generate_round2(out) == 108
    assert generate_round2_gated(out) == 108
    assert generate_round3(out, meta) == 45
    assert len(list((out / "round1").glob("*.yaml"))) == 162
    assert len(list((out / "round2").glob("*.yaml"))) == 108
    assert len(list((out / "round2_gated").glob("*.yaml"))) == 108
    assert len(list((out / "round3").glob("*.yaml"))) == 45

    sample = yaml.safe_load((out / "round2" / "ecbit_no_blockmask_medium_r40_s42.yaml").read_text())
    assert sample["model"]["variant"] == "no_blockmask"
    assert sample["missing"]["pattern"] == "mcar"
    assert sample["missing"]["target_pattern"] == "medium"
    gated = yaml.safe_load((out / "round2_gated" / "ecbit_full_medium_r40_s42.yaml").read_text())
    assert gated["model"]["fusion_type"] == "gated"
    no_cross = yaml.safe_load((out / "round2_gated" / "ecbit_no_cross_medium_r40_s42.yaml").read_text())
    assert no_cross["model"]["fusion_type"] == "concat"
    heldout = yaml.safe_load((out / "round3" / "ecbit_full_heldout_a_short_s42.yaml").read_text())
    assert heldout["data"]["test_station_ids"] == ["a"]
    assert "station_ids" not in heldout["data"]


def test_generate_sabc_config_counts(tmp_path) -> None:
    out = tmp_path / "sabc_configs"
    meta = tmp_path / "station_meta.csv"
    pd.DataFrame(
        [
            {"station_id": "a", "split": "heldout"},
            {"station_id": "b", "split": "heldout"},
        ]
    ).to_csv(meta, index=False)

    assert generate_sabc_configs(out, meta, include_baseline=True) == 36
    assert len(list(out.glob("*.yaml"))) == 36

    sample = yaml.safe_load((out / "sabc_metadata_heldout_a_medium_r40_s42.yaml").read_text())
    assert sample["runner"] == "train"
    assert sample["data"]["test_station_groups"] == ["heldout"]
    assert sample["data"]["test_station_ids"] == ["a"]
    assert sample["data"]["station_meta_csv"] == "data/station_meta_ecbit.csv"
    assert sample["model"]["name"] == "ecbit_sabc"
    assert sample["model"]["variant"] == "sabc_metadata"
    assert sample["model"]["fusion_type"] == "gated"
    assert sample["missing"]["pattern"] == "medium"
