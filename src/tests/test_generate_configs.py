from __future__ import annotations

import pandas as pd
import yaml

from scripts.generate_experiment_configs import generate_round1, generate_round2, generate_round3


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
    assert generate_round3(out, meta) == 45
    assert len(list((out / "round1").glob("*.yaml"))) == 162
    assert len(list((out / "round2").glob("*.yaml"))) == 108
    assert len(list((out / "round3").glob("*.yaml"))) == 45

    sample = yaml.safe_load((out / "round2" / "ecbit_no_blockmask_medium_r40_s42.yaml").read_text())
    assert sample["model"]["variant"] == "no_blockmask"
    assert sample["missing"]["pattern"] == "mcar"
    assert sample["missing"]["target_pattern"] == "medium"
    heldout = yaml.safe_load((out / "round3" / "ecbit_full_heldout_a_short_s42.yaml").read_text())
    assert heldout["data"]["test_station_ids"] == ["a"]
    assert "station_ids" not in heldout["data"]
