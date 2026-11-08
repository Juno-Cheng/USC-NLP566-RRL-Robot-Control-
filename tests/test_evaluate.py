import csv
from pathlib import Path

import pytest

from src.evaluate import (
    CSV_FIELDS,
    discover_runs,
    final_distance,
    is_seen,
    parse_args,
    write_eval_csv,
)


def test_parse_args():
    args = parse_args(["--eval-config", "configs/eval.yaml", "--results-dir", "results"])
    assert args.eval_config == "configs/eval.yaml"
    assert args.results_dir == "results"


def test_final_distance_from_reward_dist():
    assert final_distance({"reward_dist": -0.04}) == pytest.approx(0.04)


def test_final_distance_requires_reward_dist():
    with pytest.raises(KeyError):
        final_distance({})


@pytest.mark.parametrize(
    "condition,corruption_id,intensity,expected",
    [
        ("C0", "clean", 0.0, True),
        ("C0", "gaussian", 0.0, True),
        ("C0", "gaussian", 0.1, False),
        ("C1", "gaussian", 0.1, True),
        ("C1", "dropout", 0.2, False),
        ("C2", "dropout", 0.2, True),
        ("C2", "uniform", 0.1, False),
        ("C3", "clean", 0.0, True),
        ("C3", "uniform", 0.1, False),
    ],
)
def test_is_seen_matches_spec(condition, corruption_id, intensity, expected):
    assert is_seen(condition, corruption_id, intensity) is expected


def test_discover_runs_and_write_csv(tmp_path):
    run_dir = tmp_path / "C0_clean_seed0"
    run_dir.mkdir()
    (run_dir / "model.zip").write_bytes(b"zip")
    assert discover_runs(tmp_path) == [run_dir]

    rows = [
        {
            "run_id": "C0_clean_seed0",
            "condition": "C0",
            "name": "clean",
            "corruption_id": "clean",
            "corruption_type": "none",
            "intensity": 0.0,
            "seen": True,
            "mean_return": -1.0,
            "std_across_seeds": 0.1,
            "mean_success_rate": 0.5,
        }
    ]
    path = run_dir / "eval.csv"
    write_eval_csv(path, rows)
    with open(path, newline="") as f:
        loaded = list(csv.DictReader(f))
    assert list(loaded[0].keys()) == list(CSV_FIELDS)
    assert loaded[0]["condition"] == "C0"
