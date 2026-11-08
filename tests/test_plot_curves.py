import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_plot_module():
    path = ROOT / "scripts" / "plot_curves.py"
    spec = importlib.util.spec_from_file_location("plot_curves", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plot_and_seen_unseen_table(tmp_path):
    plot_curves = load_plot_module()
    rows = [
        {
            "run_id": "C0_clean_seed0",
            "condition": "C0",
            "name": "clean",
            "corruption_id": "clean",
            "corruption_type": "none",
            "intensity": 0.0,
            "seen": True,
            "mean_return": -4.0,
            "std_across_seeds": 0.1,
            "mean_success_rate": 0.5,
        },
        {
            "run_id": "C0_clean_seed0",
            "condition": "C0",
            "name": "clean",
            "corruption_id": "gaussian",
            "corruption_type": "gaussian",
            "intensity": 0.1,
            "seen": False,
            "mean_return": -8.0,
            "std_across_seeds": 0.1,
            "mean_success_rate": 0.2,
        },
        {
            "run_id": "C0_clean_seed0",
            "condition": "C0",
            "name": "clean",
            "corruption_id": "dropout",
            "corruption_type": "dropout",
            "intensity": 0.2,
            "seen": False,
            "mean_return": -9.0,
            "std_across_seeds": 0.1,
            "mean_success_rate": 0.2,
        },
        {
            "run_id": "C0_clean_seed0",
            "condition": "C0",
            "name": "clean",
            "corruption_id": "uniform",
            "corruption_type": "uniform",
            "intensity": 0.1,
            "seen": False,
            "mean_return": -10.0,
            "std_across_seeds": 0.1,
            "mean_success_rate": 0.1,
        },
    ]
    eval_path = tmp_path / "C0_clean_seed0" / "eval.csv"
    eval_path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(eval_path, index=False)

    df = plot_curves.load_eval_tables(tmp_path)
    out = plot_curves.plot_robustness_curves(df, tmp_path / "robustness_curves.png")
    assert Path(out).is_file()
    table = plot_curves.seen_unseen_table(df)
    row = table.iloc[0]
    assert row["condition"] == "C0"
    assert row["clean_return"] == pytest.approx(-4.0)
    assert row["unseen_absolute_drop"] == pytest.approx(5.0)
