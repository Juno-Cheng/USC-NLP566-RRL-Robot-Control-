
"""Plot robustness curves and a seen-vs-unseen table from eval.csv files.

    results/*/eval.csv  --plot_curves.py-->  robustness_curves.png, seen_vs_unseen.csv

CLI (called by scripts/run_all.sh):

    python scripts/plot_curves.py --results-dir results
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.metrics import compute_degradation

CORRUPTION_TYPES = ("gaussian", "dropout", "uniform")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Plot robustness curves from eval.csv files.")
    parser.add_argument("--results-dir", required=True)
    return parser.parse_args(argv)


def _as_bool(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1"])


def load_eval_tables(results_dir):
    paths = sorted(Path(results_dir).glob("*/eval.csv"))
    if not paths:
        raise FileNotFoundError(f"no eval.csv under {results_dir}")
    df = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    df["seen"] = _as_bool(df["seen"])
    return df


def plot_robustness_curves(df, out_path):
    fig, axes = plt.subplots(1, len(CORRUPTION_TYPES), figsize=(12, 4), sharey=True)
    for ax, corruption_type in zip(axes, CORRUPTION_TYPES):
        subset = df[df["corruption_type"] == corruption_type].sort_values("intensity")
        for (condition, name), group in subset.groupby(["condition", "name"]):
            ax.plot(
                group["intensity"],
                group["mean_return"],
                marker="o",
                label=f"{condition}:{name}",
            )
        ax.set_title(corruption_type)
        ax.set_xlabel("intensity")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("mean return")
    axes[-1].legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def seen_unseen_table(df):
    rows = []
    for (condition, name), group in df.groupby(["condition", "name"]):
        clean = group[group["corruption_id"] == "clean"]
        if clean.empty:
            raise ValueError(f"{condition}:{name} has no clean eval row")
        clean_return = float(clean["mean_return"].mean())
        seen = group[group["seen"]]
        unseen = group[~group["seen"] & (group["intensity"] > 0)]
        seen_return = float(seen["mean_return"].mean()) if not seen.empty else None
        unseen_return = float(unseen["mean_return"].mean()) if not unseen.empty else None
        drop = (
            compute_degradation(clean_return, unseen_return)
            if unseen_return is not None
            else {"absolute_drop": None, "relative_drop": None}
        )
        rows.append(
            {
                "condition": condition,
                "name": name,
                "clean_return": clean_return,
                "seen_return": seen_return,
                "unseen_return": unseen_return,
                "unseen_absolute_drop": drop["absolute_drop"],
                "unseen_relative_drop": drop["relative_drop"],
            }
        )
    return pd.DataFrame(rows)


def main(argv=None):
    args = parse_args(argv)
    results_dir = Path(args.results_dir)
    df = load_eval_tables(results_dir)
    curves_path = plot_robustness_curves(df, results_dir / "robustness_curves.png")
    table = seen_unseen_table(df)
    table_path = results_dir / "seen_vs_unseen.csv"
    table.to_csv(table_path, index=False)
    print(table.to_string(index=False))
    print(f"wrote {curves_path}")
    print(f"wrote {table_path}")


if __name__ == "__main__":
    main()
