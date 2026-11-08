# Seung Hwan Jung
"""Evaluate each trained run under configs/eval.yaml corruptions.

    results/<run_id>/model.zip  --evaluate.py-->  results/<run_id>/eval.csv

Success uses info["reward_dist"] (minus true distance), not the observation.
Seed-level returns go through compute_metrics, then aggregate_seeds.

CLI (called by scripts/run_all.sh):

    python src/evaluate.py --eval-config configs/eval.yaml --results-dir results
"""

import argparse
import csv
from pathlib import Path

import gymnasium
import yaml
from stable_baselines3 import PPO

from src.corruptions import wrap_env
from src.metrics import aggregate_seeds, compute_metrics, compute_success
from src.rolldrop import RollDropPolicy

SEEN_BY_CONDITION = {
    "C0": {"clean"},
    "C1": {"clean", "gaussian"},
    "C2": {"clean", "dropout"},
    "C3": {"clean"},
}

CSV_FIELDS = (
    "run_id",
    "condition",
    "name",
    "corruption_id",
    "corruption_type",
    "intensity",
    "seen",
    "mean_return",
    "std_across_seeds",
    "mean_success_rate",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate trained runs under corruptions.")
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--results-dir", required=True)
    return parser.parse_args(argv)


def load_eval_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def discover_runs(results_dir):
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return []
    return sorted(
        path.parent
        for path in results_dir.glob("*/model.zip")
    )


def is_seen(condition, corruption_id, intensity):
    """Spec A3: intensity 0 is identity. Unseen is any other type with intensity > 0."""
    if intensity == 0 or corruption_id == "clean":
        return True
    return corruption_id in SEEN_BY_CONDITION.get(condition, {"clean"})


def final_distance(info):
    """True fingertip-to-target distance from the simulator, not the corrupted obs."""
    if "reward_dist" not in info:
        raise KeyError("info is missing reward_dist")
    return float(-info["reward_dist"])


def load_model(model_path):
    # Importing RollDropPolicy above lets PPO.load reconstruct C3.
    return PPO.load(str(model_path), device="cpu")


def run_episode(model, env, seed, deterministic, success_distance):
    obs, info = env.reset(seed=seed)
    episode_return = 0.0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
    success = compute_success(final_distance(info), success_distance)
    return episode_return, success


def evaluate_cell(model, eval_config, corruption_type, intensity):
    """One (type, intensity) across eval seeds. Returns aggregate_seeds dict."""
    seed_returns = []
    seed_success_rates = []
    for seed in eval_config["seeds"]:
        env = wrap_env(
            gymnasium.make(eval_config["env_id"]),
            corruption_type,
            intensity,
            seed,
        )
        try:
            episode_returns = []
            success_flags = []
            for episode in range(eval_config["episodes"]):
                ret, ok = run_episode(
                    model,
                    env,
                    seed=seed * 1000 + episode,
                    deterministic=eval_config["deterministic"],
                    success_distance=eval_config["success_distance"],
                )
                episode_returns.append(ret)
                success_flags.append(ok)
            metrics = compute_metrics(episode_returns, success_flags)
            seed_returns.append(metrics["average_return"])
            seed_success_rates.append(metrics["success_rate"])
        finally:
            env.close()
    return aggregate_seeds(seed_returns, seed_success_rates)


def evaluate_run(run_dir, eval_config):
    run_dir = Path(run_dir)
    with open(run_dir / "config.yaml") as f:
        train_config = yaml.safe_load(f)
    model = load_model(run_dir / "model.zip")
    rows = []
    for corruption in eval_config["corruptions"]:
        for intensity in corruption["intensities"]:
            agg = evaluate_cell(model, eval_config, corruption["type"], intensity)
            rows.append(
                {
                    "run_id": run_dir.name,
                    "condition": train_config["condition"],
                    "name": train_config["name"],
                    "corruption_id": corruption["id"],
                    "corruption_type": corruption["type"],
                    "intensity": float(intensity),
                    "seen": is_seen(
                        train_config["condition"],
                        corruption["id"],
                        float(intensity),
                    ),
                    "mean_return": agg["mean_return"],
                    "std_across_seeds": agg["std_across_seeds"],
                    "mean_success_rate": agg["mean_success_rate"],
                }
            )
    return rows


def write_eval_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = parse_args(argv)
    eval_config = load_eval_config(args.eval_config)
    runs = discover_runs(args.results_dir)
    if not runs:
        raise FileNotFoundError(f"no model.zip under {args.results_dir}")
    for run_dir in runs:
        rows = evaluate_run(run_dir, eval_config)
        out = run_dir / "eval.csv"
        write_eval_csv(out, rows)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
