"""Train C3 once for each rolldrop.sweep_p in the YAML.

Does not pick p from uniform results. Each run gets a unique folder
C3_rolldrop_p{value}_seed{seed} so it does not overwrite the main C3 run.

CLI:

    python scripts/sweep_rolldrop.py --config configs/rolldrop.yaml
    python scripts/sweep_rolldrop.py --config configs/rolldrop.yaml --timesteps 10000 --n-steps 256
"""

import argparse
import copy

from src.train import load_config, train


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Sweep Roll-Drop dropout_p for C3.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None, dest="n_steps")
    return parser.parse_args(argv)


def p_name(base_name, dropout_p):
    return f"{base_name}_p{dropout_p}"


def sweep(config, timesteps=None, n_steps=None):
    """Train one C3 run per sweep_p. Returns the list of result dirs."""
    sweep_values = config["rolldrop"]["sweep_p"]
    if not sweep_values:
        raise ValueError("rolldrop.sweep_p is empty")
    run_dirs = []
    for dropout_p in sweep_values:
        run_config = copy.deepcopy(config)
        run_config["rolldrop"]["dropout_p"] = dropout_p
        run_config["name"] = p_name(config["name"], dropout_p)
        run_dirs.append(train(run_config, timesteps=timesteps, n_steps=n_steps))
    return run_dirs


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    run_dirs = sweep(config, timesteps=args.timesteps, n_steps=args.n_steps)
    for run_dir in run_dirs:
        print(f"saved {run_dir}")


if __name__ == "__main__":
    main()
