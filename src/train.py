# Chelroy & Juno
"""Train one Reacher-v5 PPO run from a YAML config.

This file is the only training entry point. C0-C3 all go through here.
The YAML picks the condition. This file does not evaluate or plot.

    config.yaml --> make_env --> make_model --> PPO.learn --> results/<run_id>/

    C0 / C3: wrap_env is a no-op (train_corruption.type is none)
    C1 / C2: wrap_env from src/corruptions.py corrupts observations
    C0 / C1 / C2: default PPO MlpPolicy
    C3: custom policy from src/rolldrop.py when rolldrop.dropout_p > 0

CLI (called by scripts/run_all.sh):

    python src/train.py --config configs/clean.yaml
    python src/train.py --config configs/clean.yaml --timesteps 10000 --n-steps 256

--timesteps and --n-steps override the YAML for smoke/pilot runs only.

Writes results/<run_id>/:

- model.zip: SB3 checkpoint, loaded later by evaluate.py
- config.yaml: copy of the config actually used (including CLI overrides)
- manifest.json: seed, package versions, git commit, run_id

run_id: {condition}_{name}_seed{seed}, e.g. C0_clean_seed0.
sweep_rolldrop.py should add _p{value} so C3 sweep runs do not overwrite.

Fill order: load_config and make_env (C0), then make_model + PPO.learn +
write_outputs, then C1/C2 via wrap_env, then C3 once rolldrop.py exposes a
policy with dropout on during rollouts and off during updates.
"""

import yaml
import argparse
import sys
from pathlib import Path


def parse_args(argv=None):
    """Parse CLI args.

    Input: argv list, or None to use sys.argv.
    Output: namespace with config (path), timesteps (int or None),
    n_steps (int or None).

    Pseudocode:

        parser = argparse.ArgumentParser()
        parser.add_argument("--config", required=True)
        parser.add_argument("--timesteps", type=int, default=None)
        parser.add_argument("--n-steps", type=int, default=None, dest="n_steps")
        return parser.parse_args(argv)
    """
    return None


def load_config(path):
    """Load one train YAML into a dict.

    Input: path to clean.yaml, gaussian_aug.yaml, obs_dropout_aug.yaml,
    or rolldrop.yaml.
    Output: dict with condition, name, env_id, algo, seed, PPO fields,
    train_corruption, rolldrop.

    """
      try: 
            with open(path) as f:
                  config = yaml.safe_load(f)
                  return config
      except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {path}")
      except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error loading config file: {path}") from e
    
    


def make_run_id(config):
    """Build the results folder name for this run.

    Input: config dict.
    Output: string, e.g. C0_clean_seed0.

    Pseudocode:

        return f"{config['condition']}_{config['name']}_seed{config['seed']}"
    """
    return None


def make_env(config):
    """Build Reacher-v5 and apply train_corruption.

    Input: config dict.
    Output: Gymnasium env. C0/C3 get the plain env. C1/C2 get wrap_env.

    wrap_env(env, type, intensity, seed) lives in src/corruptions.py.
    It returns env unchanged when type is none or intensity is 0.

    Pseudocode:

        env = gymnasium.make(config["env_id"])
        corr = config["train_corruption"]
        return wrap_env(env, corr["type"], corr["intensity"], config["seed"])
    """
    return None


def make_model(env, config):
    """Build a PPO model on env.

    Input: env from make_env, config dict.
    Output: stable_baselines3.PPO.

    Shared across C0-C3: n_steps, batch_size, n_epochs, learning_rate,
    gamma, net_arch, seed. algo must be ppo.

    C3 hook: if config["rolldrop"]["dropout_p"] > 0, use the policy class
    from src/rolldrop.py. That class is not ready yet (rolldrop.py currently
    drops whole transitions, not network inputs). Until it is, C0-C2 use
    MlpPolicy and C3 should fail clearly rather than train the wrong thing.

    Pseudocode:

        policy_kwargs = {"net_arch": config["net_arch"]}
        policy = "MlpPolicy"
        if config["rolldrop"]["dropout_p"] > 0:
            policy = RollDropPolicy  # from src/rolldrop.py once it exists
        return PPO(
            policy, env,
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            learning_rate=config["learning_rate"],
            gamma=config["gamma"],
            policy_kwargs=policy_kwargs,
            seed=config["seed"],
            tensorboard_log=None,
        )
    """
    return None


def write_outputs(run_dir, model, config):
    """Save model.zip, config.yaml, and manifest.json under run_dir.

    Input: results/<run_id> path, trained model, config dict (already
    with CLI overrides applied).
    Output: None. Creates the directory if needed.

    manifest.json should include condition, name, seed, run_id,
    gymnasium/stable_baselines3/torch versions, and git commit if available.

    Pseudocode:

        run_dir.mkdir(parents=True, exist_ok=True)
        model.save(run_dir / "model.zip")
        write config.yaml copy
        write manifest.json
    """
    return None


def train(config, timesteps=None, n_steps=None):
    """Run one full training job.

    Input: config dict, optional CLI overrides.
    Output: Path to results/<run_id>/.

    Apply overrides, then make_env, make_model, model.learn, write_outputs.
    Do not call evaluate.py from here.

    Pseudocode:

        if timesteps is not None: config["total_timesteps"] = timesteps
        if n_steps is not None: config["n_steps"] = n_steps
        run_dir = Path("results") / make_run_id(config)
        env = make_env(config)
        model = make_model(env, config)
        model.learn(total_timesteps=config["total_timesteps"])
        write_outputs(run_dir, model, config)
        env.close()
        return run_dir
    """
    return None


def main(argv=None):
    """CLI entry. parse_args, load_config, train.

    Pseudocode:

        args = parse_args(argv)
        config = load_config(args.config)
        train(config, timesteps=args.timesteps, n_steps=args.n_steps)
    """

    # Check Args and See What Yaml is needed

    # Load the Yaml
    config = load_config(args.config)


    return None


if __name__ == "__main__":
    main()
