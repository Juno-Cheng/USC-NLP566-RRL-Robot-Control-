# Spec

Decisions below are the working defaults. Config YAML is frozen on **Nov 1, 2026**. After that date, change only seeds, smoke overrides (`--timesteps`, `--n-steps`), and files under `results/`.

## A1. Environment

Gymnasium MuJoCo `Reacher-v5`. One environment for the main study. Observations are corrupted; actions and dynamics stay clean.

## A2. Algorithm

PPO from Stable-Baselines3. C0-C3 share the same optimizer and network size. The only intended differences are the robustness mechanism in each YAML file.

## A3. Conditions and transfer

| ID | Config | Train observations | Extra mechanism |
| --- | --- | --- | --- |
| C0 | `configs/clean.yaml` | clean | none |
| C1 | `configs/gaussian_aug.yaml` | Gaussian noise | none |
| C2 | `configs/obs_dropout_aug.yaml` | observation dropout | none |
| C3 | `configs/rolldrop.yaml` | clean | dropout on during rollouts, off during updates |

Evaluation (`configs/eval.yaml`) runs clean, Gaussian, dropout, and uniform noise at several intensities. Uniform is held out: no training condition uses it.

- C1 seen: clean and Gaussian
- C2 seen: clean and dropout
- C0 and C3 seen: clean only
- Unseen: every other eval corruption with intensity `> 0`

`scripts/sweep_rolldrop.py` sweeps `rolldrop.sweep_p` for C3.

## Success definition

Robustness is mean episodic return vs corruption intensity (robustness curve), plus success rate.

For Reacher, an episode succeeds if the final fingertip-to-target distance is `< 0.05` (last three observation dims).

The study succeeds if a robustness method beats C0 on unseen corruptions without a large drop on clean eval.

## Config schema

Train YAML (`clean.yaml`, `gaussian_aug.yaml`, `obs_dropout_aug.yaml`, `rolldrop.yaml`):

- `condition`, `name`, `env_id`, `algo`, `seed`
- `total_timesteps`, `n_steps`, `batch_size`, `n_epochs`, `learning_rate`, `gamma`, `net_arch`
- `train_corruption.type`: `none` | `gaussian` | `dropout` | `uniform`
- `train_corruption.intensity`: `0` is identity for every type
- `rolldrop.dropout_p`: used by C3; `0` elsewhere
- `rolldrop.sweep_p`: C3 only

Eval YAML (`eval.yaml`):

- `env_id`, `episodes`, `seeds`, `deterministic`, `success_distance`
- `corruptions[]`: `id`, `type`, `intensities`

Each run writes `results/<run_id>/` with `model.zip`, `manifest.json`, `config.yaml`, and `eval.csv`.
