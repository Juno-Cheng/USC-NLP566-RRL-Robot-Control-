# USC-NLP566-RRL-Robot-Control-

Robust RL for robot control under observation corruptions. We compare four training conditions: clean (C0), Gaussian augmentation (C1), observation-dropout augmentation (C2), and Roll-Drop (C3). Design decisions are in `docs/SPEC.md`.

## Overview

Robot control policies are usually trained in simulation, where sensor readings are perfect. Real sensors are noisy, drop out, and fail in ways you don't always expect. We want to know whether training with imperfect readings makes a policy more robust, and whether that robustness carries over to kinds of noise the policy never saw during training.

Our robot is a simulated two-joint arm (MuJoCo) that learns to move its fingertip to a target. There's no dataset. Like any RL setup, the arm learns through trial and error: at each step the simulator reports the arm and target positions, the policy picks joint torques, and the simulator moves the arm and returns a reward based on how close the fingertip is to the target. Over many episodes the policy learns which actions lead to higher reward.

We train the arm under four conditions:

- C0, clean: perfect sensor readings.
- C1, Gaussian noise: small random errors added to every reading.
- C2, dropout: some readings randomly zeroed out.
- C3, Roll-Drop: clean readings, but the policy network applies dropout to its inputs while collecting experience.

Each condition gets its own network, trained from scratch. The networks never see each other's data and aren't told which condition they're in; they just learn from whatever readings they get. The simulator is identical across all four, and the only difference is what happens to the readings before they reach the network. For C3, we also run a small sweep over dropout rates (`scripts/sweep_rolldrop.py`) to choose one.

After training, we evaluate all four policies under noise at increasing intensity, including uniform noise, which none of the conditions train on. Comparing return and success rate across conditions tells us which method is most robust, and whether that robustness generalizes to unseen noise.

## Task details

- Environment: Gymnasium MuJoCo `Reacher-v5`, 50 steps per episode.
- Observation (policy input): 10 values per step, made up of joint angles (cos and sin), target x/y, joint velocities, and fingertip-minus-target x/y.
- Action (policy output): 2 joint torques.
- Corruption only touches the observation. Actions and physics stay clean.

Pipeline:

```text
configs/<condition>.yaml  --train.py-->        results/<run_id>/model.zip, config.yaml, manifest.json
model.zip + eval.yaml     --evaluate.py-->     results/<run_id>/eval.csv
every eval.csv            --plot_curves.py-->  robustness curves and a seen-vs-unseen table
```

The main comparison: on uniform noise, do C1, C2, or C3 outperform C0 without a large drop on clean observations?

## Where the conditions are implemented

Each condition is defined by a YAML file in `configs/`. All four go through the same `src/train.py`, and the config decides what changes.

- C0 (`configs/clean.yaml`): no corruption. `train.py` uses the plain environment.
- C1 (`configs/gaussian_aug.yaml`) and C2 (`configs/obs_dropout_aug.yaml`): `train.py` reads `train_corruption` and wraps the environment with `wrap_env` from `src/corruptions.py`. C1 uses Gaussian noise at 0.1, C2 uses dropout at 0.2.
- C3 (`configs/rolldrop.yaml`): no environment corruption. When `rolldrop.dropout_p > 0`, `train.py` uses the policy network from `src/rolldrop.py` instead of the default one.

## Build order

Most `.py` files are still empty, and `src/corruptions.py` only has stubs, so Docker and the smoke test won't run yet.

1. `Docker/app.py`: read the stage name (`smoke`, `pilot`, or `full`) and run `sh scripts/run_all.sh <stage>`, so `docker run` kicks off the pipeline.
2. `src/corruptions.py` and `tests/test_corruptions.py`: fill in the corruption stubs and test them.
3. `src/train.py`, starting with C0. Get one clean PPO run training and saving to `results/`.
4. `src/evaluate.py`: load the checkpoint and write `eval.csv` for each corruption and intensity.
5. `tests/test_smoke.py`: a short train-then-evaluate run. Once this passes, `smoke` works.
6. Add C1 and C2 to `train.py` by wrapping the environment based on `train_corruption`. No new files needed.
7. `src/rolldrop.py` and `tests/test_rolldrop.py`, then hook C3 into `train.py`.
8. `scripts/plot_curves.py`: turn the `eval.csv` files into robustness curves and the table.
9. Run `pilot` and fix whatever breaks. Configs freeze on Nov 1, so any tuning needs to happen before then.
10. `scripts/sweep_rolldrop.py`, then the `full` run.

Steps 1-5 cover Phase 1 of the proposal, 6-9 cover Phase 2, and step 10 starts Phase 3.

## File structure

```text
.
├── configs/                  # one YAML per run, frozen on Nov 1
│   ├── clean.yaml            # C0
│   ├── gaussian_aug.yaml     # C1
│   ├── obs_dropout_aug.yaml  # C2
│   ├── rolldrop.yaml         # C3
│   └── eval.yaml             # corruptions, intensities, episodes, eval seeds
├── Docker/
│   ├── Dockerfile
│   ├── app.py                # thin entry point; runs scripts/run_all.sh
│   └── requirements.txt
├── docs/
│   ├── pre-proposal.pdf
│   ├── SPEC.md               # decisions A1-A3, success definition, config schema
│   └── CONTRIBUTIONS.md      # running log of who did what
├── scripts/
│   ├── run_all.sh            # one command per stage: smoke, pilot, full
│   ├── sweep_rolldrop.py     # p sweep for C3
│   └── plot_curves.py        # robustness curves and seen-vs-unseen table
├── src/
│   ├── corruptions.py        # seedable observation wrappers
│   ├── rolldrop.py           # policy network with rollout-only dropout
│   ├── train.py              # trains any condition from a config
│   └── evaluate.py           # runs a checkpoint under all corruptions
├── tests/
│   ├── test_corruptions.py   # identity at intensity 0, seeds reproducible
│   ├── test_rolldrop.py      # dropout on in rollout, off in update
│   └── test_smoke.py         # tiny train, then eval, end to end
├── results/                  # gitignored
│   └── <run_id>/             # checkpoint, manifest.json, training log, eval CSV
├── .cursor/rules/
├── LICENSE
├── README.md                 # setup, Docker command, smoke-test command
├── requirements.txt          # kept in sync with Docker/requirements.txt
└── .gitignore                # includes results/
```

## What each file does

Already written:

- `configs/clean.yaml`, `gaussian_aug.yaml`, `obs_dropout_aug.yaml`, `rolldrop.yaml`: one config per condition (C0-C3). PPO settings are the same across all four; only `train_corruption` and `rolldrop` differ.
- `configs/eval.yaml`: evaluation settings. Clean, Gaussian, dropout, and uniform corruptions at five intensities each, 20 episodes, seeds 0-2.
- `scripts/run_all.sh`: runs a full stage. `smoke` runs the tests, `pilot` does short training runs followed by eval and plots, and `full` is the real experiment.
- `Docker/Dockerfile`: installs Python, the MuJoCo rendering libraries, and the requirements, then runs `Docker/app.py`.

Still to do:

- `Docker/app.py`: passes the stage name to `scripts/run_all.sh`.
- `src/corruptions.py`: stubs with pseudocode for now. Observation wrapper for `none`, `gaussian`, `dropout`, and `uniform`. Takes the clean observation plus a type, intensity, and seed, and returns the corrupted observation. Gaussian uses intensity as the standard deviation, dropout uses it as the probability of zeroing each value, and uniform adds noise in [-intensity, intensity]. At intensity 0 the observation should come back unchanged.
- `src/rolldrop.py`: C3 policy network. Dropout is active while PPO collects rollouts and turned off during updates.
- `src/train.py`: takes `--config`, with optional `--timesteps` and `--n-steps` for short runs. Builds `Reacher-v5`, applies the training corruption, trains PPO, and saves `model.zip`, `config.yaml`, and `manifest.json` (seed, package versions, git commit) to `results/<run_id>/`.
- `src/evaluate.py`: takes `--eval-config` and `--results-dir`. Runs every trained model under each corruption, intensity, and seed, and writes mean return and success rate to `eval.csv`. An episode counts as a success if the final fingertip-to-target distance is under 0.05. Use `info["reward_dist"]` (the negative distance) for this rather than the observation, since the observation may be corrupted.
- `scripts/plot_curves.py`: reads all the `eval.csv` files and plots return vs. intensity for each corruption type, with one line per condition. Also prints a seen-vs-unseen table per condition.
- `scripts/sweep_rolldrop.py`: trains C3 once for each value in `rolldrop.sweep_p` so we can choose the dropout rate.
- `tests/test_corruptions.py`: checks that intensity 0 is a no-op and that the same seed gives the same noise.
- `tests/test_rolldrop.py`: checks that dropout is on during rollouts and off during updates.
- `tests/test_smoke.py`: trains for a few hundred steps, evaluates, and checks that the output files exist.

## Setup

1. Install Docker, or install Python 3.10+ and run `pip install -r requirements.txt`.
2. From the repo root, set `PYTHONPATH` to the repo root (the Docker image already does this).

## Docker

Build and run from the **repo root** (not `Docker/`):

```text
docker build -f Docker/Dockerfile -t rrl-robot .
docker run --rm -it rrl-robot smoke
```

Swap `smoke` for `pilot` or `full` to run the other stages. Use `docker ps` to check running containers.

## Smoke test

Docker:

```text
docker run --rm -it rrl-robot smoke
```

Local:

```text
python -m pytest tests -q
```

Or `sh scripts/run_all.sh smoke`.
