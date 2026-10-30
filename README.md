# USC-NLP566-RRL-Robot-Control-

Robust RL for robot control under observation corruptions. Training conditions are clean (C0), Gaussian augmentation (C1), observation-dropout augmentation (C2), and Roll-Drop (C3). See `docs/SPEC.md`.

## Overview

Real robots have imperfect sensors, but most robot control policies are trained in simulation on perfect readings. We test whether training with imperfect readings makes a policy hold up better, including against kinds of sensor error it never saw.

The robot is a simulated two-joint arm that learns to touch a target. It runs in the MuJoCo physics simulator, which installs with the Python requirements.

There is no dataset to collect. The arm learns by practice, which is how reinforcement learning works. At each step, the simulator reports where the arm and target are, and the policy decides how hard to push each joint. The simulator moves the arm and scores the move: the closer the fingertip gets to the target, the higher the score. Over many practice runs, the policy learns to make moves that score well. Those practice steps are the training data.

We train the arm four ways:

- C0, clean: perfect sensor readings.
- C1, Gaussian noise: small random errors added to every reading.
- C2, dropout: some readings randomly blanked out.
- C3, Roll-Drop: perfect readings, but the policy itself randomly ignores some of its inputs while it practices.

Each condition trains its own neural network from scratch, so there are four separate networks. A network never sees another condition's data, and it is never told which condition it is in. It only sees the readings it gets and learns from the score. The simulator is the same for all four. The condition only changes what happens to the readings between the simulator and the network. C3 also gets a few extra training runs to pick its dropout rate (`scripts/sweep_rolldrop.py`).

Then we test all four with noisy readings at increasing strength, including uniform noise, a type that none of them trained on. Comparing how often each one still reaches the target shows which training method holds up best, and whether that holds for noise it has never seen.

## What the project does

We train a robot arm policy four different ways, then test each one with noisy sensors. The question is whether robustness learned from one kind of noise carries over to a kind the policy never saw in training.

- Robot: Gymnasium MuJoCo `Reacher-v5`. A two-joint arm moves its fingertip to a random target. Each episode is 50 steps.
- Policy input: 10 numbers per step. These are the joint angles (as cos and sin), target x and y, joint velocities, and fingertip-minus-target x and y.
- Policy output: 2 joint torques per step.
- What we corrupt: only those 10 input numbers. Actions and physics stay clean.

Input to output:

```text
configs/<condition>.yaml  --train.py-->        results/<run_id>/model.zip, config.yaml, manifest.json
model.zip + eval.yaml     --evaluate.py-->     results/<run_id>/eval.csv
every eval.csv            --plot_curves.py-->  robustness curves and a seen-vs-unseen table
```

The final result is one comparison. On uniform noise, which no condition trains on, does C1, C2, or C3 beat C0 without losing much on clean observations?

## Build order

Every `.py` file is empty right now, so Docker and the smoke test do not work yet. Build in this order. Each step only needs the steps before it.

1. `Docker/app.py`. Read the stage name (`smoke`, `pilot`, or `full`) and run `sh scripts/run_all.sh <stage>`. After this, `docker run` starts the pipeline.
2. `src/corruptions.py` and `tests/test_corruptions.py`. Write the four observation wrappers and test them.
3. `src/train.py` for C0 only. Get one clean PPO run to train and save into `results/`.
4. `src/evaluate.py`. Load that checkpoint and write `eval.csv` for every corruption and intensity.
5. `tests/test_smoke.py`. A tiny train, then eval, end to end. After this, `smoke` passes.
6. C1 and C2. No new files. `train.py` wraps the environment with `train_corruption` from the YAML.
7. `src/rolldrop.py` and `tests/test_rolldrop.py`. Then have `train.py` use it when `rolldrop.dropout_p > 0` (C3).
8. `scripts/plot_curves.py`. Turn the `eval.csv` files into curves and the table.
9. Run `pilot` and fix what breaks. The YAML files freeze on Nov 1, so tune settings before then.
10. `scripts/sweep_rolldrop.py`, then run `full`.

Steps 1-5 are Phase 1 in the proposal. Steps 6-9 are Phase 2. Step 10 is Phase 3.

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

- `configs/clean.yaml`, `gaussian_aug.yaml`, `obs_dropout_aug.yaml`, `rolldrop.yaml`: one training condition each (C0-C3). PPO settings are identical. Only `train_corruption` and `rolldrop` differ.
- `configs/eval.yaml`: the test plan. It lists clean, Gaussian, dropout, and uniform corruptions at five intensities, 20 episodes, and seeds 0-2.
- `scripts/run_all.sh`: runs a whole stage. `smoke` runs the tests. `pilot` trains short runs, then evaluates and plots. `full` is the real experiment.
- `Docker/Dockerfile`: installs Python, MuJoCo rendering libraries, and the requirements, then calls `Docker/app.py`.

To write (currently empty):

- `Docker/app.py`: takes the stage name and runs `scripts/run_all.sh` with it. Nothing else.
- `src/corruptions.py`: Gymnasium observation wrappers for `none`, `gaussian`, `dropout`, and `uniform`. Input is the clean observation, the type, an intensity, and a seed. Output is the corrupted observation. A simple choice: Gaussian adds noise with standard deviation equal to the intensity, dropout zeroes each value with probability equal to the intensity, and uniform adds noise between minus and plus the intensity. Intensity 0 must return the observation unchanged.
- `src/rolldrop.py`: the C3 policy network. Dropout is on while PPO collects rollouts and off while it computes updates. Input is an observation. Output is an action.
- `src/train.py`: takes `--config`, plus optional `--timesteps` and `--n-steps` for short runs. It builds `Reacher-v5`, adds the training corruption, trains PPO, and writes `model.zip`, `config.yaml`, and `manifest.json` (seed, package versions, git commit) to `results/<run_id>/`.
- `src/evaluate.py`: takes `--eval-config` and `--results-dir`. For every run, corruption, intensity, and seed, it plays the episodes and writes mean return and success rate to `eval.csv`. Success means the final fingertip-to-target distance is under 0.05. Read that distance from `info["reward_dist"]` (it is minus the distance), not from the observation, or the noise will also corrupt the score.
- `scripts/plot_curves.py`: reads every `eval.csv`. It draws one plot per corruption type, with return against intensity and one line per condition. It also prints a table of seen versus unseen corruptions for each condition.
- `scripts/sweep_rolldrop.py`: trains C3 once for each value in `rolldrop.sweep_p`, so we can pick the dropout rate.
- `tests/test_corruptions.py`: intensity 0 changes nothing, and the same seed gives the same noise.
- `tests/test_rolldrop.py`: dropout is active during rollouts and inactive during updates.
- `tests/test_smoke.py`: trains for a few hundred steps, evaluates, and checks the output files exist.

## Setup

1. Install Docker, or install Python 3.10+ and `pip install -r requirements.txt`.
2. From the repo root, set `PYTHONPATH` to that root (the Docker image does this for you).

## Docker

Build and run from the **repo root** (not `Docker/`):

```text
docker build -f Docker/Dockerfile -t rrl-robot .
docker run --rm -it rrl-robot smoke
```

Replace `smoke` with `pilot` or `full`. Check a running container with `docker ps`.

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
