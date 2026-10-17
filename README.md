# USC-NLP566-RRL-Robot-Control-

Robust RL for robot control under observation corruptions. Training conditions are clean (C0), Gaussian augmentation (C1), observation-dropout augmentation (C2), and Roll-Drop (C3). See `docs/SPEC.md`.

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
