#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT"

stage=${1:-smoke}

smoke() {
    python -m pytest tests -q
}

pilot() {
    python src/train.py --config configs/clean.yaml --timesteps 10000 --n-steps 256
    python src/train.py --config configs/gaussian_aug.yaml --timesteps 10000 --n-steps 256
    python src/train.py --config configs/obs_dropout_aug.yaml --timesteps 10000 --n-steps 256
    python src/train.py --config configs/rolldrop.yaml --timesteps 10000 --n-steps 256
    python src/evaluate.py --eval-config configs/eval.yaml --results-dir results
    python scripts/plot_curves.py --results-dir results
}

full() {
    python src/train.py --config configs/clean.yaml
    python src/train.py --config configs/gaussian_aug.yaml
    python src/train.py --config configs/obs_dropout_aug.yaml
    python src/train.py --config configs/rolldrop.yaml
    python scripts/sweep_rolldrop.py --config configs/rolldrop.yaml
    python src/evaluate.py --eval-config configs/eval.yaml --results-dir results
    python scripts/plot_curves.py --results-dir results
}

case "$stage" in
    smoke) smoke ;;
    pilot) pilot ;;
    full) full ;;
    *)
        echo "usage: $0 {smoke|pilot|full}" >&2
        exit 2
        ;;
esac
