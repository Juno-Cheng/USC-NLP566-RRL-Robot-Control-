import json
from pathlib import Path
from unittest.mock import MagicMock

import gymnasium
import pytest
import yaml

from src.corruptions import CorruptObservation
from src.train import (
    load_config,
    main,
    make_env,
    make_model,
    make_run_id,
    parse_args,
    train,
    write_outputs,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "C0": ROOT / "configs" / "clean.yaml",
    "C1": ROOT / "configs" / "gaussian_aug.yaml",
    "C2": ROOT / "configs" / "obs_dropout_aug.yaml",
    "C3": ROOT / "configs" / "rolldrop.yaml",
}


def clean_config():
    return yaml.safe_load(CONFIGS["C0"].read_text())


def test_parse_args_requires_config():
    with pytest.raises(SystemExit):
        parse_args([])


def test_parse_args_reads_config_path():
    args = parse_args(["--config", "configs/clean.yaml"])
    assert args.config == "configs/clean.yaml"
    assert args.timesteps is None
    assert args.n_steps is None


def test_parse_args_reads_overrides():
    args = parse_args(
        ["--config", "configs/clean.yaml", "--timesteps", "10000", "--n-steps", "256"]
    )
    assert args.timesteps == 10000
    assert args.n_steps == 256


@pytest.mark.parametrize("condition,path", CONFIGS.items())
def test_load_config_reads_yaml(condition, path):
    config = load_config(path)
    assert config["condition"] == condition
    assert config["env_id"] == "Reacher-v5"
    assert config["algo"] == "ppo"
    assert "train_corruption" in config
    assert "rolldrop" in config


def test_make_run_id_uses_condition_name_and_seed():
    assert make_run_id(clean_config()) == "C0_clean_seed0"


def test_make_env_c0_is_not_wrapped():
    env = make_env(clean_config())
    try:
        assert not isinstance(env, CorruptObservation)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (10,)
    finally:
        env.close()


def test_make_env_c1_wraps_gaussian():
    env = make_env(load_config(CONFIGS["C1"]))
    try:
        assert isinstance(env, CorruptObservation)
        assert env.corruption_type == "gaussian"
        assert env.intensity == 0.1
    finally:
        env.close()


def test_make_env_c2_wraps_dropout():
    env = make_env(load_config(CONFIGS["C2"]))
    try:
        assert isinstance(env, CorruptObservation)
        assert env.corruption_type == "dropout"
        assert env.intensity == 0.2
    finally:
        env.close()


def test_make_model_c0_is_ppo():
    from stable_baselines3 import PPO

    env = gymnasium.make("Reacher-v5")
    try:
        model = make_model(env, clean_config())
        assert isinstance(model, PPO)
        assert model.n_steps == 2048
        assert model.batch_size == 64
    finally:
        env.close()


def test_make_model_c3_uses_rolldrop_policy():
    from stable_baselines3 import PPO

    from src.rolldrop import RollDropPolicy

    env = gymnasium.make("Reacher-v5")
    try:
        model = make_model(env, load_config(CONFIGS["C3"]))
        assert isinstance(model, PPO)
        assert isinstance(model.policy, RollDropPolicy)
        assert model.policy.dropout_p == 0.1
    finally:
        env.close()


def test_write_outputs_saves_model_config_and_manifest(tmp_path):
    run_dir = tmp_path / "C0_clean_seed0"
    config = clean_config()
    model = MagicMock()

    def save(path):
        Path(path).write_bytes(b"zip")

    model.save.side_effect = save
    write_outputs(run_dir, model, config)

    assert (run_dir / "model.zip").is_file()
    saved_config = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert saved_config["condition"] == "C0"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["condition"] == "C0"
    assert manifest["name"] == "clean"
    assert manifest["seed"] == 0
    assert manifest["run_id"] == "C0_clean_seed0"


def test_train_applies_overrides_and_learns(tmp_path, monkeypatch):
    config = clean_config()
    fake_env = MagicMock()
    fake_model = MagicMock()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.train.make_env", lambda cfg: fake_env)
    monkeypatch.setattr("src.train.make_model", lambda env, cfg: fake_model)
    monkeypatch.setattr("src.train.make_run_id", lambda cfg: "C0_clean_seed0")
    monkeypatch.setattr("src.train.write_outputs", lambda run_dir, model, cfg: None)

    run_dir = train(config, timesteps=128, n_steps=64)

    assert config["total_timesteps"] == 128
    assert config["n_steps"] == 64
    fake_model.learn.assert_called_once_with(total_timesteps=128)
    fake_env.close.assert_called_once()
    assert Path(run_dir) == tmp_path / "results" / "C0_clean_seed0"


def test_main_loads_config_and_calls_train(monkeypatch):
    called = {}

    def fake_train(config, timesteps=None, n_steps=None):
        called["config"] = config
        called["timesteps"] = timesteps
        called["n_steps"] = n_steps

    monkeypatch.setattr("src.train.train", fake_train)
    main(
        [
            "--config",
            str(CONFIGS["C0"]),
            "--timesteps",
            "10000",
            "--n-steps",
            "256",
        ]
    )
    assert called["config"]["condition"] == "C0"
    assert called["timesteps"] == 10000
    assert called["n_steps"] == 256
