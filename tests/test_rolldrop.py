import gymnasium
import pytest
import torch

from src.rolldrop import RollDropPolicy, apply_roll_drop
from src.train import load_config, make_model


def test_apply_roll_drop_identity_at_zero():
    x = torch.ones(8, 64)
    out = apply_roll_drop(x, 0.0)
    torch.testing.assert_close(out, x)


def test_apply_roll_drop_changes_values():
    x = torch.ones(1000, 64)
    out = apply_roll_drop(x, 0.5)
    assert not torch.equal(out, x)
    assert torch.any(out == 0)


def test_invalid_dropout_p_raises():
    env = gymnasium.make("Reacher-v5")
    try:
        with pytest.raises(ValueError):
            RollDropPolicy(
                env.observation_space,
                env.action_space,
                lr_schedule=lambda _: 3e-4,
                dropout_p=1.0,
            )
    finally:
        env.close()


def _c3_policy():
    env = gymnasium.make("Reacher-v5")
    model = make_model(env, load_config("configs/rolldrop.yaml"))
    return env, model.policy


def test_dropout_on_in_rollout_forward():
    env, policy = _c3_policy()
    try:
        obs, _ = env.reset(seed=0)
        obs_t, _ = policy.obs_to_tensor(obs)
        action_a, _, _ = policy(obs_t, deterministic=True)
        action_b, _, _ = policy(obs_t, deterministic=True)
        assert not torch.allclose(action_a, action_b)
    finally:
        env.close()


def test_dropout_off_in_update_evaluate_actions():
    env, policy = _c3_policy()
    try:
        obs, _ = env.reset(seed=0)
        obs_t, _ = policy.obs_to_tensor(obs)
        actions = torch.zeros((1, *env.action_space.shape), device=policy.device)
        policy.set_training_mode(True)
        values_a, log_a, _ = policy.evaluate_actions(obs_t, actions)
        values_b, log_b, _ = policy.evaluate_actions(obs_t, actions)
        torch.testing.assert_close(values_a, values_b)
        torch.testing.assert_close(log_a, log_b)
    finally:
        env.close()
