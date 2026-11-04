import gymnasium
import numpy
import pytest

from src.corruptions import CORRUPTION_TYPES, CorruptObservation, apply_corruption, wrap_env

NOISY_TYPES = ("gaussian", "dropout", "uniform")


def make_obs(size=10):
    return numpy.random.default_rng(123).uniform(-1.0, 1.0, size=size)


def rollout(corruption_type, intensity, seed, steps=10):
    """Observations from a short Reacher-v5 run with fixed env seed and zero actions."""
    env = wrap_env(gymnasium.make("Reacher-v5"), corruption_type, intensity, seed)
    obs, _ = env.reset(seed=0)
    observations = [obs]
    for _ in range(steps):
        obs, *_ = env.step(numpy.zeros(env.action_space.shape))
        observations.append(obs)
    env.close()
    return numpy.array(observations)


@pytest.mark.parametrize("corruption_type", CORRUPTION_TYPES)
def test_intensity_zero_is_identity(corruption_type):
    obs = make_obs()
    out = apply_corruption(obs, corruption_type, 0.0, numpy.random.default_rng(0))
    numpy.testing.assert_array_equal(out, obs)
    assert out is not obs


def test_none_is_identity_at_any_intensity():
    obs = make_obs()
    out = apply_corruption(obs, "none", 0.5, numpy.random.default_rng(0))
    numpy.testing.assert_array_equal(out, obs)


@pytest.mark.parametrize("corruption_type", CORRUPTION_TYPES)
def test_wrap_env_skips_wrapper_at_intensity_zero(corruption_type):
    env = gymnasium.make("Reacher-v5")
    assert wrap_env(env, corruption_type, 0.0, seed=0) is env
    env.close()


@pytest.mark.parametrize("corruption_type", NOISY_TYPES)
def test_wrap_env_adds_wrapper_above_zero(corruption_type):
    env = wrap_env(gymnasium.make("Reacher-v5"), corruption_type, 0.1, seed=0)
    assert isinstance(env, CorruptObservation)
    env.close()


@pytest.mark.parametrize("corruption_type", NOISY_TYPES)
def test_same_seed_same_noise(corruption_type):
    obs = make_obs()
    first = apply_corruption(obs, corruption_type, 0.5, numpy.random.default_rng(7))
    second = apply_corruption(obs, corruption_type, 0.5, numpy.random.default_rng(7))
    other = apply_corruption(obs, corruption_type, 0.5, numpy.random.default_rng(8))
    numpy.testing.assert_array_equal(first, second)
    assert not numpy.array_equal(first, other)


@pytest.mark.parametrize("corruption_type", NOISY_TYPES)
def test_wrapped_env_same_seed_same_observations(corruption_type):
    first = rollout(corruption_type, 0.1, seed=3)
    second = rollout(corruption_type, 0.1, seed=3)
    other = rollout(corruption_type, 0.1, seed=4)
    clean = rollout("none", 0.0, seed=3)
    numpy.testing.assert_array_equal(first, second)
    assert not numpy.array_equal(first, other)
    assert not numpy.array_equal(first, clean)


def test_gaussian_std_matches_intensity():
    obs = numpy.zeros(100_000)
    out = apply_corruption(obs, "gaussian", 0.2, numpy.random.default_rng(0))
    assert abs(out.mean()) < 0.01
    assert abs(out.std() - 0.2) < 0.01


def test_dropout_zeroes_values_at_intensity_rate():
    obs = numpy.ones(100_000)
    out = apply_corruption(obs, "dropout", 0.2, numpy.random.default_rng(0))
    assert set(numpy.unique(out)) <= {0.0, 1.0}
    assert abs((out == 0).mean() - 0.2) < 0.01


def test_uniform_noise_stays_in_range():
    obs = numpy.zeros(100_000)
    out = apply_corruption(obs, "uniform", 0.1, numpy.random.default_rng(0))
    assert numpy.all(numpy.abs(out) <= 0.1)
    assert out.min() < -0.09 and out.max() > 0.09


def test_unknown_type_raises():
    with pytest.raises(ValueError):
        apply_corruption(make_obs(), "salt_and_pepper", 0.1, numpy.random.default_rng(0))


def test_negative_intensity_raises():
    with pytest.raises(ValueError):
        apply_corruption(make_obs(), "gaussian", -0.1, numpy.random.default_rng(0))
