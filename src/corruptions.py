"""Observation corruptions for Reacher-v5.

This file simulates sensor noise. A wrapper sits between the simulator and the
policy and corrupts the 10 observation values before the policy receives them.
Actions and physics stay clean.

    simulator --clean obs--> CorruptObservation --corrupted obs--> policy

Corruption types (intensity 0 always means "no change"):

- none: return the observation unchanged.
- gaussian: add normal noise with standard deviation = intensity.
- dropout: zero each number with probability = intensity.
- uniform: add noise drawn evenly from [-intensity, +intensity].
  Uniform is held out: no training condition uses it.

Example: one observation (10 values, positions in meters) before and after
each corruption. Noise values are illustrative; a fresh draw happens every step.

    #  meaning                     clean   gaussian 0.1  dropout 0.2  uniform 0.1
    0  cos(joint 1 angle)          0.995   1.065         0.995        0.935
    1  cos(joint 2 angle)          0.999   0.879         0.999        1.089
    2  sin(joint 1 angle)          0.095   0.125         0.095        0.075
    3  sin(joint 2 angle)         -0.042   0.108        -0.042       -0.002
    4  target x                    0.120   0.070         0            0.200
    5  target y                   -0.080  -0.060        -0.080       -0.180
    6  joint 1 speed               0.003  -0.087         0.003        0.013
    7  joint 2 speed              -0.002   0.108        -0.002       -0.072
    8  fingertip - target, x       0.090   0.050         0            0.140
    9  fingertip - target, y       0.091   0.151         0.091        0.061

The same intensity is a different relative error per value: about 10% on the
cos values (near 1.0), but about 100% on positions (near 0.1) and more on speeds.

When it is called:

- Training (src/train.py): wrap the env with the YAML's train_corruption.
  C1 uses gaussian 0.1, C2 uses dropout 0.2. C0 and C3 use none.
- Tuning (scripts/sweep_rolldrop.py): C3 still trains with none. To pick the
  best dropout_p, compare runs on clean or seen corruptions only. Never pick
  it using uniform results, or uniform stops being unseen.
- Testing (src/evaluate.py): every trained model is wrapped once per
  corruption type, intensity, and seed listed in configs/eval.yaml.

Usage:

    env = gymnasium.make("Reacher-v5")
    env = wrap_env(env, "gaussian", 0.1, seed=0)
    obs, info = env.reset()  # obs is already corrupted

Score success from info["reward_dist"] (minus the true distance), never from
the corrupted observation.
"""

import gymnasium

CORRUPTION_TYPES = ("none", "gaussian", "dropout", "uniform")


def apply_corruption(obs, corruption_type, intensity, rng):
    """Return a corrupted copy of one observation.

    Input: obs (numpy array, shape (10,)), corruption_type (one of
    CORRUPTION_TYPES), intensity (float >= 0), rng (numpy Generator).
    Output: numpy array with the same shape as obs.

    Pseudocode:

        if corruption_type not in CORRUPTION_TYPES: raise ValueError
        if corruption_type == "none" or intensity == 0: return obs.copy()

        if corruption_type == "gaussian":
            return obs + rng.normal(0, intensity, size=obs.shape)

        if corruption_type == "dropout":
            keep = rng.random(obs.shape) >= intensity  # True means keep
            return obs * keep

        if corruption_type == "uniform":
            return obs + rng.uniform(-intensity, intensity, size=obs.shape)
    """
    return None


class CorruptObservation(gymnasium.ObservationWrapper):
    """Gymnasium wrapper that corrupts every observation the env returns.

    Gymnasium calls observation() automatically after every reset() and
    step(), so callers never call it themselves.
    """

    def __init__(self, env, corruption_type, intensity, seed):
        """Store the settings and create a seeded random generator.

        Pseudocode:

            super().__init__(env)
            self.corruption_type = corruption_type
            self.intensity = intensity
            self.rng = numpy.random.default_rng(seed)  # same seed, same noise
        """
        return None

    def observation(self, obs):
        """Return the corrupted observation.

        Pseudocode:

            return apply_corruption(obs, self.corruption_type, self.intensity, self.rng)
        """
        return None


def wrap_env(env, corruption_type, intensity, seed):
    """Wrap env so its observations are corrupted. train.py and evaluate.py call this.

    Pseudocode:

        if corruption_type == "none" or intensity == 0: return env
        return CorruptObservation(env, corruption_type, intensity, seed)
    """
    return None
