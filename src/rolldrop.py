import yaml
import numpy as np


class RollDrop:
    def __init__(self, config_path=None):
        self.dropout_p = 0.1
        self.sweep_p = []

        if config_path is not None:
            self.load_config(config_path)

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.dropout_p = config["rolldrop"]["dropout_p"]
        self.sweep_p = config["rolldrop"]["sweep_p"]

        self.validate_probability()

    def set_dropout_probability(self, p):
        self.dropout_p = p
        self.validate_probability()

    def get_dropout_probability(self):
        return self.dropout_p

    def validate_probability(self):
        if not 0.0 <= self.dropout_p < 1.0:
            raise ValueError(
                f"dropout_p must be in [0, 1), got {self.dropout_p}"
            )

    def generate_mask(self, batch_size):
        mask = np.random.rand(batch_size) > self.dropout_p

        if batch_size > 0 and not mask.any():
            mask[np.random.randint(batch_size)] = True

        return mask

    def compute_keep_indices(self, n_samples):
        return np.where(self.generate_mask(n_samples))[0]

    def apply_to_batch(self, batch):
        mask = self.generate_mask(
            len(batch["observations"])
        )

        return {
            key: value[mask]
            for key, value in batch.items()
        }

    def drop_trajectories(self, trajectories):
        mask = self.generate_mask(
            len(trajectories)
        )

        return [
            traj
            for traj, keep in zip(trajectories, mask)
            if keep
        ]

    def drop_transitions(self, transitions):
        mask = self.generate_mask(
            len(transitions)
        )

        return [
            transition
            for transition, keep in zip(transitions, mask)
            if keep
        ]