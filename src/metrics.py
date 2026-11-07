
import numpy as np


def compute_metrics(episode_returns, success_flags):
    """Compute return statistics and success rate."""
    returns = np.asarray(episode_returns, dtype=float)
    successes = np.asarray(success_flags)

    if returns.ndim != 1 or successes.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if len(returns) == 0 or len(returns) != len(successes):
        raise ValueError("Inputs must be nonempty and equal in length.")

    if not np.all(np.isfinite(returns)):
        raise ValueError("Episode returns must be finite.")

    if not np.all(np.isin(successes, [0, 1, False, True])):
        raise ValueError("Success flags must be binary.")

    return {
        "average_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "success_rate": float(np.mean(successes)),
    }


def compute_success(final_distance, threshold=0.05):
    """Return whether the final target distance meets the criterion."""
    if not np.isfinite(final_distance):
        raise ValueError("Distance must be finite.")
    if threshold <= 0 or not np.isfinite(threshold):
        raise ValueError("Threshold must be positive and finite.")

    return bool(final_distance < threshold)


def compute_degradation(clean_return, corrupted_return):
    """Compare corrupted mean return against clean mean return."""
    if not np.all(np.isfinite([clean_return, corrupted_return])):
        raise ValueError("Returns must be finite.")

    absolute_drop = float(clean_return - corrupted_return)

    relative_drop = (
        float(absolute_drop / abs(clean_return))
        if abs(clean_return) > 1e-12
        else None
    )

    return {
        "absolute_drop": absolute_drop,
        "relative_drop": relative_drop,
    }


def aggregate_seeds(seed_mean_returns, seed_success_rates):
    """Aggregate metrics across independent evaluation seeds."""
    returns = np.asarray(seed_mean_returns, dtype=float)
    successes = np.asarray(seed_success_rates, dtype=float)

    if returns.ndim != 1 or successes.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if len(returns) == 0 or len(returns) != len(successes):
        raise ValueError("Inputs must be nonempty and equal in length.")

    if not np.all(np.isfinite(returns)):
        raise ValueError("Seed returns must be finite.")

    if not np.all(np.isfinite(successes)):
        raise ValueError("Seed success rates must be finite.")

    if np.any((successes < 0) | (successes > 1)):
        raise ValueError("Success rates must be between 0 and 1.")

    return {
        "mean_return": float(np.mean(returns)),
        "std_across_seeds": float(np.std(returns)),
        "mean_success_rate": float(np.mean(successes)),
    }
