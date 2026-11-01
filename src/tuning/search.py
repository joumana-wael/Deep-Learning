import itertools
import random


def generate_grid_search_configs(search_space):
    """Generate every possible hyperparameter combination."""

    keys = list(search_space.keys())
    values = list(search_space.values())

    combinations = itertools.product(*values)

    return [
        dict(zip(keys, combination))
        for combination in combinations
    ]


def generate_random_search_configs(
    search_space,
    n_trials,
    seed=42,
):
    """Generate unique randomly selected hyperparameter configurations."""

    all_configs = generate_grid_search_configs(search_space)

    if n_trials <= 0:
        raise ValueError("n_trials must be greater than zero.")

    if n_trials > len(all_configs):
        raise ValueError(
            "n_trials cannot be greater than the number "
            "of possible configurations."
        )

    rng = random.Random(seed)

    return rng.sample(all_configs, n_trials)