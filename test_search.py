from src.tuning.search import (
    generate_grid_search_configs,
    generate_random_search_configs,
)


def main():

    search_space = {
        "learning_rate": [0.001, 0.0005],
        "batch_size": [8, 16],
    }

    print("=" * 50)
    print("GRID SEARCH")
    print("=" * 50)

    grid_configs = generate_grid_search_configs(
        search_space
    )

    for index, config in enumerate(
        grid_configs,
        start=1,
    ):
        print(f"Trial {index}: {config}")

    assert len(grid_configs) == 4

    print("\nGrid search test passed! ✅")

    print("\n" + "=" * 50)
    print("RANDOM SEARCH")
    print("=" * 50)

    random_configs = generate_random_search_configs(
        search_space=search_space,
        n_trials=3,
        seed=42,
    )
    assert len(
    {
        tuple(sorted(config.items()))
        for config in random_configs
    }
) == 3

    for index, config in enumerate(
        random_configs,
        start=1,
    ):
        print(f"Trial {index}: {config}")

    assert len(random_configs) == 3

    print("\nRandom search test passed! ✅")


if __name__ == "__main__":
    main()