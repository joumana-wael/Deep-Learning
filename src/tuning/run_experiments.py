from src.tuning.search import (
    generate_grid_search_configs,
    generate_random_search_configs,
)
from src.tuning.runner import run_tuning


def main():

    # ==========================================
    # Search space
    # ==========================================
    search_space = {
        "learning_rate": [0.001, 0.0005],
        "batch_size": [8, 16],
    }

    epochs = 5
    seed = 42

    # ==========================================
    # Grid Search
    # ==========================================
    print("\n" + "=" * 60)
    print("RUNNING GRID SEARCH")
    print("=" * 60)

    grid_configs = generate_grid_search_configs(
        search_space
    )

    grid_results, grid_best = run_tuning(
        configs=grid_configs,
        epochs=epochs,
        output_dir="outputs/grid_search",
        seed=seed,
    )

    # ==========================================
    # Random Search
    # ==========================================
    print("\n" + "=" * 60)
    print("RUNNING RANDOM SEARCH")
    print("=" * 60)

    random_configs = generate_random_search_configs(
        search_space=search_space,
        n_trials=3,
        seed=seed,
    )

    random_results, random_best = run_tuning(
        configs=random_configs,
        epochs=epochs,
        output_dir="outputs/random_search",
        seed=seed,
    )

    # ==========================================
    # Final Comparison
    # ==========================================
    print("\n" + "=" * 60)
    print("FINAL SEARCH COMPARISON")
    print("=" * 60)

    print("\nGrid Search")
    print("Best configuration:", {
        "learning_rate": grid_best["learning_rate"],
        "batch_size": grid_best["batch_size"],
    })
    print(
        f"Best validation accuracy: "
        f"{grid_best['best_val_accuracy']:.2f}%"
    )

    print("\nRandom Search")
    print("Best configuration:", {
        "learning_rate": random_best["learning_rate"],
        "batch_size": random_best["batch_size"],
    })
    print(
        f"Best validation accuracy: "
        f"{random_best['best_val_accuracy']:.2f}%"
    )

    # Overall winner
    if (
        grid_best["best_val_accuracy"]
        >= random_best["best_val_accuracy"]
    ):
        winner = "Grid Search"
        best_result = grid_best
    else:
        winner = "Random Search"
        best_result = random_best

    print("\n" + "-" * 60)
    print("OVERALL WINNER:", winner)
    print(
        "Best configuration:",
        {
            "learning_rate": best_result["learning_rate"],
            "batch_size": best_result["batch_size"],
        },
    )
    print(
        f"Best validation accuracy: "
        f"{best_result['best_val_accuracy']:.2f}%"
    )


if __name__ == "__main__":
    main()