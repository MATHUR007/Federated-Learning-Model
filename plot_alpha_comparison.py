# plot_alpha_comparison.py

import pandas as pd
import matplotlib.pyplot as plt
import os


def load_results(file_paths):
    results = {}
    for label, path in file_paths.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            results[label] = df
        else:
            print(f"Warning: file not found -> {path}")
    return results


def plot_accuracy(results):
    plt.figure(figsize=(10, 6))

    for label, df in results.items():
        plt.plot(df["round"], df["global_test_acc"], marker='o', label=label)

    plt.xlabel("Communication Round")
    plt.ylabel("Global Test Accuracy (%)")
    plt.title("FedAvg Accuracy Comparison for Different Dirichlet Alpha Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/alpha_accuracy_comparison.png")
    plt.show()


def plot_loss(results):
    plt.figure(figsize=(10, 6))

    for label, df in results.items():
        plt.plot(df["round"], df["global_test_loss"], marker='o', label=label)

    plt.xlabel("Communication Round")
    plt.ylabel("Global Test Loss")
    plt.title("FedAvg Loss Comparison for Different Dirichlet Alpha Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/alpha_loss_comparison.png")
    plt.show()


def main():
    file_paths = {
        "alpha=1.0": "results/fedavg_alpha_1.0.csv",
        "alpha=0.5": "results/fedavg_alpha_0.5.csv",
        "alpha=0.1": "results/fedavg_alpha_0.1.csv"
    }

    results = load_results(file_paths)

    if not results:
        print("No valid CSV files found in results/.")
        return

    plot_accuracy(results)
    plot_loss(results)


if __name__ == "__main__":
    main()