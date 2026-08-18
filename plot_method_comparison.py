# plot_method_comparison.py

import os
import pandas as pd
import matplotlib.pyplot as plt


def load_csv_if_exists(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"Missing file: {path}")
    return None


def plot_comparison(fedavg_path, clustered_path, alpha):
    fedavg_df = load_csv_if_exists(fedavg_path)
    clustered_df = load_csv_if_exists(clustered_path)

    if fedavg_df is None or clustered_df is None:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(fedavg_df["round"], fedavg_df["global_test_acc"], marker="o", label=f"FedAvg alpha={alpha}")
    plt.plot(clustered_df["round"], clustered_df["global_test_acc"], marker="s", label=f"Clustered FedAvg alpha={alpha}")
    plt.xlabel("Communication Round")
    plt.ylabel("Global Test Accuracy (%)")
    plt.title(f"FedAvg vs Clustered FedAvg Accuracy (alpha={alpha})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/comparison_acc_alpha_{alpha}.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(fedavg_df["round"], fedavg_df["global_test_loss"], marker="o", label=f"FedAvg alpha={alpha}")
    plt.plot(clustered_df["round"], clustered_df["global_test_loss"], marker="s", label=f"Clustered FedAvg alpha={alpha}")
    plt.xlabel("Communication Round")
    plt.ylabel("Global Test Loss")
    plt.title(f"FedAvg vs Clustered FedAvg Loss (alpha={alpha})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/comparison_loss_alpha_{alpha}.png")
    plt.show()


def main():
    alpha = "0.1"

    fedavg_path = f"results/fedavg_alpha_{alpha}.csv"
    clustered_path = "results/clustered_fedavg/clustered_fedavg_log.csv"

    plot_comparison(fedavg_path, clustered_path, alpha)


if __name__ == "__main__":
    main()