# results/final_results_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def summarize_fedavg_runs(alpha: float, seed_paths):
    final_accs = []
    final_losses = []

    for path in seed_paths:
        df = pd.read_csv(path)
        last = df.iloc[-1]
        final_accs.append(float(last["global_test_acc"]))
        final_losses.append(float(last["global_test_loss"]))

    return {
        "method": "FedAvg",
        "alpha": alpha,
        "final_loss_mean": float(np.mean(final_losses)),
        "final_loss_std": float(np.std(final_losses)),
        "final_acc_mean": float(np.mean(final_accs)),
        "final_acc_std": float(np.std(final_accs))
    }


def summarize_clustered_runs(alpha: float, seed_paths):
    final_accs = []
    final_losses = []

    for path in seed_paths:
        df = pd.read_csv(path)
        last = df.iloc[-1]
        final_accs.append(float(last["global_test_acc"]))
        final_losses.append(float(last["global_test_loss"]))

    return {
        "method": "Clustered FedAvg (delayed)",
        "alpha": alpha,
        "final_loss_mean": float(np.mean(final_losses)),
        "final_loss_std": float(np.std(final_losses)),
        "final_acc_mean": float(np.mean(final_accs)),
        "final_acc_std": float(np.std(final_accs))
    }


def summarize_personalized(alpha: float, path):
    df = pd.read_csv(path)
    # these are per-client metrics; mean across clients
    mean_loss = float(df["test_loss"].mean())
    mean_acc = float(df["test_acc"].mean())
    std_acc = float(df["test_acc"].std())
    worst_acc = float(df["test_acc"].min())

    return {
        "method": "Personalized Clustered FL",
        "alpha": alpha,
        "final_loss_mean": mean_loss,
        "final_loss_std": 0.0,   # not from multiple runs here
        "final_acc_mean": mean_acc,
        "final_acc_std": std_acc,
        "final_acc_worst": worst_acc
    }


def main():
    output_dir = Path("comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # FedAvg: 3 seeds per alpha
    fedavg_alpha_values = [0.1, 0.5, 1.0]
    for alpha in fedavg_alpha_values:
        seed_paths = [
            f"fedavg_alpha_{alpha}_seed0.csv",
            f"fedavg_alpha_{alpha}_seed1.csv",
            f"fedavg_alpha_{alpha}_seed2.csv",
        ]
        all_results.append(summarize_fedavg_runs(alpha, seed_paths))

    # Clustered (delayed clustering): match alpha=1.0 for main comparison
    clustered_seed_paths = [
        "clustered_fedavg/clustered_fedavg_alpha_1.0_seed0.csv",
        "clustered_fedavg/clustered_fedavg_alpha_1.0_seed1.csv",
        "clustered_fedavg/clustered_fedavg_alpha_1.0_seed2.csv",
    ]
    all_results.append(summarize_clustered_runs(1.0, clustered_seed_paths))

    # Personalized FL: single run, but with per-client metrics
    personalized_path = (
        "cluster_personalized/personalized_client_results.csv"
    )
    all_results.append(summarize_personalized(1.0, personalized_path))

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(
        output_dir / "final_results_summary_mean_std.csv",
        index=False
    )

    print("\n--- Final Results Summary (mean ± std) ---")
    print(summary_df[
        ["method", "alpha",
         "final_loss_mean", "final_loss_std",
         "final_acc_mean", "final_acc_std"]
    ])

    # Bar chart: accuracy comparison
    plt.figure(figsize=(10, 5))
    labels = summary_df["method"] + " (α=" + summary_df["alpha"].astype(str) + ")"
    plt.bar(labels, summary_df["final_acc_mean"],
            yerr=summary_df["final_acc_std"],
            capsize=5)
    plt.ylabel("Final Accuracy (mean ± std, %)")
    plt.title("Final Accuracy Comparison Across Methods")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "all_methods_accuracy_bar_mean_std.png")
    plt.show()

    # FedAvg alpha trend (mean accuracy vs alpha)
    fedavg_df = summary_df[summary_df["method"] == "FedAvg"].copy()
    fedavg_df = fedavg_df.sort_values("alpha")

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        fedavg_df["alpha"],
        fedavg_df["final_acc_mean"],
        yerr=fedavg_df["final_acc_std"],
        marker="o",
        capsize=5
    )
    plt.xlabel("Dirichlet Alpha")
    plt.ylabel("Final FedAvg Accuracy (mean ± std, %)")
    plt.title("FedAvg Accuracy vs Alpha")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "fedavg_alpha_trend_mean_std.png")
    plt.show()


if __name__ == "__main__":
    main()