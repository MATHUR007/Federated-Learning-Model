# compare_methods.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def read_fedavg(path):
    df = pd.read_csv(path)
    last = df.iloc[-1]
    return {
        "method": f"FedAvg ({Path(path).stem.replace('fedavg_', 'alpha=')})",
        "final_loss": float(last["global_test_loss"]),
        "final_acc": float(last["global_test_acc"]),
        "source_file": str(path)
    }

def read_clustered(path):
    df = pd.read_csv(path)
    last = df.iloc[-1]
    return {
        "method": "Clustered FedAvg",
        "final_loss": float(last["global_test_loss"]),
        "final_acc": float(last["global_test_acc"]),
        "source_file": str(path)
    }

def read_personalized(path):
    df = pd.read_csv(path)
    return {
        "method": "Personalized Clustered FL",
        "final_loss": float(df["test_loss"].mean()),
        "final_acc": float(df["test_acc"].mean()),
        "source_file": str(path)
    }

def main():
    results = []

    fedavg_files = [
        "results/fedavg_alpha_0.1.csv",
        "results/fedavg_alpha_0.5.csv",
        "results/fedavg_alpha_1.0.csv"
    ]

    clustered_file = "results/clustered_fedavg/clustered_fedavg_log.csv"
    personalized_file = "results/cluster_personalized/personalized_client_results.csv"

    for f in fedavg_files:
        try:
            results.append(read_fedavg(f))
        except Exception as e:
            print(f"Missing or unreadable: {f} -> {e}")

    try:
        results.append(read_clustered(clustered_file))
    except Exception as e:
        print(f"Missing or unreadable: {clustered_file} -> {e}")

    try:
        results.append(read_personalized(personalized_file))
    except Exception as e:
        print(f"Missing or unreadable: {personalized_file} -> {e}")

    if not results:
        print("No valid result files found.")
        return

    summary_df = pd.DataFrame(results)
    print("\n--- Method Comparison Summary ---")
    print(summary_df[["method", "final_loss", "final_acc"]])

    out_dir = Path("results/comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "method_comparison_summary.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.bar(summary_df["method"], summary_df["final_acc"])
    plt.ylabel("Accuracy (%)")
    plt.title("Final Accuracy Comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "final_accuracy_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()