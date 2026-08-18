# Federated Learning on CIFAR-10: Comparative Analysis

This repository contains the source code and experimental framework for an MSc Dissertation evaluating non-IID data partitioning strategies in Federated Learning (FL). The project implements and compares three distinct FL paradigms on the CIFAR-10 dataset across multiple Dirichlet distribution parameters (α) and random seeds.

---

## Project Overview

This framework benchmarks three federated learning architectures under varying degrees of client data heterogeneity:
1. **FedAvg Baseline:** Standard Federated Averaging evaluated across multiple Dirichlet alpha (α) values.
2. **Clustered FedAvg:** A clustered FL pipeline featuring a delayed clustering warm-up phase to improve initial global feature representation.
3. **Personalized Clustered FL:** An architecture focusing on per-cluster fine-tuning, evaluated via granular client-centric metrics.

---

##  Repository Structure

```text
├── clustering/                     # Clustering algorithmic modules
│   ├── client_similarity.py        # Computes pairwise client similarity matrices
│   ├── kmeans_cluster.py           # K-Means clustering implementation
│   └── group_clients.py            # Logic for partitioning clients into clusters
├── data/
│   └── load_data.py                # Pipeline for fetching and caching CIFAR-10
├── datasets/
│   ├── partition.py                # Dirichlet non-IID data splitter
│   └── client_dataset.py           # PyTorch Dataloader builders for individual clients
├── models/
│   └── cnn.py                      # Target Convolutional Neural Network (CNN) architecture
├── server/
│   └── fedavg.py                   # Aggregation logic (weighted parameter averaging)
├── train/
│   └── local_train.py              # Local optimization and training loops per client
├── results/                        # Generated output data directories
│   ├── clustered_fedavg/           # Outputs from Clustered FL runs
│   └── cluster_personalized/        # Outputs from Personalized FL evaluations
├── comparison/                     # Final aggregated summaries and plots
├── main_fedavg.py                  # Orchestrates the baseline FedAvg experiments
├── main_clustered_fedavg.py        # Orchestrates the Clustered FedAvg experiments
├── evaluate_personalized.py        # Evaluates local personalized models
└── final_results_analysis.py       # Aggregates results, computes stats, and plots figures
```

---

## Experimental Configurations

To ensure reproducibility, hyperparameters such as network architecture, batch size, communication rounds, and local epochs are strictly aligned across all experimental setups.

### Hyperparameter Suite

| Hyperparameter | FedAvg Baseline | Clustered FedAvg | Personalized Clustered FL |
| :--- | :---: | :---: | :---: |
| **Total Clients ($N$)** | 20 | 20 | 20 |
| **Clients per Round ($K$)** | 10 | 10 | — |
| **Communication Rounds** | 20 | 20 | — |
| **Local Epochs ($E$)** | 1 | 1 | — |
| **Batch Size ($B$)** | 64 | 64 | 64 |
| **Random Seeds** | 0, 1, 2 | 0, 1, 2 | — |
| **Dirichlet Alpha ($\alpha$)** | 0.1, 0.5, 1.0 | 1.0 | 0.5 or 1.0 (via artifacts) |
| **Warm-up Rounds** | — | 5 | — |
| **Number of Clusters ($k$)** | — | 2 | Per configuration |

---

## Execution Order

Execute the following pipeline sequentially from the project root directory to reproduce the experimental results:

```bash
# 1. Evaluate baseline FedAvg across all alpha values and seeds
python main_fedavg.py

# 2. Evaluate Clustered FedAvg with a 5-round warm-up phase
python main_clustered_fedavg.py

# 3. Process personalization tracking across client subsets
python evaluate_personalized.py

# 4. Aggregate data, calculate mean ± std dev, and generate figures
python final_results_analysis.py
```

---

##  Output Artifacts & Data Tracking

The execution pipeline generates the following structural outputs, which are omitted from version control but required for analysis and appendix inclusion:

### Raw Statistical Outputs (`.csv`)
* **Baseline FedAvg:** `results/fedavg_alpha_{0.1|0.5|1.0}_seed{0|1|2}.csv`
* **Clustered FL:** `results/clustered_fedavg/clustered_fedavg_alpha_1.0_seed{0|1|2}.csv`
* **Personalized FL:** `results/cluster_personalized/personalized_client_results.csv`
* **Aggregated Summary:** `comparison/final_results_summary_mean_std.csv`

### Visual Analytics (`.png`)
* `comparison/all_methods_accuracy_bar_mean_std.png` — Cross-paradigm structural performance comparison.
* `comparison/fedavg_alpha_trend_mean_std.png` — Analysis of standard FedAvg tracking across varying data skewness.

---

## Dissertation Narrative & Interpretation Notes

When synthesizing these results within the dissertation text or preparing for the viva presentation, account for the following analytical insights:

1. **Dirichlet Skew Factor:** Global performance in FedAvg scales positively with $\alpha$. Lower alpha values introduce severe non-IID conditions, degrading model convergence.
2. **Delayed Clustering Dynamics:** The delayed clustering layout yields metrics near standard FedAvg. This behavior should be framed carefully as a test of the hypothesis that *premature clustering before feature convergence limits optimal routing*, rather than a definitive baseline conclusion without further structural ablations.
3. **Personalization Spread vs. Accuracy:** Personalized clustered configurations may reflect lower absolute mean accuracy globally, yet showcase highly variable client distributions (granular spread). Highlight the mean, worst-performing, and standard deviation bounds of client accuracy to illustrate fairness versus overall optimization trade-offs.

---

## Notes for Reproducible Research
* If updating hyperparameter criteria or changing target code frameworks, delete downstream local `.csv` files and re-run the entire core execution pipeline to prevent mixed-variable analysis.
* Back up generated artifacts in local storage; these metrics serve as verifiable proof of experiment authenticity for your dissertation appendix.
