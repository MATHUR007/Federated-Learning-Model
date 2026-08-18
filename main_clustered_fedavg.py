# main_clustered_fedavg.py

import os
import csv
import random
import torch
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms

from data.load_data import get_cifar10_loaders
from models.cnn import SimpleCNN
from datasets.partition import dirichlet_partition
from datasets.client_dataset import get_client_loader
from train.local_train import train_local
from server.fedavg import weighted_average_weights
from clustering.client_similarity import build_similarity_matrix
from clustering.kmeans_cluster import cluster_clients
from clustering.group_clients import group_clients_by_cluster


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100.0 * correct / total


def average_cluster_models(cluster_models, cluster_sizes):
    total_size = sum(cluster_sizes)
    global_state = {}

    for key in cluster_models[0].keys():
        global_state[key] = sum(
            cluster_models[i][key] * (cluster_sizes[i] / total_size)
            for i in range(len(cluster_models))
        )

    return global_state


def run_clustered_experiment(alpha: float, seed: int, device: torch.device):
    print(f"\n=== Clustered FL run | alpha={alpha} | seed={seed} ===")
    set_seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    _, test_loader = get_cifar10_loaders(batch_size=64)

    num_clients = 20
    clients_per_round = 10
    total_rounds = 20     # match FedAvg
    local_epochs = 1
    batch_size = 64
    num_clusters = 2

    warmup_rounds = 5     # delayed clustering: FedAvg for first 5 rounds

    client_dict = dirichlet_partition(
        train_dataset,
        num_clients=num_clients,
        alpha=alpha
    )
    global_model = SimpleCNN().to(device)

    os.makedirs("results/clustered_fedavg", exist_ok=True)

    log_file = (
        f"results/clustered_fedavg/clustered_fedavg_alpha_{alpha}_seed{seed}.csv"
    )
    with open(log_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round",
            "alpha",
            "seed",
            "selected_clients",
            "cluster_map",
            "global_test_loss",
            "global_test_acc"
        ])

        for rnd in range(total_rounds):
            print(f"\n--- Round {rnd + 1}/{total_rounds} ---")

            selected_clients = random.sample(
                list(client_dict.keys()), clients_per_round
            )
            client_state_map = {}
            client_size_map = {}

            global_state_before = {
                k: v.detach().cpu().clone()
                for k, v in global_model.state_dict().items()
            }

            # Local training
            for client_id in selected_clients:
                indices = client_dict[client_id]
                client_loader = get_client_loader(
                    train_dataset, indices, batch_size=batch_size
                )

                local_state = train_local(
                    global_model,
                    client_loader,
                    device,
                    local_epochs=local_epochs
                )

                client_state_map[client_id] = local_state
                client_size_map[client_id] = len(indices)
                print(f"Client {client_id} trained on {len(indices)} samples.")

            if rnd < warmup_rounds:
                # FedAvg warm-up rounds: no clustering, just global averaging
                local_states = list(client_state_map.values())
                client_sizes = list(client_size_map.values())
                new_global_state = weighted_average_weights(
                    local_states, client_sizes
                )
                cluster_map = {}  # no clusters yet
            else:
                # Clustered rounds
                sim_matrix, client_ids, client_vectors = build_similarity_matrix(
                    global_state_before,
                    list(client_state_map.values()),
                    selected_clients
                )

                labels = cluster_clients(
                    client_vectors, num_clusters=num_clusters
                )
                cluster_map = group_clients_by_cluster(client_ids, labels)

                print("Cluster assignments:")
                for cluster_id, members in cluster_map.items():
                    print(f"Cluster {cluster_id}: {members}")

                cluster_models = []
                cluster_sizes = []

                for cluster_id, members in cluster_map.items():
                    member_states = [client_state_map[cid] for cid in members]
                    member_sizes = [client_size_map[cid] for cid in members]

                    cluster_model = weighted_average_weights(
                        member_states, member_sizes
                    )
                    cluster_models.append(cluster_model)
                    cluster_sizes.append(sum(member_sizes))

                new_global_state = average_cluster_models(
                    cluster_models, cluster_sizes
                )

            global_model.load_state_dict(new_global_state)

            test_loss, test_acc = evaluate(global_model, test_loader, device)
            print(
                f"Global Test Loss: {test_loss:.4f}, "
                f"Global Test Acc: {test_acc:.2f}%"
            )

            writer.writerow([
                rnd + 1,
                alpha,
                seed,
                selected_clients,
                dict(cluster_map),
                test_loss,
                test_acc
            ])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    alpha = 1.0          # match main FedAvg comparison
    seeds = [0, 1, 2]    # 3 seeds

    for seed in seeds:
        run_clustered_experiment(alpha, seed, device)


if __name__ == "__main__":
    main()