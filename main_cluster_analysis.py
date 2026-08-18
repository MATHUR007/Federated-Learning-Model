# main_cluster_analysis.py

import os
import csv
import random
import torch
import numpy as np
import pandas as pd
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


def save_similarity_matrix(sim_matrix, client_ids, round_num, save_dir="results/clustering"):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(sim_matrix, index=client_ids, columns=client_ids)
    df.to_csv(f"{save_dir}/similarity_round_{round_num}.csv")


def save_cluster_labels(client_ids, labels, round_num, save_dir="results/clustering"):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame({
        "client_id": client_ids,
        "cluster": labels
    })
    df.to_csv(f"{save_dir}/clusters_round_{round_num}.csv", index=False)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

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
    rounds = 5
    local_epochs = 1
    alpha = 0.5
    num_clusters = 2

    client_dict = dirichlet_partition(train_dataset, num_clients=num_clients, alpha=alpha)
    global_model = SimpleCNN().to(device)

    os.makedirs("results/clustering", exist_ok=True)

    log_file = "results/clustering/cluster_analysis_log.csv"
    with open(log_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round",
            "selected_clients",
            "client_sizes",
            "cluster_labels",
            "global_test_loss",
            "global_test_acc"
        ])

        for rnd in range(rounds):
            print(f"\n--- Round {rnd+1}/{rounds} ---")

            selected_clients = random.sample(list(client_dict.keys()), clients_per_round)
            local_states = []
            client_sizes = []

            global_state_before = {
                k: v.detach().cpu().clone()
                for k, v in global_model.state_dict().items()
            }

            for client_id in selected_clients:
                indices = client_dict[client_id]
                client_loader = get_client_loader(train_dataset, indices, batch_size=64)

                local_state = train_local(
                    global_model,
                    client_loader,
                    device,
                    local_epochs=local_epochs
                )

                local_states.append(local_state)
                client_sizes.append(len(indices))
                print(f"Client {client_id} trained on {len(indices)} samples.")

            sim_matrix, client_ids, client_vectors = build_similarity_matrix(
                global_state_before,
                local_states,
                selected_clients
            )

            labels = cluster_clients(client_vectors, num_clusters=num_clusters)

            print("Cluster assignments:")
            for cid, label in zip(client_ids, labels):
                print(f"Client {cid} -> Cluster {label}")

            save_similarity_matrix(sim_matrix, client_ids, rnd + 1)
            save_cluster_labels(client_ids, labels, rnd + 1)

            new_global_state = weighted_average_weights(local_states, client_sizes)
            global_model.load_state_dict(new_global_state)

            test_loss, test_acc = evaluate(global_model, test_loader, device)
            print(f"Global Test Loss: {test_loss:.4f}, Global Test Acc: {test_acc:.2f}%")

            writer.writerow([
                rnd + 1,
                selected_clients,
                client_sizes,
                list(labels),
                test_loss,
                test_acc
            ])


if __name__ == "__main__":
    main()