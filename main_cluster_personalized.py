# main_cluster_personalized.py

import os
import csv
import copy
import random
import torch
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


def evaluate_cluster_models(cluster_models, test_loader, device):
    results = {}
    for cluster_id, state_dict in cluster_models.items():
        model = SimpleCNN().to(device)
        model.load_state_dict(state_dict)
        loss, acc = evaluate(model, test_loader, device)
        results[cluster_id] = {"loss": loss, "acc": acc}
    return results


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

    base_model = SimpleCNN().to(device)
    base_state = copy.deepcopy(base_model.state_dict())

    cluster_models = {i: copy.deepcopy(base_state) for i in range(num_clusters)}
    client_cluster_assignments = {}

    os.makedirs("results/cluster_personalized", exist_ok=True)
    log_file = "results/cluster_personalized/cluster_personalized_log.csv"

    with open(log_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round",
            "selected_clients",
            "cluster_map",
            "cluster_eval"
        ])

        for rnd in range(rounds):
            print(f"\n--- Round {rnd+1}/{rounds} ---")
            selected_clients = random.sample(list(client_dict.keys()), clients_per_round)

            local_states = []
            selected_client_ids = []

            reference_model = SimpleCNN().to(device)
            reference_model.load_state_dict(base_state)

            for client_id in selected_clients:
                if client_id in client_cluster_assignments:
                    assigned_cluster = client_cluster_assignments[client_id]
                    start_state = cluster_models[assigned_cluster]
                else:
                    start_state = base_state

                client_model = SimpleCNN().to(device)
                client_model.load_state_dict(start_state)

                indices = client_dict[client_id]
                client_loader = get_client_loader(train_dataset, indices, batch_size=64)

                local_state = train_local(
                    client_model,
                    client_loader,
                    device,
                    local_epochs=local_epochs
                )

                local_states.append(local_state)
                selected_client_ids.append(client_id)

                print(f"Client {client_id} trained on {len(indices)} samples.")

            sim_matrix, client_ids, client_vectors = build_similarity_matrix(
                base_state,
                local_states,
                selected_client_ids
            )

            labels = cluster_clients(client_vectors, num_clusters=num_clusters)
            cluster_map = group_clients_by_cluster(client_ids, labels)

            print("Cluster assignments:")
            for cluster_id, members in cluster_map.items():
                print(f"Cluster {cluster_id}: {members}")
                for cid in members:
                    client_cluster_assignments[cid] = cluster_id

            for cluster_id, members in cluster_map.items():
                member_states = []
                member_sizes = []

                for cid in members:
                    idx = selected_client_ids.index(cid)
                    member_states.append(local_states[idx])
                    member_sizes.append(len(client_dict[cid]))

                cluster_models[cluster_id] = weighted_average_weights(member_states, member_sizes)

            cluster_eval = evaluate_cluster_models(cluster_models, test_loader, device)

            for cluster_id, metrics in cluster_eval.items():
                print(f"Cluster {cluster_id} -> Loss: {metrics['loss']:.4f}, Acc: {metrics['acc']:.2f}%")

            writer.writerow([
                rnd + 1,
                selected_clients,
                dict(cluster_map),
                cluster_eval
            ])
            
        torch.save(cluster_models, "results/cluster_personalized/cluster_models.pt")
        torch.save(client_cluster_assignments, "results/cluster_personalized/client_cluster_assignments.pt")


if __name__ == "__main__":
    main()