# evaluate_personalized.py

import os
import csv
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.cnn import SimpleCNN
from datasets.partition import dirichlet_partition


def get_client_test_loaders(test_dataset, num_clients=20, alpha=0.5, batch_size=64):
    client_test_dict = dirichlet_partition(
        test_dataset, num_clients=num_clients, alpha=alpha
    )
    client_test_loaders = {}

    for client_id, indices in client_test_dict.items():
        subset = Subset(test_dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
        client_test_loaders[client_id] = loader

    return client_test_loaders


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    acc = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, acc


def evaluate_personalized_models(cluster_models, client_cluster_assignments,
                                 client_test_loaders, device):
    results = []

    for client_id, loader in client_test_loaders.items():
        if client_id not in client_cluster_assignments:
            continue

        cluster_id = client_cluster_assignments[client_id]
        state_dict = cluster_models[cluster_id]

        model = SimpleCNN().to(device)
        model.load_state_dict(state_dict)

        loss, acc = evaluate(model, loader, device)

        results.append({
            "client_id": client_id,
            "cluster_id": cluster_id,
            "test_loss": loss,
            "test_acc": acc
        })

    return results


def save_results(results, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fieldnames = [
        "client_id",
        "cluster_id",
        "test_loss",
        "test_acc"
    ]

    with open(save_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_summary(results):
    if not results:
        print("No results found.")
        return

    accs = np.array([r["test_acc"] for r in results], dtype=float)
    losses = np.array([r["test_loss"] for r in results], dtype=float)

    avg_acc = float(np.mean(accs))
    avg_loss = float(np.mean(losses))
    worst_acc = float(np.min(accs))
    std_acc = float(np.std(accs))

    print("\n--- Personalized Evaluation Summary ---")
    print(f"Average Client Test Loss: {avg_loss:.4f}")
    print(f"Average Client Test Accuracy (mean): {avg_acc:.2f}%")
    print(f"Worst Client Test Accuracy: {worst_acc:.2f}%")
    print(f"Std Dev of Client Accuracy: {std_acc:.2f}%")

    for r in results:
        print(
            f"Client {r['client_id']} | Cluster {r['cluster_id']} | "
            f"Loss: {r['test_loss']:.4f} | Acc: {r['test_acc']:.2f}%"
        )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    num_clients = 20
    alpha = 0.5   # or 1.0 if you want to match main comparison
    batch_size = 64

    client_test_loaders = get_client_test_loaders(
        test_dataset,
        num_clients=num_clients,
        alpha=alpha,
        batch_size=batch_size
    )

    # These should be produced by your training scripts
    cluster_models = torch.load(
        "cluster_personalized/cluster_models.pt",
        map_location=device
    )
    client_cluster_assignments = torch.load(
        "cluster_personalized/client_cluster_assignments.pt",
        map_location=device
    )

    results = evaluate_personalized_models(
        cluster_models,
        client_cluster_assignments,
        client_test_loaders,
        device
    )

    save_results(
        results,
        "cluster_personalized/personalized_client_results.csv"
    )
    print_summary(results)


if __name__ == "__main__":
    main()