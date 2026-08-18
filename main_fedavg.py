import csv
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from data.load_data import get_cifar10_loaders
from models.cnn import SimpleCNN
from datasets.partition import dirichlet_partition
from datasets.client_dataset import get_client_loader
from train.local_train import train_local
from server.fedavg import weighted_average_weights


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

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


def run_fedavg_experiment(alpha: float, seed: int, device: torch.device):
    print(f"\n=== FedAvg run | alpha={alpha} | seed={seed} ===")
    set_seed(seed)

    # Data
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

    # Federated settings
    num_clients = 20
    clients_per_round = 10
    rounds = 20
    local_epochs = 1
    batch_size = 64

    client_dict = dirichlet_partition(
        train_dataset,
        num_clients=num_clients,
        alpha=alpha
    )

    global_model = SimpleCNN().to(device)

    results_file = f"results/fedavg_alpha_{alpha}_seed{seed}.csv"
    os.makedirs("results", exist_ok=True)

    with open(results_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round",
            "alpha",
            "seed",
            "num_clients",
            "clients_per_round",
            "local_epochs",
            "selected_clients",
            "client_sizes",
            "global_test_loss",
            "global_test_acc"
        ])

        for rnd in range(rounds):
            print(f"\n--- Round {rnd + 1}/{rounds} ---")

            selected_clients = random.sample(list(client_dict.keys()),
                                             clients_per_round)
            local_states = []
            client_sizes = []

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

                local_states.append(local_state)
                client_sizes.append(len(indices))
                print(f"Client {client_id} trained on {len(indices)} samples.")

            new_global_state = weighted_average_weights(local_states,
                                                        client_sizes)
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
                num_clients,
                clients_per_round,
                local_epochs,
                selected_clients,
                client_sizes,
                test_loss,
                test_acc
            ])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Alphas to explore
    alphas = [0.1, 0.5, 1.0]
    seeds = [0, 1, 2]   # 3 random seeds

    for alpha in alphas:
        for seed in seeds:
            run_fedavg_experiment(alpha, seed, device)


if __name__ == "__main__":
    main()