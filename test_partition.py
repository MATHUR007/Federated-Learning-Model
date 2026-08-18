from torchvision import datasets, transforms
from datasets.partition import iid_partition, dirichlet_partition, label_distribution

transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

iid_clients = iid_partition(train_dataset, num_clients=5)
dir_clients = dirichlet_partition(train_dataset, num_clients=5, alpha=0.5)

print("IID distribution:")
print(label_distribution(train_dataset, iid_clients))

print("\nSamples per Dirichlet client:")
for client_id, indices in dir_clients.items():
    print(f"Client {client_id}: {len(indices)} samples")