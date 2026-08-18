
import numpy as np
from collections import defaultdict


def iid_partition(dataset, num_clients=20):
    num_items = len(dataset)
    indices = np.random.permutation(num_items)
    split_indices = np.array_split(indices, num_clients)

    client_dict = {i: split_indices[i].tolist() for i in range(num_clients)}
    return client_dict


def dirichlet_partition(dataset, num_clients=20, alpha=1.0, num_classes=10):
    labels = np.array(dataset.targets)
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)

        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        split_class_indices = np.split(class_indices, proportions)

        for client_id, idxs in enumerate(split_class_indices):
            client_indices[client_id].extend(idxs.tolist())
    
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])

    client_dict = {i: client_indices[i] for i in range(num_clients)}
    return client_dict


def label_distribution(dataset, client_dict, num_classes=10):
    labels = np.array(dataset.targets)
    distribution = {}

    for client_id, indices in client_dict.items():
        client_labels = labels[indices]
        class_counts = np.bincount(client_labels, minlength=num_classes)
        distribution[client_id] = class_counts.tolist()

    return distribution