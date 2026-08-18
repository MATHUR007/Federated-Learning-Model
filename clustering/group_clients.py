# clustering/group_clients.py

from collections import defaultdict


def group_clients_by_cluster(client_ids, cluster_labels):
    cluster_map = defaultdict(list)

    for client_id, label in zip(client_ids, cluster_labels):
        cluster_map[int(label)].append(client_id)

    return dict(cluster_map)