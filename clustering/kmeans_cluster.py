# clustering/kmeans_cluster.py

from sklearn.cluster import KMeans


def cluster_clients(client_vectors, num_clusters=2, random_state=42):
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(client_vectors)
    return labels