import numpy as np
from typing import Tuple, Callable, List


def l1_dist(X_1: np.ndarray, X_2: np.ndarray, **kwargs) -> np.ndarray:
    """
    Computes manhattan distance
    """
    return np.sum(np.abs(X_1 - X_2), **kwargs)


def l2_dist(X_1: np.ndarray, X_2, **kwargs) -> np.ndarray:
    """
    Computes euclidian distance
    """
    return np.linalg.norm(X_1 - X_2, **kwargs)


def initialize_clusters(points: np.ndarray, K: int, init_centroids: np.ndarray | None) -> np.ndarray:
    """
    Initialize clusters and centroids for k-clustering algorithms
    :param points: set of all points
    :param K: number of clusters
    :param init_centroids: if none, nothing happens. Else, they're randomly selected from points
    :returns: centroids as an array
    """
    N, d = points.shape
    if init_centroids is None:
        idxs = np.random.choice(np.arange(N), K, replace=False)
        init_centroids  = points[idxs]
    assert len(init_centroids) == K
    centroids = init_centroids
    return centroids


def k_means(points: np.ndarray,
            K: int,
            init_centroids: np.ndarray | None,
            max_iter: int = 10,
            dist: Callable=np.linalg.norm) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    K means clusterization algorithm
    :param points: 2D array, where each row is a point
    :param K: number of clusters
    :param init_centroids: initial cluster centroids
    :param max_iter: number of iterations
    :param dist: for computing distance
    :returns:
        centroids: list of centers
        clusters: list of clusters represented in matrices, s.t. the
        concatenation of all clusters includes all initial points
    """
    centroids = initialize_clusters(points, K, init_centroids)
    clusters = []

    for _ in range(max_iter):
        clusters = [[] for _ in range(K)]
        # Step 1: group points to the closest centroids
        for point in points:
            clust_idx = np.argmin(dist(point, centroids, axis=1))
            clusters[clust_idx].append(point)

        # Step 2: update centroids
        centroids = [np.mean(np.array(cluster), axis=0) for cluster in clusters]

    clusters = [np.array(cluster) for cluster in clusters]
    return centroids, clusters


def k_medoids(points: np.ndarray,
              K: int,
              init_centroids: np.ndarray | None,
              max_iter: int = 10,
              dist: Callable=np.linalg.norm) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    K medoids clusterization algorithm
    :param points: 2D array, where each row is a point
    :param K: number of clusters
    :param init_centroids: initial cluster centroids
    :param max_iter: number of iterations
    :param dist: for computing distance
    :returns:
        centroids: list of centers
        clusters: list of clusters represented in matrices, s.t. the
        concatenation of all clusters includes all initial points
    """
    centroids = initialize_clusters(points, K, init_centroids)
    clusters = []

    for _ in range(max_iter):
        clusters = [[] for _ in range(K)]
        # Step 1: group points to the closest centroids
        for point in points:
            clust_idx = np.argmin(dist(point, centroids, axis=1))
            clusters[clust_idx].append(point)

        # Step 2: update centroids
        centroids = []
        for cluster in clusters:
            cluster = np.array(cluster)
            z_med = cluster[0]
            min_dist = np.inf
            # Find point that's closest to all neighbours
            for z in cluster:
                sum_dist = sum([dist(z, point) for point in cluster])
                if sum_dist < min_dist:
                    min_dist = sum_dist
                    z_med = z

            centroids.append(z_med)

    clusters = [np.array(cluster) for cluster in clusters]
    return centroids, clusters

def main():
    # Initialization
    points = np.array([[0, -6],
                       [4, 4],
                       [0, 0],
                       [-5, 2]])
    init_centroids = points[np.array([-1, 0])]

    # Part 1: k-medoids with l1 dist
    print('k-medoids, l1 dist')
    centroids, clusters = k_medoids(points, 2, init_centroids=init_centroids, max_iter=30, dist=l1_dist)
    for clus, cent in zip(clusters, centroids):
        print('centroid:', cent)
        print(clus)
    print()

    # Part 2: k-medoids with l2 dist
    print('k-medoids, l2 dist')
    centroids, clusters = k_medoids(points, 2, init_centroids=init_centroids, max_iter=30, dist=l2_dist)
    for clus, cent in zip(clusters, centroids):
        print('centroid:', cent)
        print(clus)
    print()
    
    # Part 3: k-means with l1 dist
    print('k-means, l1 dist')
    centroids, clusters = k_means(points, 2, init_centroids=init_centroids, max_iter=20, dist=l1_dist)
    for clus, cent in zip(clusters, centroids):
        print('centroid:', cent)
        print(clus)


if __name__ == '__main__':
    main()