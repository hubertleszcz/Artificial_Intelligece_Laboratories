import numpy as np
import copy
import random
import math


def findDistance(centrOne, centrTwo):
    return np.linalg.norm(np.array(centrOne) - np.array(centrTwo))

def initialize_centroids_forgy(data, k):
    centroids = random.sample(list(data), k)
    return centroids

def initialize_centroids_kmeans_pp(data, k):
    centroids = [random.choice(data)]

    for _ in range(1, k):
        distances = []
        for point in data:
            min_distance = float('inf')
            for centroid in centroids:
                dist = findDistance(point, centroid)
                if dist < min_distance:
                    min_distance = dist
            distances.append(min_distance)
        max_dist_index = distances.index(max(distances))
        new_centroid = data[max_dist_index]
        centroids.append(new_centroid)

    return centroids


def assign_to_cluster(data, centroids):
    assignments = np.zeros(len(data), dtype=int)
    for i in range(len(data)):
        minDist = float('inf')
        for j in range(len(centroids)):
            dist = findDistance(data[i], centroids[j])
            if dist < minDist:
                minDist = dist
                assignments[i] = j
    return assignments

def update_centroids(data, assignments):
    centroids = []
    for cluster in assignments:
        points = data[assignments == cluster]
        if len(points) > 0:
            mean = np.mean(points, axis=0)
            centroids.append(mean)
    return centroids

def mean_intra_distance(data, assignments, centroids):
    centroids = np.array(centroids)
    return np.sqrt(np.sum((data - centroids[assignments, :]) ** 2))



def k_means(data, num_centroids, kmeansplusplus=False):
    # centroids initizalization
    new_assignments = []
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else:
        centroids = initialize_centroids_forgy(data, num_centroids)

    assignments = assign_to_cluster(data, centroids)
    for i in range(100):  # max number of iteration = 100
        # print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.array_equal(new_assignments, assignments):  # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)
