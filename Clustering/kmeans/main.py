from k_means import k_means
import pandas as pd
import numpy as np

def load_iris():
    data = pd.read_csv("data/iris.data", names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])
    # print(data)
    classes = data["class"].to_numpy()
    features = data.drop("class", axis=1).to_numpy()
    return features, classes

def evaluate(clusters, labels):
    for cluster in np.unique(clusters):
        labels_in_cluster = labels[clusters==cluster]
        print(f"Cluster: {cluster}")
        for label_type in np.unique(labels):
            print(f"Num of {label_type}: {np.sum(labels_in_cluster==label_type)}")
    

def clustering(kmeans_pp):
    data = load_iris()
    features, classes = data
    intra_class_variance = []
    assignments = []
    for i in range(100):
        assignments, centroids, error = k_means(features, 3, kmeans_pp)

        intra_class_variance.append(error)
    # evaluate(assignments, classes)
    print(f"Mean intra-class variance: {np.mean(intra_class_variance)}")

if __name__=="__main__":
    print("k-means++")
    clustering(kmeans_pp = True)
    print("k-means Forgy")
    clustering(kmeans_pp = False)
