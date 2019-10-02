import sklearn.cluster as cluster
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import sys
np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.max_columns", 100)

from utils import *

df = load_dataset()
X, y = split_data(df)

print("Performing PCA analysis")
n_component_values = [10, 15, 20, 30]
pca_clfs = []
for n_component in n_component_values:
    pca = PCA(n_components = n_component)
    pca.fit(X)
    pca_clfs.append(pca)
    #print(pca.explained_variance_ratio_)
    print(sum(pca.explained_variance_ratio_))
X = pca_clfs[2].transform(X) # take n_components = 20 and transform the data

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, shuffle = True)

dbscan_clfs = []
eps_vals = [0.2, 0.5, 1.0]
min_samples_vals = [3, 5, 10]
for eps in eps_vals:
    for min_samples in min_samples_vals:
        print("Training DBScan with eps = {} and min_samples = {}...".format(eps, min_samples))
        clf = cluster.DBSCAN(eps = eps, min_samples = min_samples)
        pred = clf.fit_predict(X_train, y_train)
        dbscan_clfs.append(clf)
        num_outliers = list(pred.flatten()).count(-1) # samples not assigned to a cluster
        score = "N/A"
        if len(np.unique(pred)) > 1:
            score = silhouette_score(X_train, pred)
        print("Epsilon: {}, Min samples: {}, Num outliers: {}, Score: {}".format(eps, min_samples, num_outliers, score))

kmeans_clf = cluster.KMeans(n_clusters = 7)
kmeans_clf.fit(X_train, y_train)
score = kmeans_clf.score(X_train, y_train)
print("Kmeans score: {}".format(score))
