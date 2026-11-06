# Quantitative checks for 2-D embeddings: what survives, what is invented.
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors


def run_tsne(X, perplexity=30, seed=0):
    return TSNE(n_components=2, perplexity=perplexity, init="pca",
                learning_rate="auto", random_state=seed).fit_transform(X)


def knn_preservation(X, Y, k=15):
    # Fraction of each point's k nearest high-dimensional neighbours that stay neighbours in 2-D.
    hi = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    lo = NearestNeighbors(n_neighbors=k + 1).fit(Y).kneighbors(Y, return_distance=False)[:, 1:]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(hi, lo)]))


def spread(points):
    return float(np.sqrt(((points - points.mean(axis=0)) ** 2).sum(axis=1).mean()))


def spread_ratio(Z, labels, a, b):
    # How much bigger cluster b looks than cluster a.
    return spread(Z[labels == b]) / spread(Z[labels == a])


def _rank(x):
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    return r


def centroid_distance_corr(X, Y, labels):
    # Spearman correlation between between-cluster distances before and after embedding.
    groups = np.unique(labels)
    cx = np.array([X[labels == g].mean(axis=0) for g in groups])
    cy = np.array([Y[labels == g].mean(axis=0) for g in groups])
    i, j = np.triu_indices(len(groups), 1)
    dx = np.linalg.norm(cx[i] - cx[j], axis=1)
    dy = np.linalg.norm(cy[i] - cy[j], axis=1)
    return float(np.corrcoef(_rank(dx), _rank(dy))[0, 1])
