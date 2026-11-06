# Self-contained demo: clusters of known size and distance, embedded at several perplexities.
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from embedding_checks import run_tsne, knn_preservation, spread_ratio, centroid_distance_corr


def simulate(seed=0, dim=50):
    rng = np.random.default_rng(seed)
    specs = [("tight", 0.5, 0), ("wide", 3.0, 0), ("near", 0.5, 12), ("far", 0.5, 60)]
    X, labels = [], []
    for name, sd, shift in specs:
        centre = np.zeros(dim)
        centre[0] = shift
        centre[1] = 8 if name == "wide" else 0
        X.append(rng.normal(centre, sd, size=(120, dim)))
        labels += [name] * 120
    return np.vstack(X), np.array(labels)


def main():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    X, labels = simulate()
    true_ratio = spread_ratio(X, labels, "tight", "wide")
    embeddings = {"PCA (reference)": PCA(n_components=2).fit_transform(X)}
    for p in (3, 30, 150):
        embeddings[f"t-SNE perplexity {p}"] = run_tsne(X, perplexity=p)
    noise = np.random.default_rng(1).normal(size=(480, 50))
    noise_emb = run_tsne(noise, perplexity=3)

    rows = []
    for name, Y in embeddings.items():
        rows.append({"embedding": name, "knn_preserved": round(knn_preservation(X, Y), 3),
                     "wide_vs_tight_size_ratio": round(spread_ratio(Y, labels, "tight", "wide"), 2),
                     "true_size_ratio": round(true_ratio, 2),
                     "between_cluster_distance_rank_corr": round(centroid_distance_corr(X, Y, labels), 2)})
    table = pd.DataFrame(rows)
    table.to_csv("results/summary.csv", index=False)

    colours = {"tight": "#1f77b4", "wide": "#ff7f0e", "near": "#2ca02c", "far": "#d62728"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    for ax, (name, Y) in zip(axes.flat, embeddings.items()):
        for g, c in colours.items():
            m = labels == g
            ax.scatter(Y[m, 0], Y[m, 1], s=6, color=c, label=g)
        r = table.set_index("embedding").loc[name]
        ax.set_title(f"{name}\nsize ratio {r['wide_vs_tight_size_ratio']} (truth {true_ratio:.1f}), "
                     f"kNN kept {r['knn_preserved']}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    axes.flat[0].legend(fontsize=8, markerscale=2)
    ax = axes.flat[4]
    ax.scatter(noise_emb[:, 0], noise_emb[:, 1], s=6, color="grey")
    ax.set_title("t-SNE (perplexity 3) of pure Gaussian noise", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax = axes.flat[5]
    ax.axis("off")
    ax.table(cellText=table[["embedding", "knn_preserved", "wide_vs_tight_size_ratio",
                             "between_cluster_distance_rank_corr"]].values,
             colLabels=["embedding", "kNN kept", "size ratio", "distance rank corr"],
             loc="center", fontsize=9).scale(1, 1.6)
    fig.tight_layout()
    fig.savefig("figures/demo.png", dpi=110)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
