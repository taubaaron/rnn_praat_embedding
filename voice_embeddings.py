from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.stats import mode
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import os
import umap.umap_ as umap
from itertools import groupby
from tqdm import tqdm


def to_embedding(path):
    fpath = path
    wav = preprocess_wav(fpath)

    encoder = VoiceEncoder()
    embed = encoder.embed_utterance(wav)
    np.set_printoptions(precision=3, suppress=True)
    return embed


def to_clustering(digits):
    # Project the data: this step will take several seconds
    tsne = TSNE(n_components=2, init='random', random_state=0)
    digits_proj = tsne.fit_transform(digits[0])  # 1797 x 64 == num_examples x example_dim ours: examples x 256 => examples x 2
    # Compute the clusters
    kmeans = KMeans(n_clusters=10, random_state=0)
    clusters = kmeans.fit_predict(digits_proj)

    # Permute the labels
    # labels = np.zeros_like(clusters)
    # for i in range(10):
    #     mask = (clusters == i)
    #     labels[mask] = mode(digits[1])[0]

    plt.scatter(digits_proj[:, 0], digits_proj[:, 1], s=50, cmap='viridis')


def web_clustering(emb_mat):
    # Project the data: this step will take several seconds
    tsne = TSNE(n_components=2, init='random', random_state=0)
    digits_proj = tsne.fit_transform(emb_mat)  # examples x dim => examples x 2

    # Compute the clusters
    kmeans = KMeans(n_clusters=10, random_state=0)
    clusters = kmeans.fit_predict(digits_proj)

    """    # Permute the labels
    labels = np.zeros_like(clusters)
    for i in range(10):
        mask = (clusters == i)
        labels[mask] = mode(target[mask])[0]
    # Compute the accuracy
    accuracy_score(target, clusters)
    """
    # plot
    plt.scatter(digits_proj[:, 0], digits_proj[:, 1], c=clusters, s=50, cmap="tab10")
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.colorbar()
    plt.show()


def plot_projections(embeds, speakers, ax=None, markers=None, legend=True, title="", **kwargs):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # Compute the 2D projections. You could also project to another number of dimensions (e.g.
    # for a 3D plot) or use a different different dimensionality reduction like PCA or TSNE.
    reducer = umap.UMAP(**kwargs)
    projs = reducer.fit_transform(embeds)

    # Draw the projections
    speakers = np.array(speakers)

    colors_dict = {
    "f1": [0/255, 0/255, 0/255],
    "f2": [255/255, 0/255, 0/255],
    "f3": [0/255, 255/255, 0/255],
    "f4": [0/255, 0/255, 255/255],
    "f5": [255/255, 255/255, 0/255],
    "m1": [255/255, 0/255, 255/255],
    "m2": [0/255, 255/255, 255/255],
    "m3": [128/255, 0/255, 0/255],
    "m4": [0/255, 128/255, 0/255],
    "m5": [0/255, 128/255, 128/255],
    }

    for i, speaker in enumerate(np.unique(speakers)):
        speaker_projs = projs[speakers == speaker]
        marker = "o" if markers is None else markers[i]
        label = speaker if legend else None
        ax.scatter(*speaker_projs.T, c=[colors_dict[speaker]], marker=marker, label=label)

    if legend:
        ax.legend(title="Speakers", ncol=2)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    return projs


def resemblyzer_clustering(wavs, speakers):
    plot_projections(wavs, speakers, title="Embedding projections")
    plt.show()


if __name__ == "__main__":
    embedding_vectors = []
    speakers = []

    directory = r'/Users/aarontaub/Desktop/specific_nhss'
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            embed = to_embedding(Path(f"{directory}/{filename}"))
            embedding_vectors.append(embed)
            speakers.append(filename[0:2])
        else:
            continue
    print(embedding_vectors)
    embedding_matrix = np.array(embedding_vectors)

    # original code
    # digits = load_digits()  # examples X dim
    # web_clustering(digits.data)
    # web_clustering(embedding_matrix)


    resemblyzer_clustering(embedding_matrix, speakers)






    # # Amit
    # embed1 = to_embedding(Path("/Users/aarontaub/Google Drive/AaronAndAmitBIU/FinalProject/Praat/whitney houston.wav"))
    # embed2 = to_embedding(Path("/Users/aarontaub/Google Drive/AaronAndAmitBIU/FinalProject/Praat/whitney houston.wav"))
    # embed3 = to_embedding(Path("/Users/aarontaub/Google Drive/AaronAndAmitBIU/FinalProject/Praat/whitney houston.wav"))
    #
    # data = np.vstack( (embed1 , embed2) )
    # data = np.vstack( (data , embed3) )
    #
    # target = np.array([1,1,1])
    #
    # web_clustering(data,target)

# Test for commit