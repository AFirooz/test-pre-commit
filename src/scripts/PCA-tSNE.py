# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,scripts//py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# Source: https://www.kaggle.com/code/akhileshrai/intro-cnn-pytorch-pca-tnse-isomap

from torchvision import transforms

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import manifold
from sklearn.preprocessing import RobustScaler

import warnings
import os.path as path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import filestructure as fs
import nn_preprocessor as nnp

warnings.filterwarnings("ignore", category=Warning)


def plot_nD(x: str, y: str, z: str, data: pd.DataFrame, labels: str = "labels"):
    # this is needed for 3D plotting method!
    color: pd.Series = data[labels].copy()
    color.replace([0, 1], ["r", "b"], inplace=True)

    fig = plt.figure(dpi=200)

    # 2D
    if z not in list(data.columns):
        ax = fig.add_subplot(1, 1, 1)
        sns.scatterplot(
            x=x,
            y=y,
            data=data,
            hue=labels,
            legend=False,
            ax=ax,
        )
    # 3D
    else:
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.scatter(
            x,
            y,
            z,
            data=data,
            c=color,
        )
    plt.show()
    return fig, ax


def plot_side_by_side(
    x: str, y: str, z: str, data: pd.DataFrame, labels: str = "labels"
):
    if z in list(data.columns) or len(list(data.columns)) > 3:
        print("Only usable for 2D data")
        return

    fig, axs = plt.subplots(ncols=2, nrows=1, dpi=200, figsize=(8, 4))
    axs = axs.flatten()
    flt0 = data[labels] == 0
    flt1 = data[labels] == 1

    sns.scatterplot(
        x=x,
        y=y,
        data=data.loc[flt0],
        color="r",
        legend=False,
        ax=axs[0],
    )
    sns.scatterplot(
        x=x,
        y=y,
        data=data.loc[flt1],
        color="b",
        legend=False,
        ax=axs[1],
    )
    plt.show()
    return fig, axs


savefig = True
labels = nnp.Labels("mz")
grouping = nnp.Grouping("mz", group_init="person")
outdir = path.join(fs.OutputPath.lcms, "dim reduction")

data = nnp.CustomDataset("mz", labels.tissue_type, grouping.result, transpose=False)
data.pre_transforms(transform=transforms.Compose([nnp.ColPadding(data.all_cols)]))

df = data.to_df()
# -

# # Get the DataFrame

# +
# making sure the sizes are correct
labels = df["labels"].tolist()
print(f"rows: {df.shape[0]} \ncols: {df.shape[1]}")

assert df.head().shape[1] == data.__getitem__(5)[0].shape[0]
assert df.shape[0] == len(labels), "Your labels are wrong!"

df = df.drop(columns=["groups", "labels"])
# -

# # Normalization

# +
df_scaled = df
df_col_names = list(df.columns)
df_index = list(df.index)

# df_scaled = pd.DataFrame(df_scaled, columns=df_col_names, index=df_index) if isinstance(df_scaled, np.ndarray) else df_scaled

# Scaling
# df_scaled = StandardScaler().fit_transform(df_scaled)
# df_scaled = MinMaxScaler().fit_transform(df_scaled)
df_scaled = RobustScaler().fit_transform(df_scaled)

# recreating the DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=df_col_names, index=df_index)
# -

# # PCA
# Principal Component Analysis, or PCA, is a dimensionality-reduction method that is often used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information in the large set. Reducing the number of variables of a data set naturally comes at the expense of accuracy, but the trick in dimensionality reduction is to trade a little accuracy for simplicity. Because smaller data sets are easier to explore and visualize and make analyzing data much easier and faster for machine learning algorithms without extraneous variables to process. So to sum up, the idea of PCA is simple — reduce the number of variables of a data set, while preserving as much information as possible.

# +
# perform PCA
n_pc = 2

pca = PCA(n_components=n_pc)
pc = pca.fit_transform(df_scaled)
pc_col_names = [f"PC{i + 1}" for i in range(n_pc)]

# add labels
principal_df = pd.DataFrame(data=pc, columns=pc_col_names)
labels = pd.DataFrame(data=labels, columns=["labels"])
principal_df = pd.concat(
    [principal_df, labels], axis=1, join="inner", ignore_index=True
)

# removing duplicates since that won't help us (or add any extra information)
# principal_df = principal_df.loc[:, ~principal_df.columns.duplicated()]
principal_df = principal_df.drop_duplicates(ignore_index=True)
principal_df.columns = pc_col_names + ["labels"]

principal_df.head(3)

# +
# drop outlines (just experimenting)
p1 = principal_df["PC1"]
p2 = principal_df["PC2"]

principal_df["PC1"] = p1.loc[~(p1 > 500)]
principal_df["PC2"] = p2.loc[~(p2 > -20)].loc[~(p2 < -100)]
# -

# # Plotting the results

# +
fig, ax = plot_nD(x="PC1", y="PC2", z="PC3", data=principal_df)
fig.savefig(path.join(outdir, "pca_2d.png")) if savefig else None

fig, ax = plot_side_by_side(
    x="PC1",
    y="PC2",
    z="PC3",
    data=principal_df,
)
fig.savefig(path.join(outdir, "pca_2d_side.png")) if savefig else None
# -

# # t-SNE
#
# **t-Distributed Stochastic Neighbor Embedding (t-SNE)** is a (prize-winning) technique for dimensionality reduction that is particularly 
# well suited for the visualization of high-dimensional datasets. The technique can be implemented via Barnes-Hut approximations, allowing it to be applied on large real-world datasets. t-Distributed Stochastic Neighbor Embedding (t-SNE) is an unsupervised, non-linear technique primarily used for data exploration and visualizing high-dimensional data. In simpler terms, t-SNE gives you a feel or intuition of how the data is arranged in a high-dimensional space. It was developed by Laurens van der Maatens and Geoffrey Hinton in 2008.
#
# ## t-SNE vs PCA
# You’re probably wondering the difference between PCA and t-SNE. The first thing to note is that PCA was developed in 1933 while t-SNE was developed in 2008. A lot has changed in the world of data science since 1933 mainly in the realm of computing and size of data. Second, PCA is a linear dimension reduction technique that seeks to maximize variance and preserves large pairwise distances. In other words, things that are different end up far apart. This can lead to poor visualization, especially when dealing with non-linear manifold structures. Think of a manifold structure as any geometric shape like: cylinder, ball, curve, etc.
#
# t-SNE differs from PCA by preserving only small pairwise distances or local similarities, whereas PCA is concerned with preserving large pairwise distances to maximize variance. You can see that due to the non-linearity of this toy dataset (manifold) and preserving large distances that PCA would incorrectly preserve the structure of the data.

# +
n_ts = 2
ts_col_names = [f"t-SNE{i + 1}" for i in range(n_ts)]
ts = TSNE(n_components=n_ts, verbose=1, perplexity=40, n_iter=300)
tsne = ts.fit_transform(df_scaled)

# add labels
tsne = pd.DataFrame(data=tsne, columns=ts_col_names)
labels = pd.DataFrame(data=labels, columns=["labels"])
tsne = pd.concat([tsne, labels], axis=1, join="inner", ignore_index=True)

# removing duplicates since that won't help us (or add any extra information)
tsne = tsne.drop_duplicates(ignore_index=True)
tsne.columns = ts_col_names + ["labels"]

tsne.head(3)

# +
fig, ax = plot_nD(x="t-SNE1", y="t-SNE2", z="t-SNE3", data=tsne)
fig.savefig(path.join(outdir, "tsne_2d.png")) if savefig else None

fig, ax = plot_side_by_side(x="t-SNE1", y="t-SNE2", z="t-SNE3", data=tsne)
fig.savefig(path.join(outdir, "tsne_2d_side.png")) if savefig else None
# -

# # ISOMAP
# Isomap stands for **isometric mapping.** Isomap is a non-linear dimensionality reduction method based on the spectral theory which tries to 
# preserve the geodesic distances in the lower dimension. Isomap starts by creating a neighborhood network. After that, it uses graph distance to the approximate geodesic distance between all pairs of points. And then, through eigenvalue decomposition of the geodesic distance matrix, it finds the low dimensional embedding of the dataset. In non-linear manifolds, the Euclidean metric for distance holds good if and only if neighborhood structure can be approximated as linear. If a neighborhood contains holes, then Euclidean distances can be highly misleading. In contrast to this, if we measure the distance between two points by following the manifold, we will have a better approximation of how far or near two points are. Let's understand this with an extremely simple 2-D example.

# +
n_iso = 2
iso_col_names = [f"ISO{i + 1}" for i in range(n_iso)]

iso = manifold.Isomap(n_neighbors=6, n_components=n_iso)
iso.fit(df_scaled)
manifold_iso = iso.transform(df_scaled)

# add labels
manifold_iso = pd.DataFrame(data=manifold_iso, columns=iso_col_names)
labels = pd.DataFrame(data=labels, columns=["labels"])
manifold_iso = pd.concat(
    [manifold_iso, labels], axis=1, join="inner", ignore_index=True
)

# removing duplicates since that won't help us (or add any extra information)
manifold_iso = manifold_iso.drop_duplicates(ignore_index=True)
manifold_iso.columns = iso_col_names + ["labels"]

manifold_iso.head(3)

# +
fig, ax = plot_nD(x="ISO1", y="ISO2", z="ISO3", data=manifold_iso)

fig.savefig(path.join(outdir, "iso_2d.png")) if savefig else None

fig, ax = plot_side_by_side(x="ISO1", y="ISO2", z="ISO3", data=manifold_iso)
fig.savefig(path.join(outdir, "iso_2d_side.png")) if savefig else None
