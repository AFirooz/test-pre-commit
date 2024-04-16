# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,_notebooks//py:light
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
import numpy as np
import pandas as pd
import os.path as path

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from torchvision import transforms

import nn_preprocessor as nnp
import filestructure as fs

labels = nnp.Labels("mz")
grouping = nnp.Grouping("mz", group_init="person")
outdir = path.join(fs.OutputPath.lcms, "visuals")

savefig = True
# -

# # Looking at the mass to charge ratios

# +
dataT = nnp.CustomDataset("mz", labels.tissue_type, grouping.result, transpose=True)
dataT.pre_transforms(transform=transforms.Compose([nnp.ColPadding(dataT.all_cols)]))

dfT = dataT.to_df()
mz = np.array(dfT.index.tolist())
dfT_all = pd.DataFrame(
    {"mz": mz, "labels": dfT["labels"].to_list(), "groups": dfT["groups"].to_list()}
)

print(f"Number of mz: {len(mz)}")
# -

fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
sns.histplot(
    data=dfT_all, x="mz", bins=60, hue="labels", ax=ax, kde=True, element="step"
)
plt.savefig(path.join(outdir, "mz_hist_aggregated.png"), dpi=300) if savefig else print(
    "Fig not saved"
)
plt.show()

# +
fig, ax = plt.subplots(nrows=2, figsize=(10, 7), dpi=150, sharex=True)
sns.histplot(
    data=dfT_all,
    x="mz",
    bins=50,
    hue="groups",
    ax=ax[0],
    palette="crest",
    kde=True,
    kde_kws={"clip": (dfT_all["mz"].min(), dfT_all["mz"].max())},
)
ax[0].get_legend().remove()

sns.kdeplot(
    data=dfT_all,
    x="mz",
    hue="groups",
    ax=ax[1],
    legend=False,
    palette="Set1",
    clip=(dfT_all["mz"].min(), dfT_all["mz"].max()),
    bw_adjust=1.2,
)

plt.savefig(path.join(outdir, "mz_hist_groups.png"), dpi=300) if savefig else print(
    "Fig not saved"
)
plt.show()
# -

# ## will a person have 2 lines?

# +
# fig, ax = plt.subplots(nrows=2, figsize=(10, 7), dpi=150, sharex=True)
#
# grp_df = dfT_all[dfT_all['groups'] == 38]
#
# sns.histplot(data=grp_df, x='mz', bins=50, hue='labels', ax=ax[0], kde=True, element='step')
#
# sns.kdeplot(data=grp_df, x='mz', hue='groups', ax=ax[1],
#             legend=False,
#             palette='Set1',
#             clip=(dfT_all['mz'].min(), dfT_all['mz'].max()),
#             bw_adjust=1,
#             )
# # ax[1].set_ylim(0, 0.002)
# ax[1].set_yscale('log')
# ax[1].set_ylim(10e-8, 10e-3)
#
# plt.show()

# +
fig, ax = plt.subplots(
    nrows=20,
    ncols=2,
    sharex=True,
    figsize=(16, 50),
    sharey=True,
)
ax = ax.flatten()

for i in range(39):
    grp_df = dfT_all[dfT_all["groups"] == i]
    sns.histplot(
        data=grp_df, x="mz", bins=50, hue="labels", ax=ax[i], kde=True, element="step"
    )
    ax[i].get_legend().remove()
    ax[i].set_title(f"Patient {i}")

fig.tight_layout()
# plt.savefig(path.join(outdir, 'mz_hist.png'), dpi=300) if savefig else print("Fig not saved")
plt.savefig(path.join(outdir, "mz_hist_scaled.png"), dpi=300) if savefig else print(
    "Fig not saved"
)
plt.close(fig)
# -

# # Looking at retention times

# +
data = nnp.CustomDataset("mz", labels.tissue_type, grouping.result, transpose=False)
data.pre_transforms(transform=transforms.Compose([nnp.ColPadding(data.all_cols)]))

df = data.to_df()
rt = np.array(df.index.tolist())
df_all = pd.DataFrame(
    {"rt": rt, "labels": df["labels"].to_list(), "groups": df["groups"].to_list()}
)
rt_filter = rt > 1

df_all = df_all[rt_filter]

print(f"Number of retention times: {len(rt)}")
# -

fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
sns.histplot(
    data=df_all,
    x="rt",
    hue="labels",
    bins=50,
    ax=ax,
    element="step",
    kde=True,
    kde_kws={"bw_adjust": 0.8},
)
plt.savefig(path.join(outdir, "rt_hist_aggregated.png"), dpi=300) if savefig else print(
    "Fig not saved"
)
plt.show()

fig, ax = plt.subplots(figsize=(10, 7), dpi=150, nrows=2, sharex=True)
sns.histplot(
    data=df_all,
    x="rt",
    hue="groups",
    bins=50,
    ax=ax[0],
    element="step",
    palette="crest",
    kde=True,
)
sns.kdeplot(
    data=df_all,
    x="rt",
    hue="groups",
    ax=ax[1],
    legend=False,
    palette="tab10",
    bw_adjust=1.2,
    clip=(df_all["rt"].min(), 7),
)
ax[0].get_legend().remove()
plt.savefig(path.join(outdir, "rt_hist_groups.png"), dpi=300) if savefig else print(
    "Fig not saved"
)
plt.show()

# +
fig, ax = plt.subplots(
    nrows=20,
    ncols=2,
    figsize=(16, 50),
    sharex=True,
    sharey=True,
)
ax = ax.flatten()

for i in range(39):
    grp_df = df_all[df_all["groups"] == i]
    sns.histplot(
        data=grp_df, x="rt", bins=25, hue="labels", ax=ax[i], kde=True, element="step"
    )
    ax[i].get_legend().remove()
    ax[i].set_title(f"Patient {i + 1}")

fig.tight_layout()
# plt.savefig(path.join(outdir, 'rt_hist.png'), dpi=300) if savefig else print("Fig not saved")
plt.savefig(path.join(outdir, "rt_hist_scaled.png"), dpi=300) if savefig else print(
    "Fig not saved"
)
plt.close(fig)
# -

# # Looking at the patient data as heatmaps

# Create the colormap
colors = [(0, "lightblue"), (1, "red")]
colors = LinearSegmentedColormap.from_list("myc", colors)

# +
fig, ax = plt.subplots(nrows=2, sharey=True, sharex=True, figsize=(20, 10))
ax = ax.flatten()

for i in range(39):
    hmap = df  # this so I can change the df fast without changing the code
    hmap = hmap[hmap["groups"] == i]
    p0 = hmap[hmap["labels"] == 0]
    p1 = hmap[hmap["labels"] == 1]

    drop_cols = ["labels", "groups"]
    for col in hmap.columns:
        t = hmap.loc[:, col]
        if t.sum() == 0.0:
            drop_cols.append(col)

    p0 = p0.drop(columns=drop_cols)
    p1 = p1.drop(columns=drop_cols)

    sns.heatmap(p0, ax=ax[0], cmap=colors, cbar=False, square=False)
    sns.heatmap(p1, ax=ax[1], cmap=colors, cbar=False, square=False)

    ax[0].set_title(f"Patient {i + 1} - Label 0")
    ax[1].set_title(f"Patient {i + 1} - Label 1")

    fig.tight_layout()
    # plt.savefig(path.join(outdir, 'heatmaps', f'patient {i + 1} map.png'), dpi=300) if savefig else print("Fig not saved")
plt.close(fig)
