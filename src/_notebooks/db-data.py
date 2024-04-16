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

from modules import filestructure as fs
from submodules import utils as ut
import pandas as pd
import numpy as np
import pickle

# +
# importing database results as dataframes
with open(ut.join(fs.PklPath.lcms, "merged_df_raw.pkl"), "rb") as pickle_out:
    db: pd.DataFrame = pickle.load(pickle_out)

with open(ut.join(fs.PklPath.lcms, "all_db_dict.pkl"), "rb") as pickle_out:
    all_db: dict = pickle.load(pickle_out)

# with open(ut.join(fs.PklPath.lcms, "merged_df_group.pkl"), "rb") as pickle_out:
#     merged_df_group: pd.DataFrame = pickle.load(pickle_out)

print("Imported Pickled Data: Done")

# +
# read the AliquotIDs in LCMS
keys = pd.read_excel(ut.join(fs.PklPath.lcms, "sample-keys.xlsx"))
# renaming the cols
nums = list(range(9))
keys.columns = ["samples"] + [f"ID{num}" for num in nums]

# getting the AliquotIDs into a list
lcms_ali_ids = keys.loc[:, list(keys.columns)[1:]]
lcms_ali_ids = lcms_ali_ids.to_numpy().flatten()
lcms_ali_ids = lcms_ali_ids[~np.isnan(lcms_ali_ids)].astype(int)
lcms_ali_ids = np.unique(lcms_ali_ids)

print(f"The number of AliquotIDs in key.xlsx: {lcms_ali_ids.shape[0]}")

# +
# getting only the id and aliquot-id from the db
allids = db.loc[:, ["ID", "AliquotID"]]

found_ids = dict()

for aliquot_key in lcms_ali_ids:
    tmp_id = []
    tmp_ali = []
    tmp_idx = []
    for i, temp in enumerate(allids["AliquotID"]):
        if temp == aliquot_key:
            tmp_id.append(allids.loc[i, ["ID"]].to_list()[0])
            tmp_ali.append(allids.loc[i, ["AliquotID"]].to_list()[0])
            tmp_idx.append(i)

    tmp_id = list(set(tmp_id))
    tmp_ali = list(set(tmp_ali))
    tmp_idx = list(set(tmp_idx))

    if len(tmp_id) == 1:
        found_ids[tmp_id[0]] = {"AliquotID": tmp_ali, "indexes": tmp_idx}
    else:
        print("Warning: found several IDs related to one AliquotID")
        ut.dprint(tmp_id)
        found_ids[f"id{i}"] = {"ID": tmp_id, "AliquotID": tmp_ali, "indexes": tmp_idx}

print(f"Found {len(found_ids.keys())} matching IDs in the database")

# +
fltr = list()
for i in range(allids.shape[0]):
    if allids["ID"][i] in list(found_ids.keys()):
        fltr.append(True)
    else:
        fltr.append(False)

lcmsdb: pd.DataFrame = db.iloc[fltr].drop_duplicates(ignore_index=True)
lcmsdb

# +
# temp: find statistics about the data

print(lcmsdb.drop_duplicates("ID").Stage.value_counts())
# -

t = (
    lcmsdb.loc[:, ["ID", "TreatmentCourses"]]
    .dropna()
    .drop_duplicates(ignore_index=True)
)
t

# todo: need to resolve this problems with db. Some treatments defer, see below:
t[t.duplicated(subset="ID", keep=False)]

# finding mutation data in db
genes: pd.DataFrame = all_db["Genes"]
genes

# patients with gene mutation data + treatment information + lcms data
lcmsdb.loc[:, ["ID", "AKT1", "TreatmentCourses"]].dropna().drop_duplicates(subset="ID")
