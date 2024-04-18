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

# # Mapping missed Aliquot IDs to FWIDs

# +
import pandas as pd

# nums = [1, 2, 3, 4, 5, 6]
# dfs = []
# for i in nums:
#     base = pd.read_excel(f"../../../RawFiles/AliquotID_ref/{i}.xlsx",
#                        index_col=None,
#                        header=0,
#                        na_values=("NA", "N/A", "na", "n/a", "NULL", "null", "Not documented", "Not Documented", 'nan', 'Unknown', 'unknown'))
#     base.rename(columns={
#     'Tissue Site ' : 'Site',
#     'Specimen Type ' : 'Specimen Type',
#     'Stage ' : 'Stage'
#     }, inplace=True)
#     dfs.append(base)
#
# df = pd.concat(dfs)
# df.drop_duplicates(keep='first', inplace=True)
# df.drop(columns=['Specimen Type.1'], inplace=True)

df = pd.read_excel(
    f"../../../RawFiles/AliquotID_ref/{i}.xlsx",
    index_col=None,
    header=0,
    na_values=(
        "NA",
        "N/A",
        "na",
        "n/a",
        "NULL",
        "null",
        "Not documented",
        "Not Documented",
        "nan",
        "Unknown",
        "unknown",
    ),
)
df_dict = df.groupby("FWID")

# df.to_excel('../../../RawFiles/AliquotID_ref/combined.xlsx')

# +
from src.modules.filestructure import GlobVar
from os.path import join

file_6 = pd.read_excel(
    join(GlobVar.raw_data_path, "6.Galectins.xlsx"),
    index_col=None,
    header=0,
    na_values=(
        "NA",
        "N/A",
        "na",
        "n/a",
        "NULL",
        "null",
        "Not documented",
        "Not Documented",
        "nan",
    ),
)
# -

# this cell showing that out of 153 IDs in query, for all 36 IDs in the query that map to several Aliquot IDs in the base, have the same entries for Site, Spec Considered, Sample Considered. Should we just choose one Aliquot ID by random?
count = 0
for id in file_6["FWID"]:
    if id in df_dict.groups and len(temp := df_dict.get_group(id)) > 1:
        if len(set(temp["Sample Considered"].values)) > 1:
            count += 1
count

# +
# TODO: do same thing for other files 5, 4, 3
file_6_id = {}
count = 0
for id in file_6["FWID"]:
    if id in df_dict.groups:
        count += 1
        file_6_id[id] = set(df_dict.get_group(id)["Unique Aliquot ID"].values)

print(
    f"A number of {len(file_6) - count} out of {len(file_6)} IDs are yet to be identified"
)
file_6_id
# -

# temp
print("test output, should be cleared. Remove this line after testing")
