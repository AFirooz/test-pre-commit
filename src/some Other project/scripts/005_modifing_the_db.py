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
from glob import glob
from os import remove
import pickle
from os.path import join
from src.modules.filestructure import PklPath, OutputPath, RunLib, RunDataImporter

RunLib, RunDataImporter
# -

# # Backing up the Database

# %run $RunLib.db_backup

# +
# Clear files (this deletes files on the system)
pkl_files = glob(f"{PklPath.data_importer}/*.pkl")
db_files = glob(f"{PklPath.db}/*.pkl")
sql_files = glob(f"{OutputPath.sql}/*.sql")

if pkl_files or sql_files:
    delete_files = (
        input(
            f"Are you sure you want to delete files in\n{PklPath.data_importer} and \n{OutputPath.sql} ? (y/n)"
        ).lower()
        == "y"
    )
    if not delete_files:
        raise Exception("Your files were not deleted")
else:
    print("No files were found")

for item in [pkl_files, db_files, sql_files]:
    if item and delete_files:
        try:
            for temp in item:
                remove(temp)
        except Exception as e:
            print(e)
# -

# # Adding pk to tables
# We will just run 4_Pk_PersonInfo first so that the foreign key constraint will be fulfilled

# %run $RunDataImporter.file1

# %run $RunDataImporter.file2

# %run $RunDataImporter.file3

# %run $RunDataImporter.file4

# +
# TODO make sql code updates from within here. I need to find a way to deal with problems and resuming updates
# TODO: make all updates in a single statement for the sake of speed
# -

# # removing rows that only have primary keys

# +
try:
    RunLib  # this is just so that the import statement doesn't get removed
    # %run $RunLib.db_query
except Exception:
    print(
        "Error:\nFalling back to previously created pickle files.\nCheck ReadMe_history.txt for more information."
    )

# Relation dictionary
with open(join(PklPath.db, "pk_schema.pkl"), "rb") as pickle_in:
    pk_schema = pickle.load(pickle_in)

with open(join(PklPath.db, "db_schema.pkl"), "rb") as pickle_in:
    db_schema = pickle.load(pickle_in)

# +
# Only run this if the db schema changed, otherwise use the one in "Do Not Delete" folder

# ignore_col = set(['AnalysisType', 'GalectinSpecimenType'])
# for tableName in set(pk_schema.loc[:,'table_name']):
#     pk_col = set(pk_schema[pk_schema['table_name'] == tableName].loc[:,'col_name'])
#     col = set(db_schema[db_schema['table_name'] == tableName].loc[:,'col_name'])
#     null_col = col - pk_col
#
#     preText: str = f"DELETE FROM HealthProject.{tableName} WHERE\n"
#     values:str = ""
#
#     for header in null_col:
#         values = values + f'{header} IS null AND \n'
#
#     final_statement = f'{preText}{values[:-6]};\n\n'
#
#     with open (f'../../outputData/5_removeNull.sql', 'a') as file:
#         file.write(final_statement)

