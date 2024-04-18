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

# # Importing the data from a dataframe and mySQL DB

import pandas as pd
import pickle
import re
from glob import glob
from os.path import join
from src.modules.filestructure import PklPath, RunLib

# # 1 - Importing database results as dataframes

# +
try:
    RunLib  # this is just so that the import don't get removed
    # %run $RunLib.db_query_py
except Exception:
    print(
        "Error: {e}\nFalling back to previously created pickle files.\nCheck ReadMe_history.txt for more information."
    )

with open(join(PklPath.db, "db_schema.pkl"), "rb") as pickle_in:
    db_schema: pd.DataFrame = pickle.load(pickle_in)

with open(join(PklPath.db, "pk_schema.pkl"), "rb") as pickle_in:
    pk_schema: pd.DataFrame = pickle.load(pickle_in)

# Getting the primary keys entries for all relations in the database. This is done to check and not insert a duplicate entry.
with open(join(PklPath.db, "base_df.pkl"), "rb") as pickle_in:
    base_df: pd.DataFrame = pickle.load(pickle_in)

# Original df
with open(join(PklPath.data_importer, "df.pkl"), "rb") as pickle_in:
    df = pickle.load(pickle_in)

# Relation dictionary
with open(join(PklPath.data_importer, "df_dict.pkl"), "rb") as pickle_in:
    df_dict = pickle.load(pickle_in)

print("Done")


# -

# # 2 - Comparing Primary Keys


# +
def comparePK(
    aBase_df: pd.DataFrame,
    aDict_df: pd.DataFrame,
    aPK_df: pd.DataFrame,
    ignore_columns: list = None,
) -> "insert_PK:dict , update_PK:dict":
    """
    This method takes in the old data (present in the database, called base_df), new data (dictionary result of "0. editing Excel values"), and the primary keys dataframe for all tables.
    The result will be two dictionaries where their keys are "table names" and their values are dataframes of new primary keys that need to be added to the database or updated.

    :param ignore_columns: A list of column names to ignore when comparing the primary keys
    :param aBase_df: The data acquired from the database
    :type aBase_df: Pandas DataFrame
    :param aDict_df: The new data that was passed from previous files
    :type aDict_df: Pandas DataFrame
    :param aPK_df: The primary key information of the database
    :type aPK_df: Pandas DataFrame
    :return: Two dictionaries (the insert primary keys, and the update primary keys)
    """

    # A list of all db table names
    tableNames = set(aDict_df.keys())
    run_PersonInfo_check = False
    add_PI_to_dict = False
    run_SampleInfo_check = False
    add_SI_to_dict = False

    # making sure PersonInfo (ID) and SampleInfo (ID and AliqoutID) are either in dDict_df or if not, checked for new primary keys
    if "SampleInfo" not in tableNames and aBase_df is not None:
        run_SampleInfo_check = True
        # TODO: create a function that would return all PKs for a table so we don't clutter this function
        sample_pks = (
            base_df.loc[:, ("SampleInfo_ID", "SampleInfo_AliquotID")]
            .copy()
            .dropna()
            .applymap(int)
            .rename({"SampleInfo_ID": "ID", "SampleInfo_AliquotID": "AliquotID"})
        )
        insert_SampleInfo_series = pd.DataFrame()
        if sample_pks.empty:
            run_SampleInfo_check = False

    if "PersonInfo" not in tableNames and aBase_df is not None:
        run_PersonInfo_check = True
        person_pks = (
            base_df.loc[:, "PersonInfo_ID"].copy().dropna().map(int).rename("ID")
        )
        insert_PersonInfo_series = pd.Series(dtype="float64")
        if person_pks.empty:
            run_PersonInfo_check = False

    # initiating the returned values
    insert_pk: dict = {}
    update_pk: dict = {}

    # Checking the primary keys of each table to itself
    # 0. skipping the check if aBase is None (the database is empty)
    if aBase_df is None:
        # Then all the primary keys are new and must add them all
        # 1. Looping over all tables
        for aTable in tableNames:
            # 1.1 list the primary keys of aTable
            aTable_pk = list(aPK_df[aPK_df["table_name"] == aTable]["col_name"])

            # making sure that ignore_columns is a list if not None
            if type(ignore_columns) != list and ignore_columns is not None:
                raise TypeError(
                    "ignore_columns must be a list of column names to ignore when comparing the primary keys"
                )
            elif ignore_columns is not None:
                aTable_pk = list(set(aTable_pk) - set(ignore_columns))

            # 2. get the primary keys of that table from df_dict (skipping if a tabel only has the primary keys)
            if set(aDict_df[aTable].columns) - set(aTable_pk) != set():
                insert_df = aDict_df[aTable].loc[:, aTable_pk]

                # resetting the dataframe index, just in case
                insert_df.reset_index(inplace=True)
                insert_df.drop(labels="index", axis=1, inplace=True)

                # 5. Saving the new PKs in a dict
                insert_pk.update({aTable: insert_df})
            else:
                continue
    else:
        # 1. Looping over all tables
        for aTable in tableNames:
            # 1.1 list the primary keys of aTable
            aTable_pk = list(aPK_df[aPK_df["table_name"] == aTable]["col_name"])

            # making sure that ignore_columns is a list if not None
            if type(ignore_columns) != list and ignore_columns is not None:
                raise TypeError(
                    "ignore_columns must be a list of column names to ignore when comparing the primary keys"
                )
            # removing the columns to ignore from the primary keys
            elif ignore_columns is not None:
                aTable_pk = list(set(aTable_pk) - set(ignore_columns))

            # 2. get the primary keys of that table from df_dict (skipping if a tabel only has the primary keys)
            if set(aDict_df[aTable].columns) - set(aTable_pk) != set():
                query_pk = aDict_df[aTable].loc[:, aTable_pk]
            else:
                continue

            # 3. get the primary key entries from base_df
            ## creating column names to match base_pk (like tableName_columnName)
            base_pk_names: list = [str(f"{aTable}_{element}") for element in aTable_pk]
            ## getting the primary key rows that are related to aTable and already in the database
            base_pk = aBase_df.loc[:, base_pk_names].dropna(subset=[(aTable + "_ID")])

            # Renaming base_pk columns to match query_pk and changing the data type
            # TODO: see if you can use the function map() or applymap() instead of looping
            for i, element in enumerate(aTable_pk):
                base_pk.rename(inplace=True, columns={base_pk_names[i]: element})
                # changing the data type to int
                PK_dataType: str = str(
                    list(aPK_df[aPK_df["col_name"] == element]["data_type"])
                )
                isText = re.search("varchar*", PK_dataType)
                if isText:
                    pass
                else:
                    base_pk[element] = base_pk[element].astype(int)

            # 4. compare primary keys (Merge DataFrames with indicator)
            result_df = base_pk.merge(query_pk, indicator=True, how="outer")
            insert_df = result_df[result_df["_merge"] == "right_only"].drop(
                columns=["_merge"]
            )
            update_df = result_df[result_df["_merge"] == "both"].drop(
                columns=["_merge"]
            )

            # resetting the dataframe index, just in case
            insert_df.reset_index(inplace=True)
            insert_df.drop(labels="index", axis=1, inplace=True)

            # 5. Saving the new PKs in a dict
            insert_pk.update({aTable: insert_df})

            if len(update_df) > 0:
                # resetting the dataframe index, just in case, before adding the dataframe
                update_df.reset_index(inplace=True)
                update_df.drop(labels="index", axis=1, inplace=True)
                update_pk.update({aTable: update_df})

            # 6. check if we need to insert any new IDs in PersonInfo or SampleInfo
            if run_PersonInfo_check:
                add_PI_to_dict = True
                temp_df = pd.merge(
                    person_pks,
                    query_pk.loc[:, "ID"],
                    on="ID",
                    indicator=True,
                    how="outer",
                )
                insert_PersonInfo_df = temp_df[temp_df["_merge"] == "right_only"].drop(
                    columns=["_merge"]
                )
                insert_PersonInfo_df.reset_index(inplace=True)
                insert_PersonInfo_df.drop(labels="index", axis=1, inplace=True)
                insert_PersonInfo_series = pd.concat(
                    [insert_PersonInfo_series, insert_PersonInfo_df]
                )

            if run_SampleInfo_check and "AliquotID" in list(query_pk.columns):
                add_SI_to_dict = True
                temp_df = sample_pks.merge(
                    query_pk.loc[:, ("ID", "AliquotID")], indicator=True, how="outer"
                )
                insert_SampleInfo_df = temp_df[temp_df["_merge"] == "right_only"].drop(
                    columns=["_merge"]
                )
                insert_SampleInfo_df.reset_index(inplace=True)
                insert_SampleInfo_df.drop(labels="index", axis=1, inplace=True)
                insert_SampleInfo_series = pd.concat(
                    [insert_SampleInfo_series, insert_SampleInfo_df]
                )

        # adding PersonalInfo and SampleInfo to the insert dictionary
        if run_PersonInfo_check and len(insert_PersonInfo_series) > 0:
            insert_PersonInfo_series = (
                insert_PersonInfo_series.drop(columns=0)
                .drop_duplicates()
                .reset_index()
                .drop(columns="index")
                .astype(int)
            )
        if run_SampleInfo_check and len(insert_SampleInfo_series) > 0:
            insert_SampleInfo_series = (
                insert_SampleInfo_series.drop_duplicates()
                .reset_index()
                .drop(columns="index")
                .astype(int)
            )  # fixme: need to make sure this is correct !
        if add_PI_to_dict and len(insert_PersonInfo_series) > 0:
            # insert_PersonInfo_series.drop(labels='index', axis=1, inplace=True)
            insert_pk.update({"PersonInfo": insert_PersonInfo_series})
        if add_SI_to_dict and len(insert_SampleInfo_series):
            insert_pk.update({"SampleInfo": insert_SampleInfo_series})

    return insert_pk, update_pk


print("Done")

# +
insert_PK, update_PK = comparePK(
    base_df,
    df_dict,
    pk_schema,
    ignore_columns=["EdgeCase", "GlycanSpecimenType", "AliquotID"],
)

print("Done")

# +
# fixme: need to make sure this is correct if you have AliquotIDs. look for "fixme" in the "comparePK()" function!
# insert_PK['SampleInfo']
# -

# # 6 - Exporting Data

# +
overwrite = True
for name in ["2_insertPK", "2_updatePK"]:
    test = glob(join(PklPath.data_importer, f"{name}.pkl"))
    if test:
        overwrite = input(f"{name} exists, Overwrite? (y/n)").lower() == "y"
        if not overwrite:
            break
        # todo: make it check each one independently

if overwrite:
    with open(join(PklPath.data_importer, "2_insertPK.pkl"), "wb") as pickle_out:
        pickle.dump(insert_PK, pickle_out)

    with open(join(PklPath.data_importer, "2_updatePK.pkl"), "wb") as pickle_out:
        pickle.dump(update_PK, pickle_out)

else:
    raise Exception("Allow overwriting, or rename")

print("Done")
