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

# # 1 - Importing data into a dataframe

# +
import pandas as pd
import pickle
from varname import argname
import re
import orjson
from os.path import join
from src.modules.filestructure import PklPath, OutputPath

# Relation dictionary
with open(join(PklPath.data_importer, "df_dict.pkl"), "rb") as pickle_in:
    df_dict = pickle.load(pickle_in)

with open(join(PklPath.db, "pk_schema.pkl"), "rb") as pickle_in:
    pk_schema = pickle.load(pickle_in)

with open(join(PklPath.db, "db_schema.pkl"), "rb") as pickle_in:
    db_schema = pickle.load(pickle_in)

print("Done")
# -

# # 1.2 - Removing empty dataframes
# Already done that in file No. 0
# # 1.3 - Replacing NaN to 'null'
# you can do this at the end, on the result string, but you might get some unique result if a word has null in it

# +
temp_dict: dict = {}
# looping over all tables in the dictionary
for tableName in df_dict.keys():
    # creating a copy of the table to do changes on
    dfNAN = df_dict[tableName]
    # finding all null values and replacing them with the string "null"
    df = dfNAN.where(pd.notnull(dfNAN), "null")
    temp_dict.update({tableName: df})

# updating the dictionary
df_dict = temp_dict
del temp_dict, df, dfNAN

print("Done")


# -

# # 2 - Define functions


# +
#  separate the PK from the table df
def _is_blank(myString: str) -> bool:
    try:
        if type(myString) == tuple or type(myString) == list:
            myString = myString[0]
    except IndexError:
        return True
    if myString and myString.strip():
        return False
    return True


def sub_df(aDF: pd.DataFrame, colList: list) -> pd.DataFrame:
    """
    This function returns the sub columns from a dataframe based on the headers in colList.
    """
    dfList: list = []

    for aCol in colList:
        if aCol in list(aDF.columns):
            dfList.append(aDF[aCol].copy())
    return pd.concat(dfList, axis=1)


def get_pk(
    aTable: pd.DataFrame,
    aPK_df: pd.DataFrame,
    aTable_str: str = "",
    ignore_columns: list = None,
) -> pd.DataFrame:
    """
    :param ignore_columns: columns to ignore from the primary keys
    :param aTable: a dataframe of the table in question
    :param aTable_str: aTable but as a string (this is needed for operations in the function)
    :param aPK_df:
    :return: Two DataFrames, one is the primary keys of the database table and the other is without the primary keys
    """
    # Populate aTable_str if empty
    if _is_blank(aTable_str):
        aTable_str: str = argname("aTable")

    # 1. pick out the pk from aPK_df -> table_pk_df
    table_pk_df = aPK_df.groupby(["table_name"]).get_group(aTable_str)

    # 2. put all new_pk_df['pk'] in a list -> pk_list
    pk_list = list(table_pk_df["col_name"])
    # fixme: we are removing the 'EdgeCase' from the list, because it has default value of zero, but it is in the database schema. Need to fix this if I find an EdgeCase.
    if type(ignore_columns) != list and ignore_columns is not None:
        raise TypeError("ignore_columns must be a list of column names to ignore")
    elif ignore_columns is not None:
        for col in ignore_columns:
            try:
                del pk_list[pk_list.index(col)]
            except ValueError:
                pass

    # 3. separate the pks and non-pks from aTable
    aTable_pk_df = sub_df(aTable, pk_list)

    # 4. delete pk columns form aTable
    aTable_no_pk_df = aTable.drop(pk_list, axis=1)

    # 4. return both DFs
    return aTable_pk_df, aTable_no_pk_df


def getRealType(aHeader: str, aTable: str, type_df: pd.DataFrame) -> "Class":
    # categorizing the `type_df` based on relation names
    table_types = type_df.groupby(["table_name"]).get_group(aTable)

    # getting the header datatype as the one defined in the database
    realType = table_types[table_types["col_name"] == str(aHeader)]["data_type"]
    if type(realType) == list or type(realType) == tuple:
        realType = realType[0]
    realType = str(realType)

    isText = re.search(r"varchar", realType)
    isInt = re.search(r"int", realType)
    isFloat = re.search(r"decimal", realType)
    isJosn = re.search(r"json", realType)

    if isText:
        return str
    if isJosn:
        return dict
    elif isInt:
        return int
    elif isFloat:
        return float
    else:
        return None


print("Done")
# -

# # 3 - Creating the SQL code

# +
# TODO: uncomment if the file doesn't have these (they have default values in my sql)
ignore_columns = ["EdgeCase", "AliquotID", "GlycanSpecimenType"]
# ignore_columns = ['EdgeCase', 'AliquotID']

for tableName in list(df_dict.keys()):
    print(f"Working on {tableName} -> ", end="")
    aTable = df_dict[tableName]
    df_pk, df_no_pk = get_pk(
        aTable, pk_schema, tableName, ignore_columns=ignore_columns
    )
    headers = list(df_no_pk.columns)
    headers_pk = list(df_pk.columns)

    # loop over rows
    for irow in range(df_no_pk.shape[0]):
        preText: str = f"UPDATE HealthProject.{tableName} SET\n"
        postText: str = "\nWHERE "
        values: str = ""
        final_statement: str = ""

        # loop over non pk columns
        for icol in range(df_no_pk.shape[1]):
            dataPoint = df_no_pk.iloc[irow, icol]
            real_type = getRealType(headers[icol], tableName, db_schema)

            if dataPoint == "null":
                continue
                # values = values + f"{headers[icol]} = {dataPoint},\n"
            elif dataPoint != "null" and real_type == str:
                values = values + f"{headers[icol]} = '{dataPoint}',\n"
            elif real_type == int:
                values = values + f"{headers[icol]} = {str(int(dataPoint))},\n"
            elif real_type == float:
                values = values + f"{headers[icol]} = {str(float(dataPoint))},\n"
            elif real_type == dict:
                if len(dataPoint.keys()) > 0:
                    values = (
                        values
                        + f"{headers[icol]} = JSON_MERGE_PATCH(IFNULL({headers[icol]},'{{}}'), '{orjson.dumps(dataPoint).decode()}'),\n"
                    )
                else:
                    continue
            else:
                raise Exception(
                    f"Data type unknown\n"
                    f"row num: {irow} ; col num (no pk): {icol}\n"
                    f"data point: {dataPoint}\n"
                    f"real type: {real_type}"
                )

        # testing: if there were nothing to add, we skip that row
        if _is_blank(values):
            continue
        else:
            final_statement = preText + values[:-2] + postText
            values: str = ""

        # loop over pk columns
        for icol in range(df_pk.shape[1]):
            values: str = values + headers_pk[icol] + " = "
            dataPoint = df_pk.iloc[irow, icol]
            real_type = getRealType(headers_pk[icol], tableName, pk_schema)

            if dataPoint == "null":
                values = values + dataPoint + " AND "
            elif dataPoint != "null" and real_type == str:
                values = values + "'" + dataPoint + "' AND "
            elif real_type == int:
                values = values + str(int(dataPoint)) + " AND "
            elif real_type == float:
                values = values + str(float(dataPoint)) + " AND "
            else:
                raise Exception(
                    f"Primary Key data type unknown\n"
                    f"row num: {irow} ; col num (pk): {icol}\n"
                    f"data point: {dataPoint}\n"
                    f"real type: {real_type}"
                )

        final_statement = final_statement + values[:-5] + ";\n\n"

        # Export code to file
        with open(join(OutputPath.sql, f"{tableName}_4_update.sql"), "a") as file:
            file.write(final_statement)
    print("Done")

print("Finished")
