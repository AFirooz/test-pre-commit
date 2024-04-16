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

# # 1 - Importing Data

# +
import pickle
import pandas as pd
from varname import argname
from os.path import join
from src.modules.filestructure import PklPath, OutputPath

with open(join(PklPath.data_importer, "2_insertPK.pkl"), "rb") as pickle_in:
    insertPK: dict = pickle.load(pickle_in)

print("Done")


# -

# # 2 - Creating Insert for New Primary Keys
# Even if you have more information to insert to PersonalInfo, add the IDs first and then create update statements
# ## 2.1 - Person Info
# Adding the new IDs to PersonalInfo

# +
def insertPK_statement(
    aTable_pk: pd.DataFrame,
    aTable_pk_str: str = None,
    file_name: str = None,
    export: bool = False,
) -> str:
    """
    Generates a SQL statement for inserting the IDs into a PersonInfo table.
    The time complexity of this function is O(nm) where n is the number of rows and m is the number of columns. Since m is limited (constant), so we can just say O(n). Every other operation in this function is also O(n).
    :param file_name: Specify what name should the file be exported to
    :param aTable_pk: The dataframe having all primary key entries.
    :type aTable_pk: pd.DataFrame

    :param aTable_pk_str: The name of the table to insert the data into. If not provided, the function will try to determine the name of the DataFrame.
    :type aTable_pk_str: str, optional

    :param export: Determines whether the SQL statement should be written to a file or returned as the output of the function.
    :type export: bool, optional

    :return: The generated SQL statement.
    :rtype: str
    """

    # Populate aTable_pk_str if empty
    if aTable_pk_str is None or type(aTable_pk_str) != str:
        aTable_pk_str: str = argname("aTable_pk")

    if file_name is None or type(file_name) != str:
        file_name = aTable_pk_str

    # 1- create the statement for all pk in the df
    startStatement: str = "insert into HealthProject." + aTable_pk_str + " "
    intermediateState: str = " values" + "\n" + "\t"
    resultStatement: str = (
        startStatement
        + str(list(aTable_pk.columns))
        .replace("[", "(", 1)
        .replace("]", ")", 1)
        .replace("'", "")
        + intermediateState
    )

    # 2- create the values
    for index in range(aTable_pk.shape[0]):
        resultStatement: str = resultStatement + "("
        for colData in range(aTable_pk.shape[1]):
            if type(aTable_pk.iloc[index][colData]) == str:
                resultStatement: str = (
                    resultStatement + "'" + aTable_pk.iloc[index][colData] + "'"
                )
            else:
                resultStatement: str = resultStatement + str(
                    aTable_pk.iloc[index][colData]
                ).replace("'", "")
            if colData != (aTable_pk.shape[1] - 1):
                resultStatement: str = resultStatement + ", "

        resultStatement: str = resultStatement + "), \n\t"

    resultStatement = resultStatement[:-4] + ";"

    if export:
        with open(
            join(OutputPath.sql, f"{file_name}_3_primary_keys.sql"), "w"
        ) as txtFile:
            txtFile.write(resultStatement)
    else:
        return resultStatement


print("Done")

# +
for key in insertPK:
    df = insertPK[key]
    insertPK_statement(aTable_pk=df, aTable_pk_str=key, export=True)

print("Done")
