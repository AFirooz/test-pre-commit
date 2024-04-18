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
#     display_name: Python 3.9.5 64-bit
#     language: python
#     name: python3
# ---

# # Getting DB information
# ## Getting all tables, all columns data types

# +
# TODO: put all functions in a file and just import it !
# TODO: Use os.path to check and not override saving things

import pandas as pd
import numpy as np
from varname import argname, nameof
import pickle
import re
from typing import Type
import orjson as json
from glob import glob
from os.path import join
import os.path as path

import filestructure as fs
import importer as imp
from filestructure import PklPath, RunLib
from importer import sub_df

# +
# importing database results as dataframes
try:
    RunLib  # this is just so that the import statement doesn't get removed
    # %run $RunLib.db_query_py
except Exception:
    print(
        "Error:\nFalling back to previously created pickle files.\nCheck ReadMe_history.txt for more information."
    )

with open(join(PklPath.db, "db_schema.pkl"), "rb") as pickle_in:
    db_schema: pd.DataFrame = pickle.load(pickle_in)

with open(join(PklPath.db, "pk_schema.pkl"), "rb") as pickle_in:
    pk_schema: pd.DataFrame = pickle.load(pickle_in)


# +
def slice_df(
    aDF: pd.DataFrame, aTable: str, aSchema_df: pd.DataFrame = db_schema
) -> pd.DataFrame:
    """
    A function that takes a name of a table, retrieve its columns from the database schema, and test each column name for existence in the cluttered DataFrame. When found, it separates it and finally, it will join all columns together and returns them as a one DataFrame.
    :param aDF: A disordered / cluttered DataFrame
    :param aTable: A table name in as it appears in the database
    :param aSchema_df: A database schema as a DataFrame
    :return: A Pandas DataFrame
    """

    # making sure aSchema_df is not empty
    if db_schema.empty:
        raise Exception("The database schema DataFrame is empty")

    colList: list = list(aSchema_df[aSchema_df["table_name"] == aTable]["col_name"])
    dfList: list = []

    for aCol in colList:
        if aCol in list(aDF.columns):
            dfList.append(aDF[aCol].copy())
    return pd.concat(dfList, axis=1)


def getPrimaryKey(
    aTable: str,
    aPK_df: pd.DataFrame = pk_schema,
    ignore_columns: list = None,
    include_columns: list = None,
) -> list:
    """
    A function that takes a table name (as it appears in the database) as a string, and optionally the database primary keys schema as a DataFrame, and returns a list of the table's primary keys.
    :param include_columns: Columns to include in the primary key list
    :param ignore_columns: A list of columns to ignore
    :param aTable: A table name in as it appears in the database
    :param aPK_df: A database primary keys schema as a DataFrame
    :return: list of primary keys
    """

    # making sure that aTable is a table name, so later we can use eval() with lower security risk
    if aTable not in set(db_schema["table_name"]):
        raise Exception(f"Enter a valid table name, {aTable} is not valid!")

    # if the table don't have any data other than the primary keys, we just return something basic, so we don't break the code.
    # we are assuming that if we have table with data, it will have at least 3 columns, including the primary key.
    if len(eval(aTable).columns) < 2:
        return ["ID"]

    # getting the primary key
    # if aPK_df is pk_schema and pk_schema.empty:
    if type(aPK_df) == pd.DataFrame and aPK_df.empty:
        raise Exception("There is no primary key found in the database")
    elif type(aPK_df) != pd.DataFrame:
        raise Exception("Enter a valid DataFrame for the primary keys schema")
    else:
        pk_list = list(aPK_df[aPK_df["table_name"] == aTable]["col_name"])

        # making sure that ignore_columns is a list if not None
        if type(ignore_columns) != list and ignore_columns is not None:
            raise TypeError("ignore_columns must be a list of column names to ignore")
        elif ignore_columns is not None:
            pk_list = list(set(pk_list) - set(ignore_columns))

        if type(include_columns) != list and include_columns is not None:
            raise TypeError("include_columns must be a list of column names")
        elif include_columns is not None:
            pk_list = pk_list + include_columns

        return pk_list


def print_pretty_json(data: str = None) -> None:
    """
    Prints pretty JSON values
    :param data: a json / str
    :return: None
    """
    if len(data) == 0 or data is None:
        print("JSON values are empty!")
        return
    data = json.loads(data)
    print(json.dumps(data, option=json.OPT_INDENT_2).decode())


def str_to_dict(
    data: str,
    data_json_name: str = "data",
    include_count: bool = True,
    keep_original: bool = False,
    cast_data: Type = None,
) -> str:
    """
    This function is used to create a dictionary of the data then it will be converted to JSON values and returned as a string.
    Case Usage:
    1. This function will split the gene mutations, put them in an array, and assign the key 'mutations' to them. Also, it will count them and add a 'count' key.
    2. In Pathways, it will just take the data, convert to json and return
    :param cast_data: cast the data to a specific type
    :param keep_original: To keep the original data and not cast it to a string
    :param data_json_name: The name of key to save the data into
    :param include_count: If 'count' key should be included in the json value
    :param return_bytes: To specify the return value type
    :param data: Data to parse into json
    :type data: String
    :return: Either a string or bytes of the result json value
    """
    if data is None or pd.isna(data):
        return np.NaN
    if keep_original:
        if cast_data is not None:
            data = cast_data(data)
        data_blocks = data
    else:
        data_blocks = str(data).replace(" ", "").split(",")
        if cast_data is not None:
            for i, block in enumerate(data_blocks):
                data_blocks[i] = cast_data(block)

    if include_count:
        # result = json.dumps({str(data_json_name): data_blocks, 'count':len(data_blocks)}
        return {str(data_json_name): data_blocks, "count": len(data_blocks)}
    else:
        # result = json.dumps({str(data_json_name): data_blocks})
        return {str(data_json_name): data_blocks}


def _clean_nulls_helper(value: dict or list) -> dict or [dict]:
    """
    Recursively remove all None values from dictionaries and lists, and returns
    the result as a new dictionary or list.
    source: https://stackoverflow.com/questions/4255400/exclude-empty-null-values-from-json-serialization
    """
    if isinstance(value, list):
        return [_clean_nulls_helper(x) for x in value if x is not None]
    elif isinstance(value, dict):
        return {
            key: _clean_nulls_helper(val)
            for key, val in value.items()
            if val is not None
        }
    else:
        return value


def clean_nulls(data: dict or str) -> dict:
    """
    Remove all None values from dictionaries and lists, and returns
    the result as a new dictionary or list.
    """
    if data is None:
        raise ValueError("data must not be None")
    if type(data) is str:
        result = _clean_nulls_helper(json.loads(data))
    else:
        result = _clean_nulls_helper(data)
        # if you want the return to be a string, uncomment the following line
        # result = json.dumps(result).decode()
    return result


#  This function can find duplicates, return them, and drop them if needed.
# TODO: add functionality to the function to create a df['EdgeCase']=np.ones(len(subtracted_d)) when told to. This is helpful for Genes table. Let's worry about it when we need to.
def dropDup(
    aTable: str,
    drop_full_dup=True,
    drop_pk_dup=False,
    return_dup=False,
    keep_dup=False,
    ignore_columns: list = None,
    include_columns: list = None,
) -> pd.DataFrame:
    """
    A function to check for duplicate entries in a database using the primary keys of that table as a subset. You pass it the table name you are interested in checking and specify any additional parameters, or just keep them as default.
    :param include_columns: columns to include in the check for duplicates
    :param ignore_columns: A list of columns to ignore when checking for duplicates
    :param aTable: Table name
    :param drop_pk_dup: Drop rows that have duplicated primary keys
    :param drop_full_dup: Drop rows that are fully duplicated
    :param return_dup: return the duplicated rows found
    :param keep_dup: This is Pandas.duplicated() argument, meaning which duplicated rows to keep.
    :return: A DataFrame with (or without) duplicate rows
    """

    # making sure that aTable is a table name, so later we can use eval() with lower security risk
    if aTable not in set(db_schema["table_name"]):
        raise Exception(f"Enter a valid table name, {aTable} is not valid!")

    # if the table don't have enough data, we just return something basic, so we don't break the code. These tables will be eliminated in following steps.
    if len(eval(aTable).columns) == 1:
        return eval(aTable)

    # if a table have json type values as dictionary, we need to convert them to a hashable data structure first (like tuples), then drop duplicates
    # Notice that we don't need to convert them back to a dictionary since we either notice duplicates in primary keys and rais an exception, or choose to drop primary key duplicates, and then we wouldn't need to check dictionary values.
    aTable_hashed = eval(aTable).copy()
    json_cols = set(db_schema.groupby("data_type").get_group("json")["col_name"])
    for col in aTable_hashed.columns:
        if col in json_cols:
            aTable_hashed.loc[:, col] = aTable_hashed.loc[:, col].map(
                lambda x: json.dumps(x), na_action="ignore"
            )

    # getting primary keys
    temp_keys: list = getPrimaryKey(
        aTable, ignore_columns=ignore_columns, include_columns=include_columns
    )

    # Either dropping rows that are fully duplicate and then check for duplicate primary key rows, or just skipping the first step
    # if we plan on dropping duplicated primary key rows, there is no need to drop fully duplicated row since we will eventually do it.
    if not drop_pk_dup:  # drop_pk_dup == False
        if drop_full_dup:
            # dropping fully duplicated rows
            temp_df = aTable_hashed.drop_duplicates(ignore_index=True, keep="first")
            # checking for duplicate primary keys
            temp_dup_df = temp_df[temp_df.duplicated(subset=temp_keys, keep=keep_dup)]
        else:
            # if we don't want to drop fully duplicated rows, we'll just check for duplicate primary keys
            temp_dup_df = eval(aTable)[
                eval(aTable).duplicated(subset=temp_keys, keep=keep_dup)
            ]

    # notifying the user of duplicate key existence
    # notice that if we are going to drop duplicated primary keys, then we don't need to raise an error about them
    if len(temp_dup_df) > 0 and not drop_pk_dup:  # drop_pk_dup == False
        if return_dup:
            return temp_dup_df  # if you don't want the dictionaries in a table, you will need to convert them back to a dictionary
        display(temp_dup_df)
        raise Exception(
            f"You have duplicated primary keys entry in table {aTable}.\nYou can start by returning the duplicated rows and maybe set drop_duplicates argument to True"
        )

    # if no duplicates, or if we choose to drop them we run this
    else:  # same as saying (elif drop_pk_dup or len(temp_dup_df) == 0:)
        try:
            return eval(aTable).drop_duplicates(
                subset=temp_keys, ignore_index=True, keep="first"
            )
        except KeyError:
            print(f"Table {aTable} -> Error: Primary key was not found")


def _is_blank(myString: str) -> bool:
    """
    A helping function to check if a string given as an input is empty or blank. Also, there is a check to make sure if the input was a tuple or a list of one string, it would take that string and do not return false.
    Note that empty ("") and blank ("  ") strings and `None` objects are different things, but all mean that the string is not there or empty. Hence,, the use of `str.strip()` method.
    Inspired by stackoverflow.com discussion:
    https://stackoverflow.com/questions/9573244/how-to-check-if-the-string-is-empty

    :param myString: A string to check if it is empty or blank
    :return: boolean value
    """
    try:
        # just in case myString wasn't a string
        if type(myString) == tuple or type(myString) == list:
            myString = myString[0]
    except IndexError:
        return True

    # checking if myString is empty
    if myString and myString.strip():
        return False
    return True


def check_dtype(
    aTable: pd.DataFrame, aHeader: str, dataType: Type, ignoreDataType: Type = None
) -> None:
    """
    This function accepts a dataframe and a header name along with the expected data type that should be stored in that column, and optionaly data types that should be ignored. The function will loop over each row and examine the aHeader data and make sure it matched the type given. If in case it didn't, the function will save that data in another dataframe along with its position index (not index name) and type. The function will finally check if there is any entry that didn't match, and will rais an exception requiring the user the fix it, otherwise, the function will just exit.

    :param aTable: The DataFrame containing the column to check.
    :param aHeader: The name of the column to check.
    :param dataType: The expected data type of the column.
    :param ignoreDataType: A data type to ignore when checking the column, defaults to None.

    :raises Exception: If the data type of any data in the column does not match the specified data type or is not of the specified ignored data type.
    """
    # a temp dataframe to store wrong data that have wrong datatype
    temp_df = pd.DataFrame({"rowNum": [], "WrongType": [], "ActualValue": []})

    # looping over all the data and checking their types one by one
    for i, data in enumerate(aTable[aHeader]):
        # we saved it in a temp variable so that if need be we use regular expression to find the truth
        temp = type(data)

        # if the data type don't match what we are looking for, or what we should ignore, that data gets appended to the temp_df. Otherwise, we just move on to check the next row.
        if temp != dataType and str(data) != "nan":
            if ignoreDataType != None and temp == ignoreDataType:
                continue
            temp_df.loc[len(temp_df)] = [i, temp.__name__, data]

    # return an error if we found a problem
    if len(temp_df) != 0:
        otherTypes = set(temp_df["WrongType"])
        print(
            f"Not valid types found in the {aHeader} header -> {otherTypes} \nShould've been {dataType}"
        )
        print("------------------------------------------")
        print(f"The problem rows was found in: \n{temp_df}")
        raise Exception(f"You need to fix all rows in {aHeader} to match {dataType}")


def verify_dtype(
    aTable: pd.DataFrame,
    aDB_schema: pd.DataFrame,
    aTable_str: str = "",
    ignoreHeaders: tuple = None,
) -> None:
    # this function is very similar to `getRealType()` in file no. 5
    """
    This function takes in a table (dataframe) to loop through its headers and grab each one of them, then it grabs datat type of that same header from the database schema table (dataframe), and compare all the columns data types with the database's one. If you have columns that you know will not be in the database directly, like columns that will be used to create JSON values, you can add them to the ignore list.

    :param ignoreHeaders: A tuple of headers names that you want to ignore when checking and verifying the types. This is mostly used for data that will be converted to JSON later on.
    :type ignoreHeaders: tuple

    :param aTable: The DataFrame to verify the column data types for.
    :type aTable: pd.DataFrame

    :param aDB_schema: The DataFrame containing the column data types to compare against.
    :type aDB_schema: pd.DataFrame

    :param aTable_str: The name of the table in the aDB_schema DataFrame to compare against. If not provided, it will try to infer the name from the variable name of aTable.
    :type aTable_str: str, optional

    :return: A dictionary containing the column names and their verified data types.
    :rtype: dict

    :raises Exception: If the function is unable to match a column name to any column in the aDB_schema DataFrame or if the data type of a column in aTable does not match the corresponding column in aDB_schema.
    """
    print(f"Table {aTable_str} -> ", end="")

    # making sure that "ignoreHeaders" is a list
    if ignoreHeaders is None or len(ignoreHeaders) == 0:
        ignoreHeaders = ()

    # Populate aTable_str if empty
    if _is_blank(aTable_str):
        aTable_str: str = argname("aTable")

    # Getting the types of data stored in aTable columns
    aTable_types = aDB_schema.groupby(["table_name"]).get_group(aTable_str)

    # aTable's header names
    headers: list = list(aTable.columns)
    headersType: dict = {}
    for aHeader in headers:
        realType = aTable_types[aTable_types["col_name"] == str(aHeader)]["data_type"]

        # just in case we get a tuple or a list with one item, we grab that one item and don't break the code
        if type(realType) == list or type(realType) == tuple:
            realType = realType[0]
        realType = str(realType)

        # using RE to find out if it is text, number, or float
        # Since we already converted relevant columns to JSON, we need to figure out how to check the data in them
        isText = re.search(r"varchar", realType)
        isInt = re.search(r"int", realType)
        isFloat = re.search(r"decimal", realType)
        isJson = re.search(r"json", realType)

        if isText:
            check_dtype(aTable, aHeader, str)
            # headersType.update({str(aHeader) : str})
        elif isInt:
            check_dtype(aTable, aHeader, int, float)
            # headersType.update({str(aHeader) : int})
        elif isFloat:
            check_dtype(aTable, aHeader, float, int)
            # headersType.update({str(aHeader) : float})
        elif isJson:
            check_dtype(aTable, aHeader, dict)
        elif aHeader in ignoreHeaders:
            continue
        else:
            raise Exception(
                f"Couldn't match the header {aHeader} to any header in the database"
            )

    print("Data types verified")


def date_to_list(x: str or int) -> list or np.NaN:
    """
    This function takes in a date in a string format (if it has unknown value in it like 'xx') or in an 5-integer format and converts it to a list of integers. The date will be in format of 'YYYY-MM-DD'. The function will return a list of integers in the format of [YYYY, MM, DD]. It will return None, if the input is None.
    :param x: The date to convert to a list of integers.
    :return: A list of integers in the format of [YYYY, MM, DD].
    """
    if x is None or pd.isna(x):
        return np.NaN
    elif type(x) is int:
        return list(
            map(
                int,
                pd.to_datetime(arg=x, errors="ignore", unit="D", origin="1899-12-30")
                .strftime("%Y-%m-%d")
                .split("-"),
            )
        )
    else:
        new_list: list = []
        for i, element in enumerate(str(x).replace("/", "-").split("-")[::-1]):
            try:
                new_list.append(int(element))
            except ValueError:
                new_list.append(element)
        return new_list


print("Done")
# -

# # 1. Reading the Excel file

# The dates in Excel file have a format of 5-digit numbers, which are the number of days since 00/00/1900 (or 1899-12-30). We need to convert them to a date format.
# Note that dates before 1900 will be an exact string representation of the exact date.
# For some reason, I couldn't use `converters` to convert the dates or `dtyps` while reading the Excel file, so I had to use `applymap` to convert them.

# +
file = glob(path.normpath(fs.InputPath.new_data + "/*.xlsx"))[0]
print(f"Read the file: {file}")
na_values = [
    "NA",
    "N/A",
    "na",
    "n/a",
    "NULL",
    "null",
    "not documented",
    "Not documented",
    "Not Documented",
    "nan",
    "NAN",
    "None",
    "none",
]
df = pd.read_excel(
    file,
    index_col=None,
    header=0,
    na_values=set(na_values),
    # true_values=('Yes', 'yes', 'TRUE', 'True', 'true'),
    # false_values=('No', 'no', 'FALSE', 'False', 'false'),
)

# This is used just in case if there are any spaces in the beginning or the end of the string and not being cought by the na_values
df = df.map(lambda x: x.strip() if type(x) is str else x)
df.replace(na_values, NaN, inplace=True)

# Topic: converting the dates to a date format,
# Converting the dates to a date format and splitting them into a year, month, and day list.
# This is done to make it easier to convert them to json later on.
date_cols = list(df.columns[df.columns.str.contains(r"date", re.IGNORECASE, re.I)])
df.loc[:, date_cols] = df.loc[:, date_cols].map(date_to_list)

display(date_cols)
df.head(5)
# -

# # 2. Changing some data values to a standard
#
# Below we will be using `df.loc[index, ('header')]` rather than `df[header][index]` since it is safer and faster.
# Https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
#
# Note that we are using `replace` method rather than using for loops and if statements.

# +
# below, I implemented {while loop -> try / except a -> match case} to make sure all statements run independently of one another and the try/except would still catch any errors
i = 0
modified_cols = []
while True:
    try:
        match i:
            # PersonInfo
            case 0:
                df["Gender"].replace(["Male", "Female"], [1, 0], inplace=True)
                modified_cols.append("Gender")
                i += 1
            case 1:
                df["Ethnicity"].replace(
                    ["Non-Spanish; Non-Hispanic", "Spanish; Hispanic"],
                    [0, 1],
                    inplace=True,
                )
                modified_cols.append("Ethnicity")
                i += 1

            # SampleInfo
            # CancerInfo
            case 2:
                df["Diagnosis"].replace(
                    [
                        "Non Small Cell Lung Cancer",
                        "Small Cell Lung Cancer",
                        "Breast Cancer",
                    ],
                    ["Non Small Cell Lung", "Small Cell Lung", "Breast"],
                    inplace=True,
                )
                modified_cols.append("Diagnosis")
                i += 1
            case 3:
                df["IHC Assay ER"].replace(
                    ["positive", "negative"], [1, 0], inplace=True
                )
                modified_cols.append("IHC Assay ER")
                i += 1
            case 4:
                df["IHC Assay PR"].replace(
                    ["positive", "negative"], [1, 0], inplace=True
                )
                modified_cols.append("IHC Assay PR")
                i += 1
            case 5:
                df["IHC Assay HER-2"].replace(
                    ["Not Amplified", "Equivocal", "Amplified"],
                    [-1, 0, 1],
                    inplace=True,
                )
                modified_cols.append("IHC Assay HER-2")
                i += 1
            case 6:
                df["FISH Test HER-2"].replace(
                    ["Not Amplified", "Equivocal", "Amplified"],
                    [-1, 0, 1],
                    inplace=True,
                )
                modified_cols.append("FISH Test HER-2")
                i += 1
            case 14:
                df["Stage"].replace([0], ["0"], inplace=True)
                modified_cols.append("Stage")
                i += 1

            # TreatmentInfo
            case 7:
                df["Tissue Exposure"].replace(["Yes", "No"], [1, 0], inplace=True)
                modified_cols.append("Tissue Exposure")
                i += 1
            case 8:
                df["Rad Tissue Exposure"].replace(["Yes", "No"], [1, 0], inplace=True)
                modified_cols.append("Rad Tissue Exposure")
                i += 1
            case 9:
                df["Chemo"].replace(
                    ["No", "Yes", "Prior Chemo", "Current Chemo"],
                    [0, 1, 2, 3],
                    inplace=True,
                )
                modified_cols.append("Chemo")
                i += 1
            case 10:
                df["Hormonal Therapy"].replace(["Yes", "No"], [1, 0], inplace=True)
                modified_cols.append("Hormonal Therapy")
                i += 1
            case 11:
                df["Immunotherapy"].replace(["Yes", "No"], [1, 0], inplace=True)
                modified_cols.append("Immunotherapy")
                i += 1
            case 12:
                response_columns = list(
                    df.columns[
                        df.columns.str.contains(
                            r"response to chemo course", re.IGNORECASE, re.I
                        )
                    ]
                )
                # df.loc[:, response_columns] = df.loc[:, response_columns].map(
                #     lambda x: x.strip() if type(x) is str else x)  # No need for this
                df.loc[:, response_columns] = df.loc[:, response_columns].replace(
                    [
                        "CR - Complete Response",
                        "PR - Partial Response",
                        "SD - Stable Disease",
                        "PD - Progressive Disease",
                    ],
                    ["CR", "PR", "SD", "PD"],
                )
                modified_cols += response_columns
                i += 1
            case 13:
                course_columns = list(
                    df.columns[
                        df.columns.str.fullmatch(r"course \d{,2}", re.IGNORECASE, re.I)
                    ]
                )
                df.loc[:, course_columns] = df.loc[:, course_columns].map(
                    lambda x: imp.split_strip(x, ["/", ";"]) if type(x) is str else x
                )
                modified_cols += course_columns
                i += 1
        print(f"Case {i - 1} -> Done")
        if i > 14:
            break

    except KeyError as e:
        print(f"Case {i} -> Error: {e} not found")
        i += 1
        pass

for col in modified_cols:
    print("------------------------------------------")
    print(df[col].value_counts(dropna=False))

# df.loc[25:30, modified_cols]
# -


# # 3. Changing header names to match DB

# +
# TODO: rewrite the rename function to use RE instead of hard coded names to better find the column names in df.

# making sure that specimen type is known
# checking the columns for 'specimen type'
spec_type_num = re.findall(
    r"\'specimen\s*type\s*", str(list(df.columns)), flags=re.IGNORECASE
)
message = f"A number of {len(spec_type_num)} specimen type column(s) were found,\ncontinue? (y/n)"
question = False  # initializing

# if there is more than one 'Specimen Type'
if len(spec_type_num) > 1 and (question := input(message).lower() == "n"):
    raise Exception(
        '1. Figure out which "specimen type" each one is\n2. Re-run the code'
    )

# Otherwise, if we found a specimen type
elif len(spec_type_num) == 1 or question:
    target = str(set(df.loc[:, "Specimen Type"]))
    is_glycan_spec = re.findall(
        r"Malignant \w*", target, flags=re.IGNORECASE
    ) + re.findall(r"Normal \w*", target, flags=re.IGNORECASE)
    if is_glycan_spec:
        # Glycan table: Normal Serum, Malignant Serum, Normal Tissue, Malignant Tissue
        df.rename(columns={"Specimen Type": "GlycanSpecimenType"}, inplace=True)
    else:
        # Galectin table: Serum, Plasma
        df.rename(columns={"Specimen Type": "GalectinSpecimenType"}, inplace=True)

df.rename(
    columns={
        # PersonalInfo Table
        "FWID": "ID",
        "Patient Birth Year": "BirthYear",
        "Smoking History": "SmokeHistory",
        "Gender": "Sex",
        # SampleInfo Table
        "Unique Aliquot ID": "AliquotID",
        "Specimen Considered": "SpecimenConsideration",
        "Specimen age in years": "SpecimenAge",
        "Specimen Collection Year": "SpecimenCollectionYear",
        "Patient Age at Collection": "PatientAgeAtCollection",
        "Additional Histology": "OtherHistology",
        "Tissue Site (Histo)": "Site",
        "Specimen Grade": "Grade",
        # CancerInfo Table
        "IHC Assay ER": "IHC_Assay_ER",
        "IHC ER %": "IHC_ER",
        "IHC Assay PR": "IHC_Assay_PR",
        "IHC PR %": "IHC_PR",
        "IHC Assay HER-2": "IHC_Assay_HER2",
        "FISH Test HER-2": "FISH_Test_HER2",
        # TreatmentInfo Table
        "Tissue Exposure": "TissueExposure",
        "Rad Tissue Exposure": "RadTissueExposure",
        "Hormonal Therapy": "HormonalTherapy",
        "Treatment Status": "TreatmentStatus",
        "Surgery Date": "SurgeryDate",
        "Patient Chemo": "Chemo",
        # Galectins
        "Gal-1": "Gal1",
        "Gal-3": "Gal3",
        "Gal-7": "Gal7",
        "Gal-8": "Gal8",
        "Gal-9": "Gal9",
        # Glycan
        # Genes
        # this is under the assumption that a single file will not have duplicate names
        "ABLI": "ABL1",
        "ABL": "ABL1",
        "AKT": "AKT1",
        "AKTI": "AKT1",
        "CDHI": "CDH1",
        "CDH": "CDH1",
        "CSFIR": "CSF1R",
        "CTNNBI": "CTNNB1",
        "CTNNB": "CTNNB1",
        "FGFRI": "FGFR1",
        "GNAII": "GNA11",
        "GNA": "GNA11",
        "HNFIA": "HNF1A",
        "MLHI": "MLH1",
        "NOTCHI": "NOTCH1",
        "NPMI": "NPM1",
        "NPM": "NPM1",
        "PTPN": "PTPN11",
        "PTPNII": "PTPN11",
        "SMARCBI": "SMARCB1",
        "SMARCB": "SMARCB1",
        "SDK": "SDK11",
        "SDKII": "SDK11",
        "RBI": "RB1",
        # Pathways
        "jak/stat": "JAK_STAT",
        "p53": "P53",
    },
    inplace=True,
)

print("Done")
# -

# This cell is to fix a specific error in the data
run_this = False
run_this = (
    input("Do you want to run this cell\nto modify some data? (y/n)").lower() == "y"
)
if run_this:
    # df.loc[199, 'IHC_PR']=np.NaN
    # df.loc[:, 'Grade'].replace(['U'], [0], inplace=True)
    # df.loc[:, 'Grade'] = df.loc[:, 'Grade'].map(int, na_action='ignore')
    pass

# Checking if we ended up with duplicated column names from last step
columns = df.columns
temp = False
for i in range(len(columns)):
    for j in range(i + 1, len(columns)):
        if columns[i] == columns[j]:
            temp = True
            print(f"Columns '{columns[i]}' and '{columns[j]}' have the same name")
if temp:
    raise Exception(
        "You have duplicate columns that you need to either fix or merge them"
    )
print("Done")

# # 4. Dividing the Data Frame
# We use `df2 = df[['col1', 'col2']].copy()` to choose related columns and separate them. Although we can ignore the `.copy()`, using that would avoid a `SettingWithCopyWarning` error. The inner square brackets define a Python list with column names, whereas the outer brackets are used to select the data from a pandas DataFrame as seen.
# Also, depending on the number of columns in your original dataframe, it might be more succinct to express this using a drop (this will also create a copy by default), like `df2 = df.drop('col3', axis=1)`
#
# To merge separate dataframes together, we use `pd.concat(dfList)`. As the default argument `axis = 0`, this will **union** the dataframes in a vertical order and the result will be like: (df1.col1 + df2.col1 + ...), (df1.col2 + df2.col2 + ...), etc.
# In other words, `axis = 0` means that all the values from first df first col will be added to the result df first col, then add all the values from second df first col will be added to the result df first col, and so on.
# If we want to merge the dataframes in a horizontal order, we must specify `axis = 1`. This will result in a dataframe looking like: df1.col1, df1.col2, ..., df2.col1, df2.col2, ..., etc.
# There is also an argument called `ignore_index` that by default is set to false. If set True, the result dataframe will not have any headers. Read source for more options explanation.
#
# source:
# https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html
# https://sparkbyexamples.com/pandas/pandas-create-new-dataframe-by-selecting-specific-columns/
# https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html

# +
# Dropping any row that doesn't have an ID
df.dropna(subset=["ID"], inplace=True)

# dataframes related to the database
PersonInfo = slice_df(df, "PersonInfo")
CancerInfo = slice_df(df, "CancerInfo")
SampleInfo = slice_df(df, "SampleInfo")
Galectin = slice_df(df, "Galectin")
Glycan = slice_df(df, "Glycan")
Genes = slice_df(df, "Genes")
Pathways = slice_df(df, "Pathways")
try:
    TreatmentInfo = slice_df(df, "TreatmentInfo").drop(
        columns=["SurgeryDate"]
    )  # SurgeryDate will be added to JSON "TreatmentCourse" later
except KeyError:
    print("No 'SurgeryDate' column was found")
    TreatmentInfo = slice_df(df, "TreatmentInfo")

print("Done")
# -

# # 5. Adding JSON values
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_json.html?utm_source=pocket_saves
# https://github.com/ijl/orjson
#
# ## 5.1 Gene Mutations
# To improve the speed if needed, you can look into using package "swiftly", Numba, or CUPY. See obsidian notes.

# +
# todo: Verify the mutations using hgfv library. You may want to add it to the genes_str_json function
# todo: see if np.vectorize() have any impact on function speed
# Victorizing the function, assuming it will make it run faster
# vec_func = np.vectorize(str_to_dict)

# creating a pattern to match the mutations but not ID and EdgeCase
cols_to_apply_g = ~Genes.columns.isin(["ID", "EdgeCase"])

# Applying the function to the dataframe to convert the string mutations to json format
Genes.loc[:, cols_to_apply_g] = Genes.loc[:, cols_to_apply_g].applymap(
    str_to_dict, data_json_name="mutations", na_action="ignore"
)
print("Done")
# -

# ## 5.2 Pathways
# We will keep this simple just having the count of mutations. This is exactly what is reported to us. To do this, I will use the same helper function `genes_str_json` that I used for the Genes table.
# In the future, we may want to add the genes that are related to the pathway in the json value.

cols_to_apply_p = ~Pathways.columns.isin(["ID", "EdgeCase"])
Pathways.loc[:, cols_to_apply_p] = Pathways.loc[:, cols_to_apply_p].applymap(
    str_to_dict,
    keep_original=True,
    na_action="ignore",
    include_count=False,
    cast_data=int,
)
print("Done")

# ## 5.3 TreatmentInfo

# +
# fixme: check to see if you need to split TreatmentStatus too?
# todo: find a way to sort courses better in the json str like {course01 : {treatment: [], chemo: xxx, startdate: [], response: xxx},...}

# to get all the columns that have the word "course" or "SurgeryDate" or "TreatmentStatus"
pattern = r"course\s+\d+|SurgeryDate|TreatmentStatus"

# Getting the columns that match the pattern from Excel file dataframe
treatment_json_col = list(
    df.columns[df.columns.str.contains(pattern, re.IGNORECASE, re.I)]
)

# Getting all the data in a single dataframe
treatment_json_df = sub_df(df, treatment_json_col)

# this gives back a list of strings that we are assigning to "TreatmentStatus"
# Creating json values for each row like {col1: val1, col2: val2, ...}
# if you don't want to have a dictionary, just remove [json.loads(x) for x in ...] and keep the rest (the ... code)
if type(treatment_json_df) == pd.DataFrame:
    if del_null_values := True:
        TreatmentInfo["TreatmentCourses"] = [
            clean_nulls(x)
            for x in (
                treatment_json_df.to_json(orient="records", lines=True).splitlines()
            )
        ]
    else:
        TreatmentInfo["TreatmentCourses"] = [
            json.loads(x)
            for x in treatment_json_df.to_json(
                orient="records", lines=True
            ).splitlines()
        ]
else:
    print("No JSON values for 'TreatmentInfo' table was found")
print("Done")
# -

# ## 5.4 Verifying the JSON values

# +
# Note that if del_null_values is True, then you might see treatmentCourses empty, because it was removed.
if treatment_json_df is not None:
    print(f"TreatmentInfo\n{TreatmentInfo.loc[0, 'TreatmentCourses']}")
    print("----------------------------------------------------")

print(f"Genes\n{Genes.loc[0, cols_to_apply_g]}")
print("----------------------------------------------------")
print(f"Pathways\n{Pathways.loc[0, cols_to_apply_p]}")
# -


# # 6. Dropping Duplicates
# Based on table's primary key. The function will reset the index in the background.
#
# source: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html

# +
# todo: use a loop to do this rather than hard coding the table names
PersonInfo = dropDup("PersonInfo")
Pathways = dropDup("Pathways", ignore_columns=["EdgeCase"])
Genes = dropDup("Genes", ignore_columns=["EdgeCase"])

# fixme: remove additional arguments when you get access to AliquotIDs
TreatmentInfo = dropDup("TreatmentInfo", ignore_columns=["AliquotID"])
CancerInfo = dropDup("CancerInfo", ignore_columns=["AliquotID"])
SampleInfo = dropDup("SampleInfo", ignore_columns=["AliquotID"])
Galectin = dropDup("Galectin", ignore_columns=["AliquotID"])
Glycan = dropDup("Glycan", ignore_columns=["AliquotID", "GlycanSpecimenType"])

print("Done")
# -

# ## 6.1 Saving the result to a dictionary and removing empty tables

# +
# todo: use a loop to do this rather than hard coding the table names
df_dict: dict = {
    "PersonInfo": PersonInfo,
    "CancerInfo": CancerInfo,
    "SampleInfo": SampleInfo,
    "TreatmentInfo": TreatmentInfo,
    "Galectin": Galectin,
    "Glycan": Glycan,
    "Genes": Genes,
    "Pathways": Pathways,
}

# Removing Empty tables
# Getting all the table names from the database
tableNames = set(pk_schema["table_name"])
# Setting the ignore_columns to ignore when comparing the primary keys
ignore_columns = ["EdgeCase"]

for aTable in tableNames:
    # 1.1 list the primary keys of aTable
    aTable_pk = set(pk_schema[pk_schema["table_name"] == aTable]["col_name"])

    # making sure that ignore_columns is a list if not None
    if ignore_columns is not None:
        aTable_pk = aTable_pk - set(ignore_columns)

    # 2. get the primary keys of that table from df_dict and deleting that table if it is empty
    if set(df_dict[aTable].columns) - set(aTable_pk) == set():
        del df_dict[aTable]
        print(f"Table {aTable} -> Empty and has been removed")

# todo: remove empty rows from each table that only have primary keys in them
# -

# # 7. Verifying & Fixing data Types
#
# ## what is the difference between `check_dtype` function and `verify_dtype` ?
# check_dtype function checks the data types of a specific column in a pandas DataFrame and raises an exception if any data in that column does not match the specified data type or is not of the specified ignored data type.
#
# verify_dtype function verifies the data types of all the columns in a pandas DataFrame against a specified DataFrame of column data types. It compares the data types of each column in the given DataFrame (aTable) with the corresponding column in the specified DataFrame of column data types (aDB_schema). If the data type of a column in aTable does not match the corresponding column in aDB_schema, it raises an exception.
#
# In summary check_dtype checks for the datatype of one specific column, while verify_dtype checks for the datatype of all columns in a DataFrame against another DataFrame.

try:
    for key in df_dict.keys():
        if len(df_dict[key].columns) > 1:
            # print(f"Table {key} -> ", end='')
            verify_dtype(df_dict[key], db_schema, key)
            # print(f"Data types verified")

    print("\nAll good!\n")
except Exception as e:
    raise Exception(
        f"Error: {e} \n\n======= Please fix the problem in the IMPORT FILE cell and run ALL cells again ======="
    )
print(
    "===== Note that JSON values datatype were verified as dictionaries =====\n\t\t\t===== In-dict dataTypes were not verified ====="
)

# # 8. Pickling Data

# +
overwrite = True
for name in [nameof(df), nameof(df_dict)]:
    test = glob(join(PklPath.data_importer, f"{name}.pkl"))
    if test:
        overwrite = input(f"{name} exists, Overwrite? (y/n)").lower() == "y"
        if not overwrite:
            break
        # todo: make it check each one independently

if overwrite:
    with open(join(PklPath.data_importer, "df.pkl"), "wb") as pickle_out:
        pickle.dump(df, pickle_out)

    with open(join(PklPath.data_importer, "df_dict.pkl"), "wb") as pickle_out:
        pickle.dump(df_dict, pickle_out)
else:
    raise Exception("Allow overwriting, or rename")

print("Done")
