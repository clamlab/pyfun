"""
misc functions for wrangling pandas data
"""

import pandas as pd, numpy as np
from collections import defaultdict

import pandas as pd
import numpy as np


def bin_col(df, col_to_bin, n_bins, bin_range=None, bins_from_range=False):
    """
    Bin values in a column within a specified range.

    :param df: DataFrame with the data
    :param col_to_bin: column name to be binned
    :param n_bins: number of bins or array of bin edges
    :param bin_range: tuple (a, b) to specify the range of values to be binned
    :param bins_from_range: if True, the bins are derived from bin_range, otherwise from data
    :return: DataFrame with an additional column for the binned values
    """

    df = df.copy()  # To avoid modifying the original dataframe

    # If a bin_range is specified, set values outside of this range to NaN
    if bin_range:
        mask_outside_range = (df[col_to_bin] < bin_range[0]) | (df[col_to_bin] > bin_range[1])
        df.loc[mask_outside_range, col_to_bin] = np.nan

    # Decide the bin edges
    if bins_from_range and bin_range:
        bin_edges = np.linspace(bin_range[0], bin_range[1], n_bins + 1)
    else:
        bin_edges = n_bins

    # Use pandas cut to create bins
    df['bin'] = pd.cut(df[col_to_bin], bins=bin_edges)

    # Extract midpoints from bin intervals
    df[col_to_bin + '_bin'] = df['bin'].apply(lambda x: (x.left + x.right) / 2 if pd.notna(x) else np.nan)

    # Drop the 'bin' column
    df = df.drop('bin', axis=1)

    return df




def convert_dtype(df, col, dtype):
    df[col] = df[col].astype(dtype)



def copy_dict_dfs(dict_of_dfs):
    """
    copy dictionary containing dataframes
    :param dict_of_dfs:
    :return:
    """
    new_dict = {}
    for k, v in dict_of_dfs.items():
        new_dict[k] = v.copy()

    return new_dict


def concat_df_dicts(df_dict, reset_index=True):
    """
    concatenate a dictionary of dfs, in order of the keys, after sorting
    keys are e.g. date strings
    """

    # Sort the keys in ascending order
    sorted_keys = sorted(df_dict.keys())

    # Concatenate the dataframes in the correct order
    grand_df = pd.concat([df_dict[key] for key in sorted_keys])

    if reset_index:
        grand_df = grand_df.reset_index(drop=True)

    return grand_df



def chainslice(df, slice_instructions):
    """ perform a series of slicing operations using slice_df
     slice_instructions: nested list, each element [col, [vals], polarity] """

    for [col, vals, polarity] in slice_instructions:
        df = slice(df, {col: vals}, polarity)

    return df


def find_row_closest(search_row, analog_col, df_haystack):
    """
    given row in one df, find closest rows in another df based on analog value of a column
    :param search_row: #TODO complete docs
    :param cols_to_match:
    :param df_haystack:
    :return: index of found row
    """
    df_haystack = df_haystack.copy()
    df_haystack.loc[:, 'centered'] = df_haystack[analog_col] - search_row[analog_col].values[0]
    df_haystack.loc[:, 'error'] = df_haystack['centered'].apply(lambda x: x ** 2)
    min_i = df_haystack['error'].idxmin()

    return min_i


def find_row_match(search_row, cols_to_match, df_haystack, find_single=True, verbose=True):
    """
    given row in one df, find matching rows in another df based on some columns
    :param search_row: pandas Series corresponding to single row #TODO complete docs
    :param cols_to_match:
    :param df_haystack:
    :return:
    """
    m = []  # define search instructions
    for c in cols_to_match:
        m.append([c, [search_row[c]], '+'])

    matched_rows = chainslice(df_haystack, m)

    if len(matched_rows) == 0:
        if verbose==True:
            print('No matches found')
        return None
    elif len(matched_rows) > 1:
        if find_single:
            print(search_row)
            print(matched_rows)
            raise ValueError('More than one matching row found')
        else:
            return matched_rows
    elif len(matched_rows) == 1:
        return matched_rows


def match_dfs(df1, df2, label1, label2, cols_to_match):
    """
    find df1 within df2
    :param df1:
    :param df2:
    :param label1:
    :param label2:
    :param label2:
    :param cols_to_match:
    :return:
    """


    #define column containing row numbers of matches in the *other* df
    df1_col = label2 + '_row'
    df2_col = label1 + '_row'

    for df, col in zip([df1, df2], [df1_col, df2_col]):
        df[col] = None

    for i1, row1 in df1.iterrows():
        row2 = find_row_match(row1, cols_to_match, df2, find_single=True)
        try:
            i2 = row2.index[0]
        except AttributeError:
            if row2 is None:
                print(row1)
            continue

        df1.loc[i1, df1_col] = i2
        df2.loc[i2, df2_col] = i1


def slice_notnull(df, col):
    return df[df[col].notnull()]


def slice(df, col_row_vals, polarity='+', print_counts=False):
    """ slice df by finding matching row values in a given cols
    col_row_vals: a dict of {colname1: [value1, value2...], -->
                             colname2: [value3, value4...]}
        --> finds (colname1 with value1 or value2 or ...) AND  (colname2 with value3 or value4 or ...)

    polarity = + or -, returns matching or nonmatching
    
    returns a copy, not a view
    """


    #find the intersecting set
    intersect = df.copy()
    for k, v in col_row_vals.items():
        intersect = intersect[intersect[k].isin(v)]

    if polarity == '+':
        df_small = intersect
    elif polarity == '-':
        df_small = df.drop(intersect.index, axis='index')
    else:
        raise ValueError('Matching logic not found.')


    if print_counts:
        for col in col_row_vals:
            print(col)
            print(df_small[col].value_counts())

    return df_small

def slice_col_range(df, col, val_range):
    return df[df[col].between(val_range[0], val_range[1])]




def group_datetime_objects_by_date(datetimes):
    """
    Group pandas datetime objects by the date (i.e. those on same date will be collected together)
    :param datetimes: list of pd datetime objects
    :return: dictionary where keys are dates, values are lists of datetime obj which share that date
    """
    # Sort the keys
    sorted_keys = sorted(datetimes)

    # Group datetime objects by date
    grouped_datetimes = defaultdict(list)  # default dict has items sorted by when they were added
    for key in sorted_keys:
        grouped_datetimes[key.date()].append(key)

    return grouped_datetimes

def merge_by_index(dfs, index=True, join='outer'):
    #TODO: combine function with merge_update()
    """
    Merge a list of dataframes on their index (or a specified column) with an outer or inner join.

    Parameters:
    - dfs (list of pandas.DataFrame): The list of dataframes to be merged.
    - index (bool, optional): Whether to use the index as the merge key (True) or to use a specified column (False). Default is True.
    - join_method: e.g. 'outer' or 'inner'
    Returns:
    - pandas.DataFrame: The merged dataframe with all the rows from all the dataframes and NaN values where there is no match.
    """

    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, left_index=index, right_index=index, how=join)
    return merged_df


def merge_update(main_df, added_values, update_col, match_column='TrialNum'):
    # TODO: combine this function with merge_by_index
    """
    Update the values of column update_col in main_df with values from added_values based on the match_column.

    Parameters:
    - main_df (pd.DataFrame): The main dataframe to be updated.
    - added_values (pd.DataFrame): The dataframe from which values will be taken.
    - update_col (str): The column name in main_df to be updated.
    - match_column (str, optional): The column on which both dataframes should be matched. Defaults to 'TrialNum'.

    Returns:
    pd.DataFrame: The updated dataframe.
    """
    if added_values is not None:

        main_df[update_col] = main_df[match_column].map(added_values.set_index(match_column)[update_col]).fillna(
            main_df[update_col])
    return main_df


def merge_within_day(input_dict, date_col='date'):
    """
    Combine dataframes occurring on same date (day)
    :param input_dict: Dict of dataframes where keys are datetime strings, and values are dataframes
    :return: Dict where keys are dates, values are combined dataframes
    """
    grouped_datetimes = group_datetime_objects_by_date(input_dict.keys())
    combined_dataframes = {}

    for date, datetime_objects in grouped_datetimes.items():
        # Concatenate the dataframes
        combined_df = pd.concat([input_dict[dt] for dt in datetime_objects])

        # Add a new column with the shared date
        combined_df[date_col] = date.strftime('%Y-%m-%d')

        # Add the combined dataframe to the new dictionary
        combined_dataframes[date.strftime('%Y-%m-%d')] = combined_df

    return combined_dataframes


def swap_col_with_index(df, colname, name_of_index):
    """
    Parameters:
    -----------
    df : pd.Dataframe
    colname: name of column to be set as index
    name_of_index: name of new column to be created, to take the old index
    """

    df = df[[colname]].copy()
    df[name_of_index] = df.index
    df = df.set_index(colname)

    return df