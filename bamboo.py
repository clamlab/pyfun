"""
misc functions for wrangling pandas data
"""

import pandas as pd, numpy as np
from collections import defaultdict
import os

from pyfun import string_utils
from IPython.display import display



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

    ''' === this seems unnecessary ===
    # If a bin_range is specified, set values outside of this range to NaN
    if bin_range:
        mask_outside_range = (df[col_to_bin] < bin_range[0]) | (df[col_to_bin] > bin_range[1])
        df.loc[mask_outside_range, col_to_bin] = np.nan
    '''

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

def chainslice(df, slice_instructions):
    """ perform a series of slicing operations using slice_df
     slice_instructions: nested list, each element [col, [vals], polarity] """

    for [col, vals, polarity] in slice_instructions:
        df = slice(df, {col: vals}, polarity)

    return df

def concat_df_dicts(df_dict, reset_index=True):
    """
    concatenate a dictionary of dfs, in order of the keys, after sorting
    keys are e.g. date strings
    """

    # Sort the keys in ascending order
    sorted_keys = sorted(df_dict.keys())

    # Concatenate the dataframes in the correct order
    list_of_dfs = [df_dict[key] for key in sorted_keys]
    if len(list_of_dfs) > 0:
        grand_df = pd.concat(list_of_dfs)
    else:
        return pd.DataFrame()

    if reset_index:
        grand_df = grand_df.reset_index(drop=True)

    return grand_df


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

def count_null(df, col):
    """
    count number / pct of null entries in a column
    :param col: Column name to count
    """

    n = len(df)
    n_null = np.sum(df[col].isnull())

    print(str(n_null), 'NaNs (', str(n_null / n * 100), '%).')



def euclid(df, xy_col1, xy_col2):
    """
    computes euclidean distance between two sets of xy coordinates
    :param xy_col1: list with pair of column names for [x,y] coordinate
    :param xy_col2: second pair of column names
    :return:
    """

    A = np.array(df[xy_col1]).astype('float')
    B = np.array(df[xy_col2]).astype('float')
    C = np.linalg.norm(A-B,axis=1)

    return C

def find_row_closest(search_row, analog_col, df_haystack, direction='bi'):
    """
    given row in one df, find closest rows in another df based on analog value of a column
    :param search_row: #TODO complete docs
    :param cols_to_match:
    :param df_haystack:
    :return: index of found row
    """
    df_haystack = df_haystack.copy()
    df_haystack.loc[:, 'centered'] = df_haystack[analog_col] - search_row[analog_col].values[0]
    if direction == 'bi':
        df_haystack.loc[:, 'error'] = df_haystack['centered'].apply(lambda x: x ** 2)
    elif direction == 'lt':
        df_haystack.loc[:, 'error'] = df_haystack['centered'].apply(lambda x: x)

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

def read_csv_or_create(csv_path,colnames):
    """
    check if fpath (pointing to csv) exists.
    if it does, load csv and return
    if it does not, create csv with colnames, and return df
    """
    if os.path.exists(csv_path):
        print('Existing file loaded')
        return pd.read_csv(csv_path)
    else:
        print(csv_path, 'not found, creating new')
        df = pd.DataFrame(columns=colnames)
        df.to_csv(csv_path, index=False)
        return df

def slice_notnull(df, col):
    return df[df[col].notnull()]


def slice(df, col_row_vals, polarity='+', print_counts=False):
    """
    Slices a pandas DataFrame based on given column values or ranges.

    Parameters:
        df (pd.DataFrame): The DataFrame to slice.
        col_row_vals (dict): A dictionary where keys are column names and values are:
            - a list of values to match, or
            - a string representing a range in the form '(a,b]', '[a,b)', '[a,b]', or '(a,b)'.
        polarity (str): '+' to return matching rows, '-' to return non-matching rows.
        print_counts (bool): If True, prints the value counts of the resulting DataFrame for each column in `col_row_vals`.

    Returns:
        pd.DataFrame: The resulting DataFrame.

    Raises:
        ValueError: If the polarity is not '+' or '-'.
    """

    #find the intersecting set
    intersect = df.copy()
    for k, v in col_row_vals.items():
        if isinstance(v, list):
            intersect = intersect[intersect[k].isin(v)]
        elif isinstance(v, str): #string that represents numerical interval to slice
            intersect = slice_col_range(intersect, k, v)


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


def slice_col_range(df, colname, range_str):
    """
    Slices a pandas DataFrame based on a column in a given range string.

    Parameters:
        df (pd.DataFrame): The DataFrame to slice.
        colname (str): The name of the column to slice.
        range_str (str): The range string, e.g., '(1,4)' or '[1,4]'.

    Returns:
        pd.DataFrame: The sliced DataFrame.

    Raises:
        ValueError: If the range_str is not formatted correctly.
    """

    # Check the first and last characters for brackets
    if range_str[0] not in '([' or range_str[-1] not in ')]':
        raise ValueError("range_str should start with '(' or '[' and end with ')' or ']'.")

    # Determine inclusiveness for the boundaries
    left_inclusive = range_str[0] == '['
    right_inclusive = range_str[-1] == ']'

    inclusive = 'both' if left_inclusive and right_inclusive else \
        'left' if left_inclusive else \
            'right' if right_inclusive else \
                'neither'

    # Extract the numeric values
    try:
        left_value, right_value = map(float, range_str[1:-1].split(','))
    except ValueError:
        raise ValueError("range_str should contain two numeric values separated by a comma.")

    return df[df[colname].between(left_value, right_value, inclusive=inclusive)]


class SliceLabel:
    """
    Utility class for converting slice instructions to labels.
    """
    # Class-level attributes
    _bool2prefix = {True: '', False: '¬'}
    _sign2prefix = {'+': '', '-': '¬'}
    _sign2symbol = {'+': '∈', '-': '∉'}
    _bool_trialflags = ['WMTrial', 'optoTrial']

    @staticmethod
    def make(slicer):
        """
        Go from a set of chainslice instructions to a label for plot/analysis.
        """
        label_parts = []

        for s in slicer:
            if isinstance(s[1], str):  # str representing slice interval for boo.slice_col_range()
                label_part = SliceLabel._range_to_label(s)
            elif isinstance(s[1], list):
                if s[0] == 'TrialType':
                    label_part = SliceLabel._trialtype_to_label(s)
                elif s[0] in SliceLabel._bool_trialflags:
                    label_part = SliceLabel._bool_to_label(s)
                else:
                    raise ValueError('unknown slicer')
            else:
                raise ValueError('unknown slicer')

            label_parts.append(label_part)

        label_parts = sorted(label_parts, key=lambda x: x.lstrip("¬"))  # sort as if '¬' character is absent
        return ', '.join(label_parts)

    @staticmethod
    def _bool_to_label(instr):
        """
        Convert single bamboo.chainslice() instruction for bool trial type (WMTrial, optoTrial...) value,
        to label part.
        """
        trial_var = instr[0].replace('Trial', '')  # e.g. 'WMTrial', 'optoTrial'...
        slice_bool = instr[1:]  # e.g. [[True], '+']

        if slice_bool[0] not in [[True], [False]]:
            raise ValueError('slice_bool[0] must be either [True] or [False]')

        bool_value = slice_bool[0][0]

        if slice_bool[1] == '-':
            bool_value = not bool_value
        elif slice_bool[1] != '+':
            raise ValueError("slice_bool[1] must be '+' or '-'.")

        return SliceLabel._bool2prefix[bool_value] + trial_var

    @staticmethod
    def _trialtype_to_label(instr):
        """
        Convert single bamboo.chainslice instruction for trial type, to label part.
        """
        trialtypes = instr[1]

        if len(trialtypes) == 1:
            label = trialtypes[0]
        elif len(trialtypes) > 1:
            label = "(" + "+".join(trialtypes) + ")"
        else:
            raise ValueError('The number of trialtypes should be >= 1.')

        return SliceLabel._sign2prefix[instr[2]] + label

    @staticmethod
    def _range_to_label(instr):
        """
        Convert single bamboo.chainslice instruction which specifies slice interval by str, to
        label part.
        """
        return f"{instr[0]} {SliceLabel._sign2symbol[instr[2]]} {instr[1]}"




class SliceReader:
    """
    read chainslicers from csv file.
    usage: SliceReader.read_csv(csv_file)
    """

    @staticmethod
    def _parse_col_type(df):
        """
        Determine whether column values should be list-ified.

        Parameters:
        df (pandas.DataFrame): DataFrame to analyze.

        Returns:
        dict: A dictionary where keys are column names and values are True or False indicating if the column should be list-ified.
        """
        listify = {}  # value can be True or False
        # enclose value in list if the slice var type is bool or str
        # do not listify if slice var type is str representing interval

        for colname, vals in df.items():

            if vals.dtype == 'bool':
                listify[colname] = True
                continue

            vals = vals.dropna()

            if vals.dtype == 'object':
                if string_utils.is_interval(vals.iloc[0]):
                    listify[colname] = False
                elif type(vals.iloc[0]) is bool:
                    listify[colname] = True
                else:
                    listify[colname] = True

        return listify

    @staticmethod
    def read_csv(csv_file):
        """
        Convert CSV file to a list of chainslice instructions.

        Parameters:
        csv_file (str): Path to the CSV file.

        Returns:
        list: A list of chainslice instructions.
        """
        slicer_df = pd.read_csv(csv_file)
        all_chainslicers = []

        listify = SliceReader._parse_col_type(slicer_df)

        for i, row in slicer_df.iterrows():
            chainslicer = []
            for k, v in row.items():
                if pd.isna(v):
                    continue

                if listify[k]:
                    slice_val = [v]
                else:
                    slice_val = v

                slicer = [k, slice_val, '+']
                chainslicer.append(slicer)
            all_chainslicers.append(chainslicer)

        labels = [SliceLabel.make(c) for c in all_chainslicers]
        slicer_df['label'] = labels
        display(slicer_df)


        return all_chainslicers, slicer_df


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