"""
misc functions for wrangling pandas data
"""

from collections import defaultdict
import os

from . import string_utils



def bin_col(df, col_to_bin, n_bins, bin_range=None, bins_from_range=False, single_values=[]):
    """
    Bin values in a column within a specified range, allowing specified single values to have their own bins.
    Creates bins in subranges, ensuring bins are as close to the desired size as possible,
    while treating single values as special cases directly assigned their own bins.

    :param df: DataFrame with the data
    :param col_to_bin: column name to be binned
    :param n_bins: number of bins or array of bin edges
    :param bin_range: tuple (a, b) to specify the range of values to be binned
    :param bins_from_range: if True, the bins are derived from bin_range, otherwise from data
    :param single_values: list of values to exclude from the range-based bins and give their own bins
    :return: DataFrame with an additional column for the binned values
    """
    df = df.copy()  # Avoid modifying the original dataframe

    # Ensure single_values is sorted and unique
    single_values = sorted(set(single_values))

    # Determine the binning range
    if bin_range:
        range_start, range_end = bin_range
    else:
        range_start, range_end = df[col_to_bin].min(), df[col_to_bin].max()

    # Filter single_values to include only those within the binning range
    single_values = [v for v in single_values if range_start <= v <= range_end]

    # Calculate desired bin size
    total_range = range_end - range_start
    desired_bin_size = total_range / n_bins

    # Split the range into subranges separated by single_values
    subranges = [range_start] + single_values + [range_end]

    # Create bins for each subrange
    bin_edges = []
    for i in range(len(subranges) - 1):
        start, end = subranges[i], subranges[i + 1]

        if start in single_values:
            # Single value gets its own bin
            bin_edges.append(start)  # Start and end of the same value
            bin_edges.append(start)
        else:
            # Create bins within the subrange
            num_bins = max(1, round((end - start) / desired_bin_size))
            subrange_bins = np.linspace(start, end, num_bins + 1)
            bin_edges.extend(subrange_bins[:-1])  # Exclude the last edge to avoid duplication
    bin_edges.append(range_end)  # Add the final edge

    # Ensure unique and sorted bin edges
    bin_edges = sorted(set(bin_edges))

    # Use pandas cut to create bins
    df['bin'] = pd.cut(df[col_to_bin], bins=bin_edges, include_lowest=True)

    # Assign midpoints for single values or regular bins
    def calculate_midpoint(interval):
        if interval.left == interval.right:  # Single value case
            return interval.left
        return (interval.left + interval.right) / 2

    df[col_to_bin + '_bin'] = df['bin'].apply(
        lambda x: calculate_midpoint(x) if pd.notna(x) else np.nan
    )

    # Drop the 'bin' column
    df = df.drop('bin', axis=1)

    return df




def chainslice(df, slice_instructions):
    """ perform a series of slicing operations using slice_df
     slice_instructions: nested list, each element [col, [vals], polarity] """


    for [col, vals, polarity] in slice_instructions:
        df = slice(df, {col: vals}, polarity)

    return df


import pandas as pd
import numpy as np


def chunk(df, mode, n_chunks=None, chunk_size=None):
    """
    Split a pandas DataFrame into chunks based on the specified mode.

    Parameters:
    df : pd.DataFrame
        The DataFrame to split.
    mode : str
        Chunking mode. One of:
        - 'n_chunks': Split into a given number of chunks.
        - 'chunk_size_strict': Split into chunks of fixed size, last chunk may be smaller.
        - 'chunk_size_rough': Split into chunks close to chunk_size, balanced.
    n_chunks : int, optional
        Number of chunks to split into (only for 'n_chunks' mode).
    chunk_size : int, optional
        Desired chunk size (only for 'chunk_size_strict' or 'chunk_size_rough').

    Returns:
    List[pd.DataFrame]
        List of DataFrame chunks.

    Raises:
    ValueError:
        If required parameters for the selected mode are missing or invalid.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    total_rows = len(df)

    if mode == "n_chunks":
        if n_chunks is None or n_chunks <= 0:
            raise ValueError("n_chunks must be a positive integer for mode 'n_chunks'.")
        avg_chunk_size = round(total_rows / n_chunks)
        chunks = []
        for i in range(n_chunks):
            start = i * avg_chunk_size
            end = start + avg_chunk_size
            chunks.append(df.iloc[start:end])
        if end < total_rows:
            chunks[-1] = pd.concat([chunks[-1], df.iloc[end:]])
        return chunks

    elif mode == "chunk_size_strict":
        if chunk_size is None or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer for mode 'chunk_size_strict'.")
        return [df.iloc[i:i + chunk_size] for i in range(0, total_rows, chunk_size)]

    elif mode == "chunk_size_rough":
        if chunk_size is None or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer for mode 'chunk_size_rough'.")
        num_chunks = int(np.ceil(total_rows / chunk_size))
        return np.array_split(df, num_chunks)

    else:
        raise ValueError("Invalid mode. Must be one of: 'n_chunks', 'chunk_size_strict', 'chunk_size_rough'.")


def concat_df_dicts(df_dict, reset_index=True):
    """
    Concatenate a dictionary of dataframes in the order of sorted keys.
    Filters out empty or all-NA dataframes to ensure compatibility with future pandas versions.

    Args:
        df_dict (dict): Dictionary of dataframes to concatenate, keyed by e.g. date strings.
        reset_index (bool): Whether to reset the index in the final concatenated dataframe.

    Returns:
        pd.DataFrame: Concatenated dataframe, or an empty dataframe if no valid dataframes exist.
    """

    # Sort the keys in ascending order
    sorted_keys = sorted(df_dict.keys())

    # Filter out empty or all-NA dataframes
    list_of_dfs = [
        df_dict[key]
        for key in sorted_keys
        if not df_dict[key].empty and not df_dict[key].isna().all(axis=None)
    ]

    # Check if there are valid dataframes to concatenate
    if list_of_dfs:
        grand_df = pd.concat(list_of_dfs)
        if reset_index:
            grand_df = grand_df.reset_index(drop=True)
        return grand_df

    # Return an empty dataframe if no valid dataframes
    return pd.DataFrame()



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


def find_duplicates(df, columns):
    """
    Find duplicate rows in a DataFrame based on specified columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of column names to check for duplicates.

    Returns:
        pd.DataFrame: A DataFrame containing the duplicate rows.
    """
    duplicates = df[df.duplicated(subset=columns, keep=False)]
    return duplicates


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


def read_csv(filepath, dtype_dict, **kwargs):
    """
    Enhanced CSV reader when specifying dtype of columns

    Use single dtype dictionary (`dtype_dict`) is used to specify column data types.
    pandas original read_csv does this, but cannot interpret datatime dtypes
    directly (has to use additional input parameter 'parse_dates'. This custom function
    automatically handles this from dtype dict supplied

    Parameters:
        filepath (str): Path to the CSV file.
        dtype_dict (dict): Dictionary mapping column names to data types.
        **kwargs: Additional keyword arguments to pass to `pd.read_csv`.

    Returns:
        pd.DataFrame: DataFrame with the specified column types.
    """
    # Separate datetime columns from other columns
    datetime_cols = [col for col, dtype in dtype_dict.items() if dtype == 'datetime64[ns]']
    other_dtypes = {col: dtype for col, dtype in dtype_dict.items() if dtype != 'datetime64[ns]'}

    # Read the CSV file
    df = pd.read_csv(
        filepath,
        dtype=other_dtypes,  # Apply non-datetime types
        parse_dates=datetime_cols,  # Parse datetime columns
        **kwargs  # Pass any additional arguments to read_csv
    )

    return df


def read_csv_or_create(csv_path,colnames):
    """
    check if fpath (pointing to csv) exists.
    if it does, load csv and return
    if it does not, create csv with colnames, and return df
    """
    if os.path.exists(csv_path):
        #print('Existing file loaded')
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


import pandas as pd

def slice_col_range(df, colname, range_str, dtype=None):
    """
    Slices a pandas DataFrame based on a column in a given range string.
    Works for numeric and datetime columns, with optional dtype specification.

    Parameters:
        df (pd.DataFrame): The DataFrame to slice.
        colname (str): The name of the column to slice.
        range_str (str): The range string, e.g., '(1,4)', '[1,4]', or "[2024-10-24, 2024-11-24]".
        dtype (str, optional): The type of the column ("numeric" or "datetime").
                               If None, the function will infer the type.

    Returns:
        pd.DataFrame: The sliced DataFrame.

    Raises:
        ValueError: If the range_str is not formatted correctly or dtype is invalid.
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

    # Helper function for type conversion
    def convert_type(value, dtype):
        if dtype == "numeric":
            return float(value)
        elif dtype == "datetime":
            return pd.to_datetime(value)
        else:
            raise ValueError("Invalid dtype. Use 'numeric' or 'datetime'.")

    # Extract the values from the range string
    try:
        range_str_inner = range_str[1:-1].strip()  # Remove outer brackets
        left_value, right_value = map(lambda x: x.strip().strip("'\""), range_str_inner.split(','))

        # Determine the conversion type
        if dtype is None:
            if pd.api.types.is_numeric_dtype(df[colname]):
                dtype = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(df[colname]):
                dtype = "datetime"
            else:
                raise ValueError(f"Column '{colname}' must be numeric or datetime for slicing.")

        # Convert values
        left_value = convert_type(left_value, dtype)
        right_value = convert_type(right_value, dtype)

    except Exception as e:
        raise ValueError(f"Error parsing range_str: {range_str}. Details: {e}")

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
    Combine dataframes occurring on the same date (day).

    Args:
        input_dict (dict): Dict of dataframes where keys are datetime strings, and values are dataframes.
        date_col (str): Name of the column to add, representing the shared date.

    Returns:
        dict: A dictionary where keys are dates (as strings), and values are combined dataframes.
    """
    grouped_datetimes = group_datetime_objects_by_date(input_dict.keys())
    combined_dataframes = {}

    for date, datetime_objects in grouped_datetimes.items():
        # Filter out empty or all-NA dataframes
        valid_dfs = [
            input_dict[dt]
            for dt in datetime_objects
            if not input_dict[dt].empty and not input_dict[dt].isna().all(axis=None)
        ]

        # Only concatenate if there are valid dataframes
        if valid_dfs:
            combined_df = pd.concat(valid_dfs)

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

