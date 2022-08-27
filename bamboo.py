"""
misc functions for wrangling pandas data
"""

import pandas as pd

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


def chainslice(df, slice_instructions):
    """ perform a series of slicing operations using slice_df
     slice_instructions: nested list, each element [col, [vals], polarity] """

    for [col, vals, polarity] in slice_instructions:
        df = slice(df, {col: vals}, polarity)

    return df

def find_row_match(search_row, cols_to_match, df_haystack, find_single=True):
    """
    given row in one df, find matching rows in another df based on some columns
    :param search_row: #TODO complete docs
    :param cols_to_match:
    :param df_haystack:
    :return:
    """
    m = []  # define search instructions
    for c in cols_to_match:
        m.append([c, [search_row[c]], '+'])

    matched_rows = chainslice(df_haystack, m)

    if len(matched_rows) == 0:
        print('No matches found')
        return None
    elif len(matched_rows) > 1:
        if find_single:
            print(search_row)
            print(matched_rows)
            raise ValueError('More than one matching row found')
        else:
            return matched_rows[0]
    elif len(matched_rows) == 1:
        return matched_rows


def match_dfs(df1, df2, label1, label2, cols_to_match):
    """
    find df1 within df2
    :param df1:
    :param df2:
    :param label1:
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
        i2 = row2.index[0]

        df1.loc[i1, df1_col] = i2
        df2.loc[i2, df2_col] = i1


def slice_notnull(df, col):
    return df[df[col].notnull()]


def slice(df, col_row_vals, polarity='+', print_counts=False):
    """ slice df by finding matching row values in a given cols
    col_row_vals: a dict of {colname1: [value1, value2...],
                             colname2: [value3, value4...]}
    polarity = + or -, returns matching or nonmatching
    returns a copy, not a view
    """

    df_small = df.copy()
    if polarity == '+':
        for k, v in col_row_vals.items():
            df_small = df_small[df_small[k].isin(v)]

    elif polarity == '-':
        for k, v in col_row_vals.items():
            df_small = df_small[~df_small[k].isin(v)]
    else:
        raise ValueError('Matching logic not found.')


    if print_counts:
        for col in col_row_vals:
            print(col)
            print(df_small[col].value_counts())

    return df_small

def slice_col_range(df, col, val_range):
    return df[df[col].between(val_range[0], val_range[1])]