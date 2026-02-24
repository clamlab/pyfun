import fnmatch
import pandas as pd
import datetime

time_formats_default = {"????-??-??T??_??_??": "%Y-%m-%dT%H_%M_%S",
                        "????-??-??_??-??-??": "%Y-%m-%d_%H-%M-%S",
                        "????_??_??-??_??":    "%Y_%m_%d-%H_%M"}


BONSAI_TIMESTAMP_FMT = "%H:%M:%S.%f"

#TODO: account for some time formats being substrings of other time formats



def calc_dt(pd_t2, pd_t1, output='millisecs'):
    """
    takes two pandas timestamps and returns the timedelta (t2 - t1)
    """
    if pd_t2 is None or pd_t1 is None:
        return None

    if output == 'millisecs':
        dt = (pd_t2 - pd_t1).total_seconds() * 1000
    elif output == 'secs':
        dt = (pd_t2 - pd_t1).total_seconds()
    else:
        raise ValueError('Unrecognized output format.')

    return dt


def calc_dt_col(df, timestamp_col, dt_col_name, ref_t, drop=True, output='millisecs'):
    """
    calculate the difference in time between timestamp values in a df column, and a reference timestamp,
    using the calc_dt function.
    TODO: make timestamp format specification more general
    """
    df[dt_col_name] = df[timestamp_col].apply(lambda t: calc_dt(t, ref_t, output=output))

    if drop:  # drop timestamp_col
        df = df.drop(timestamp_col, axis=1)

    return df

def compare_same_date(dt1, dt2):
    """
    Compare two pandas datetime objects and return the common date as a string if they share the same date.

    Args:
        dt1 (pd.Timestamp): The first pandas datetime object.
        dt2 (pd.Timestamp): The second pandas datetime object.

    Returns:
        str: The common date in the format 'YYYY-MM-DD' if the input datetime objects share the same date.
        None: If the input datetime objects do not share the same date.
    """
    date1 = dt1.date()
    date2 = dt2.date()

    if date1 == date2:
        return date1.strftime('%Y-%m-%d')
    else:
        return None

def date_to_filename(fn, subj_name=None):
    """
    add today's date to filename string (fn)
    the fn already contains full extension e.g. "abc.pdf
    optionally specify subj name that is inserted
    """
    current_date = datetime.datetime.now()
    formatted_date = current_date.strftime('%m%d%y')

    if subj_name is None:
        subj_name  = ''
    else:
        subj_name += '_'

    fn2 = fn.split('.')[0] + '_' + subj_name + formatted_date + '.' + fn.split('.')[1]

    return fn2


def filter_by_date(data, target_date):
    """
    Filter a dictionary with pandas datetime keys
    to only include entries from a specific date.

    :param data: Dictionary with keys as pandas datetime objects.
    :param target_date: Desired date string in the format 'YYYY-MM-DD'.
    :return: Filtered dictionary with only the entries from the target date.
    """
    data_dt = {search(key)[1]: value for key, value in data.items()}
    target_date_obj = pd.to_datetime(target_date)
    return {key: value for key, value in data_dt.items() if key.date() == target_date_obj.date()}


def parse_time(time_str, formats = ["%H:%M:%S.%f", "%H:%M:%S"]):
    # TODO due to some bonsai idiocy, sometimes the values are rounded off,
    # and this throws an error in pd.to_datetime this is why two formats are provided

    parsed_time = pd.NaT

    for fmt in formats:
        parsed_time = pd.to_datetime(time_str, format=fmt, errors="coerce")
        if not pd.isna(parsed_time):
            break
        else:
            return None
    return parsed_time


def parse_time_col(time_series,  formats=["%H:%M:%S.%f", "%H:%M:%S"]):
    # substitute for pd.to_datetime, specifically because of one observed exception case where
    # bonsai timestamp was %H:%M:%S without the decimals (likely would be .0000000)

    parsed_time = pd.Series(pd.NaT, index=time_series.index)

    for fmt in formats:
        # Find the indices of remaining failed conversions
        failed_indices = pd.isna(parsed_time)

        # If there are no more failed conversions, break the loop
        if not failed_indices.any():
            break

        # Attempt to parse the failed conversions using the current format
        parsed_time[failed_indices] = pd.to_datetime(time_series[failed_indices], format=fmt, errors="coerce")

    return parsed_time



def search(input_str, output_mode='dt_and_string', time_formats=time_formats_default, verbose=True):
    """
    search within a string, for a matching time string embedded within

    :param input_str: string with datetime
    :param output_mode: give output as datetime object or as a string
    :param time_formats: dictionary of templates to search for

    """

    for k, v in time_formats.items():

        n = len(k)

        timestr, timefmt = None, None
        for i in range(len(input_str) - n+1):

            s = input_str[i:i + n]

            res = fnmatch.filter([s], k)
            if len(res) == 1:
                timestr = res[0]
                timefmt = v
                break

        if timestr is not None:
            break




    if timestr is None:
        if verbose==True:
            print('No time string found')
        return None
    else:
        dt = pd.to_datetime(timestr, format=timefmt)

        if output_mode=='dt_obj':
            return dt
        elif output_mode=='dt_and_string':
            return timestr, dt
        elif output_mode=='strings':
            return timestr, timefmt

