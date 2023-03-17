import fnmatch
import pandas as pd


time_formats_default = {"????-??-??T??_??_??": "%Y-%m-%dT%H_%M_%S",
                        "????-??-??_??-??-??": "%Y-%m-%d_%H-%M-%S",
                        "????_??_??-??_??":    "%Y_%m_%d-%H_%M"}
#TODO: account for some time formats being substrings of other time formats


import pandas as pd



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


def search(input_str, output_mode='dt_and_string', time_formats=time_formats_default):
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

