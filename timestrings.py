import fnmatch
import pandas as pd


time_formats_default = {"????-??-??T??_??_??": "%Y-%m-%dT%H_%M_%S",
                        "????-??-??_??-??-??": "%Y-%m-%d_%H-%M-%S",
                        "????_??_??-??_??":    "%Y_%m_%d-%H_%M"}
#TODO: account for some time formats being substrings of other time formats


def search(input_str, output_mode='strings', time_formats=time_formats_default):
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
    else:
        if output_mode=='dt_obj':
            return pd.to_datetime(timestr, format=timefmt)
        elif output_mode=='strings':
            return timestr, timefmt

