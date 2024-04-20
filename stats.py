import scipy.stats
import numpy as np

def linregress(x, y):
    res = scipy.stats.linregress(x, y)

    y_hat = res.slope * x + res.intercept
    y_hat_err = y_hat - y

    return y_hat, y_hat_err, res.slope, res.intercept


def select_averager(measure):
    if measure == 'mean':
        func = np.mean
    elif measure == 'median':
        func = np.median
    elif measure == 'identity':
        func = lambda x: x #identity function
    else:
        raise ValueError('Unknown measure. Do you want mean, median or identity?')

    return func


# === basic functions for computing distances between pairs of cartesian points ===
# A and B are n x 2 arrays where each row is an x,y coordinate pair

def euclid_distance(A, B):
    return np.linalg.norm(A - B, axis=1)

def distance_x(A, B):
    return np.abs(A[:,0] - B[:,0])

def distance_y(A, B):
    return np.abs(A[:,1] - B[:,1])

def corr_x(A, B):
    return np.corrcoef(A[:, 0], B[:, 0])[0, 1]

def corr_y(A, B):
    return np.corrcoef(A[:, 1], B[:, 1])[0, 1]
