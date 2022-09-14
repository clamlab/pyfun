import scipy.stats

def linregress(x, y):
    res = scipy.stats.linregress(x, y)

    y_hat = res.slope * x + res.intercept
    y_hat_err = y_hat - y

    return y_hat, y_hat_err, res.slope, res.intercept