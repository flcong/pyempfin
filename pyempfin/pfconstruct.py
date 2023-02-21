import numpy as np
import pandas as pd
from numba import njit


def get_port_ret(data,
                 nq,
                 timevar,
                 retvar,
                 rnkvars,
                 rnkvarnames,
                 wvar=None,
                 dep=False,
                 retfull=False):
    """
    Construct equal-weighted and/or value-weighted portfolio returns sorted on different variables,
        dependently or independently, from a DataFrame containing panel returns of securities for
        different times.

    Parameters
    ----------
    data : DataFrame
        A panel (long-type or stacked) data of returns for different securities at different times.
    nq : int or list of int
        Number(s) of portfolios for each rank variable. The order should be consistent with `rnkvars`.
        If it is an integer, the number is applied for all rank variables.
    timevar : str
        Name of the variable representing time. This variable determines the time variable in the output.
    retvar : str
        Name of the variable representing security returns.
    rnkvars : list of str
        Name(s) of variables to rank on.
    rnkvarnames : list of str
        Name(s) of the created variable representing different portfolios
    wvar : str or None, default None
        Name of the variable representing portfolio weights. If it is None, only equal-weighted returns are calcualted
    dep : bool, default False
        Whether or not variable sorts are dependent. If `dep=True`, the order of variable names in `rnkvars`
        indicates the order of dependent sorting.
    retfull : bool, default False
        If True, a tuple will be returned from the function. The first element contains portfolios while the second
        one contains the full input data set with rank variables.

    Returns:
    ----------
    output : DataFrame
        A DataFrame containing a time-series of returns for different portfolios
    data : DataFrame
        The input data adding portfolio numbers
    """
    data = data.copy()

    # If nq is an integer
    if np.ndim(nq) == 0:
        nq = [nq] * len(rnkvars)
    # Remove NaN
    data = data.dropna(subset=rnkvars+[retvar], how='any')
    # Generate group-by variable list
    by_list = [[timevar]]
    if dep:
        for v in rnkvarnames:
            by_list.append(by_list[-1] + [v])
        del by_list[len(rnkvars)]
    else:
        by_list = by_list * len(rnkvars)

    # For each sort variable
    for i in range(len(by_list)):
        rnkvar = rnkvars[i]
        rnkvarname = rnkvarnames[i]
        data = data.groupby(by_list[i]).apply(
            lambda x: _get_qcut(x, nq[i], rnkvar, rnkvarname)
        ).reset_index(drop=True)

    # Calculate equal-weighted and value-weighted portfolio returns
    grouped = data.groupby([timevar] + rnkvarnames)
    if wvar is None:
        out = grouped[retvar].agg('mean').to_frame('ew_ret')
    else:
        out = pd.concat([
            grouped[retvar].agg('mean'),
            grouped.apply(lambda x: np.average(x[retvar], weights=x[wvar]))
            ], axis=1, keys=['ew_ret', 'vw_ret']) \
            .join(grouped[retvar].count().rename('cnt'), how='left') \
            .reset_index()

    if retfull:
        return out, data
    else:
        return out

def _get_qcut(x, q, rnkvar, rnkvarname):
    """
    Similar to pd.qcut, but allow for closed interval and duplicate observations
    with multiple closed interval matched. This function is to be used in a
    `pandas.groupby()` function.

    Parameters
    ----------
    x : DataFrame
        The input array, typically a dataframe or chunk of a dataframe split by a `groupby()` function
    q : int
        Number of portfolios (e.g. 5 for quintile)
    rnkvar : str
        Name of the variable to sort on
    rnkvarname : str
        Name of the created variable representing different portfolios

    Returns
    -------
    out : DataFrame
        The output DataFrame is the same as the input `x` except that:

        * A new variable named `rnkvarname` is added representing intervals (if `retinterval=True`)
        or portfolio numbers (if `retinterval=False`).
        * Observations (rows) corresponding to multiple intervals are duplicated with the multiple intervals
        (or numbers) assigned to `rnkvarname`.

    """
    # Quantiles
    quantiles = np.linspace(0, 1, q + 1)
    bins = x[rnkvar].quantile(quantiles).values
    # Adjust the first and last endpoint
    bins[0] = bins[0] - (bins[1] - bins[0]) * 1e-2
    bins[-1] = bins[-1] + (bins[-1] - bins[-2]) * 1e-2
    # Intervals (seems to be unnecessary to use pd.Interval, which is slow)
    intervals = np.vstack((bins[:-1], bins[1:])).T
    interval = intervals[0]

    x_dup = pd.concat([_get_matched_data(intervals[i, :], i, x, rnkvar) for i in range(intervals.shape[0])], axis=0)
    # Drop variables
    x_dup = x_dup.drop(columns=['__interval_l__', '__interval_r__']).rename(columns={'__portnumber__': rnkvarname})
    return x_dup


# For each interval, find all obs (boolean array) whose rnkvar is in the interval
def _get_matched_data(interval, intervalno, x, rnkvar):
    # Get all obs where rnkvar is within an interval
    x_tmp = x.loc[lambda y: (interval[0] <= y[rnkvar].values) & (y[rnkvar].values <= interval[1])].copy()
    # Assign interval and number
    x_tmp['__interval_l__'] = interval[0]
    x_tmp['__interval_r__'] = interval[1]
    x_tmp['__portnumber__'] = intervalno + 1

    return x_tmp
