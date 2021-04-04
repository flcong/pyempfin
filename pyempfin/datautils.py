import numpy as np
import pandas as pd
from functools import partial
from typing import Union
from numba import njit



def winsor(datacol: pd.Series, lower: float, upper: float) -> pd.Series:
    """Winsorize the pandas series at the given percentiles

    Parameters
    ----------
    datacol : pandas.Series
    lower : float
        Lower percentile to winsorize. The value should be between 0 and 1
    upper : float
        Upper percentile to winsorize. The value should be between 0 and 1

    Returns
    -------
    pandas.Series
    """
    assert 0 <= lower <= 1, 'Lower percentile is not between 0 and 1'
    assert 0 <= upper <= 1, 'Upper percentile is not between 0 and 1'
    return datacol.clip(
        lower=np.nanquantile(datacol, q=lower),
        upper=np.nanquantile(datacol, q=upper)
    )


def sumstat(data: Union[pd.DataFrame,pd.Series], subset: Union[list,None]=None,
            percentiles: tuple=(.01, .05, .50, .95, .99)) -> pd.DataFrame:
    """Print summary statistics

    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        The data table to generate summary statistics
    subset : list of str, default: None
        The list of column names to generate summary statistics. If None, all
        columns are used.
    percentiles : tuple of float, default: (.01, .05, .25, .50, .75, .95, .99)
        The list of percentiles in the table.

    Returns
    -------
    pandas.DataFrame or pandas.Series
        The summary statistics. Each row represents one variable and each
        column represents one statistic.
    """
    funclist = [
                   lambda x: np.isfinite(x).sum(),
                   np.nanmean,
                   np.nanstd,
                   np.nanmin
               ] + [partial(lambda x, i: np.nanpercentile(x, q=i * 100), i=i)
                    for i in percentiles
                    ] + [np.nanmax]
    if isinstance(data, pd.DataFrame):
        if subset is None:
            subset = data.columns
        res = pd.DataFrame(
            [[func(data[x].to_numpy()) for func in funclist] for x in subset],
            index=subset,
            columns=[
                        'N', 'Mean', 'Std', 'Min'
                    ] + ['p' + str(int(i * 100)) for i in percentiles] + ['Max']
        )
        return res
    elif isinstance(data, pd.Series):
        res = pd.Series(
            [func(data.to_numpy()) for func in funclist],
            index=[
                        'N', 'Mean', 'Std', 'Min'
                    ] + ['p' + str(int(i * 100)) for i in percentiles] + ['Max']
        )
        return res
    else:
        raise ValueError('data is not of pd.DataFrame or pd.Series type!')



def ndup(data: pd.DataFrame, subset: Union[list,None]=None) -> int:
    """Number of duplicates over the subset of variables

    Parameters
    ----------
    data : pandas.DataFrame
    subset : list of str, default: None

    Returns
    -------
    int
    """
    # If subset=None, duplicated uses all columns by default
    return data.loc[lambda x: x.duplicated(subset=subset)].shape[0]


def getdup(data: pd.DataFrame, subset: Union[list,None]=None) -> pd.DataFrame:
    """Return rows of duplicated values over the subset of variables

    Parameters
    ----------
    data : pandas.DataFrame
    subset : list of str, default: None

    Returns
    -------
    pandas.DataFrame
    """
    return data.loc[lambda x: x.duplicated(subset=subset, keep=False)]


# Count missing values
def desmiss(data: pd.DataFrame, subset: Union[list,None]=None) -> pd.DataFrame:
    """Return a description of missing observations in the data set

    Parameters
    ----------
    data : pandas.DataFrame
    subset : list of str, default: None

    Returns
    -------
    pandas.DataFrame
    """
    if subset is None:
        subset = data.columns
    out = pd.DataFrame({
        '# NaN': data[subset].apply(lambda x: x.isna().sum())
    })
    out['% NaN'] = out['# NaN'].apply(
        lambda x: '{:.2f}%'.format(x / data.shape[0] * 100)
    )
    out = out.sort_values('# NaN', ascending=False)
    return out


# Mimic the coalesce function in SAS
def coalesce(*argv) -> pd.Series:
    """Mimic the coalesce function in SAS.

    Parameters
    ----------
    *argv : pandas.Series or a number
        Each argument is a pandas Series or a number.

    Returns
    -------
    pandas.Series
    """
    narg = len(argv)
    out = argv[0]
    for i in range(narg - 1):
        # Construct a constant series if a number is passed
        if isinstance(argv[i + 1], pd.Series):
            out = out.combine_first(argv[i + 1])
        else:
            out = out.combine_first(
                pd.Series([argv[i + 1]] * out.shape[0], index=out.index)
            )
    return out


@njit
def wmean(data: np.ndarray, weights: np.ndarray) -> float:
    """Weighted average

    Parameters
    ----------
    data : numpy.ndarray
    weights : numpy.ndarray

    Returns
    -------
    float
    """
    assert data.ndim == 1, 'data is not 1-d array'
    assert weights.ndim == 1, 'weights is not 1-d array'
    assert data.shape[0] == weights.shape[0], \
        'data and weights are not of the same shape'
    numer = 0
    denom = 0
    for i in range(len(data)):
        if np.isfinite(data[i]) and np.isfinite(weights[i]):
            numer += data[i] * weights[i]
            denom += weights[i]
    if denom > 0:
        return numer / denom
    else:
        return np.nan
