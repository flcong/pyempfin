import numpy as np
import pandas as pd
from numba import njit
from joblib import Parallel, delayed, cpu_count
from scipy.stats import t as tdist
from typing import Union, Optional, Tuple, List, Callable
import re
from functools import partial
from statsmodels.regression.linear_model import OLS, OLSResults
from statsmodels.stats.sandwich_covariance import cov_hac
import scipy.stats
from numba import njit, int32, float64
import warnings
from .datautils import groupby_apply


# =============================================================================#
# Functions to estimate beta
# =============================================================================#

def estbeta(leftdata: pd.DataFrame, rightdata: pd.DataFrame, models: dict,
            window: tuple, minobs: int, hasconst: bool, ncore: int=cpu_count()
            ) -> pd.DataFrame:
    """Estimate beta given a list of models. leftdata is a pandas.DataFrame of
    dependent variables (returns) with MultiIndex representing assets and periods.
    rightdata is a pandas.DataFrame with index representing periods and columns
    of independent variables (factors) to estimate beta.

    Parameters:
    ----------
    leftdata : pandas.DataFrame
        The index should be a MultiIndex whose level 0 represents assets and
        level 1 represents time periods
    rightdata : pandas.DataFrame
        The index should be a MultiIndex whose level 0 represents assets and
        level 1 represents time periods. This DataFrame will be reindexed
        according to `leftdata`
    model : dict (keys: str, values: list of str)
        The keys are model names (used for referencing and output) and the
        corresponding values are lists of names of variables in the model. The
        first element represents the dependent variable and should be in the
        columns of `leftdata`. The rest represents independent variables and
        should be in the columns of `rightdata`.
    window : tuple of 2 integers
        The period used to estimate beta. For example, `(-24,-1)` means using
        data between `t-24` and `t-1` to estimate beta at time `t`.
    minobs : int
        Minimum number of observations to estimate beta
    hasconst : bool
        If True, a constant term is added to the regression.
    ncore : int, default `cpu_count()`
        The number of cores used for parallel computing. The default value is
        the total number of cores in the computer.

    Returns
    ----------
    output : DataFrame
        Its index is the same as that of `leftdata`. Its column is also a
        MultiIndex, whose number of levels depending on whether the columns
        of `rightdata` is a MultiIndex. Its level 0 represents different models
        (keys in `models`). Its other levels represent the betas for each
        factors (values in `models`) with the same column names as the factor.
        Hence, if the columns of `rightdata` is not a MultiIndex, then the columns
        of the returned dataframe have only 2 levels. If the columns of `rightdata`
        is a MultiIndex with 2 levels, then the columns of the returned dataframe
        have 3 levels.
    """
    outbeta = pd.concat([
        estbeta1m(
            leftdata=leftdata[x[0]],
            rightdata=rightdata[x[1:]],
            model=x[1:],
            window=window,
            minobs=minobs,
            hasconst=hasconst,
            ncore=ncore
        ) for x in models.values()
    ], axis=1, keys=models.keys(), join='outer'
    )
    return outbeta

def estbeta1m(leftdata: pd.Series, rightdata: pd.DataFrame, model: list,
              window: tuple, minobs: int, hasconst: bool, ncore: int=cpu_count()
              ) -> pd.DataFrame:
    """Estimate beta for a single model. leftdata is a pandas.Series of
    dependent variable (returns) with MultiIndex representing assets and periods.
    rightdata is a pandas.DataFrame with index representing periods and columns
    of independent variables (factors) to estimate beta.

    Parameters
    ----------
    leftdata : pandas.Series
        The index should be a MultiIndex whose level 0 represents assets and
        level 1 represents time periods
    rightdata : pandas.DataFrame
        The index represents time periods and is the same as the level 1 of
        the index of `leftdata`.
    model : list
        The elements should be a subset of column names of rightdata.
    window : tuple of 2 integers
        The period used to estimate beta. For example, `(-24,-1)` means using
        data between `t-24` and `t-1` (both inclusive) to estimate beta at
        time `t`. The first element cannot be greater than the second one
    minobs : int
        Minimum number of observations to estimate beta
    hasconst : bool
        If True, a constant term is added to the regression.
    ncore: int
        Number of cores for parallel computing

    Returns
    -------
    pandas.DataFrame
        The index is the same as `leftdata`. Each column represents the estimated
        beta.
    """
    # Check arguments
    if not isinstance(leftdata, pd.Series):
        raise TypeError('leftdata is not a pandas.Series')
    if not isinstance(rightdata, pd.DataFrame):
        raise TypeError('rightdata is not a pandas.DataFrame')
    if not set(model).issubset(set(rightdata.columns)):
        raise ValueError('elements of model is not a subset of columns of rightdata')
    if not len(window) == 2:
        raise ValueError('window does not have length 2')
    if not window[0] <= window[1]:
        raise ValueError('start time is after end time for window')
    # Drop NaN and construct matrix for return
    leftmat = leftdata.dropna().unstack(level=0).sort_index()
    rightdata = rightdata.dropna(how='all').reindex(leftmat.index)
    # Construct X
    exog = rightdata[model].to_numpy()
    out = pd.DataFrame(
        np.concatenate(
            Parallel(n_jobs=ncore)(
                delayed(
                    lambda x: _get_beta(
                        x, exog, window, minobs, hasconst)
                )(leftmat[v].to_numpy())
                for v in leftmat.columns), axis=1
        ),
        index=leftmat.index,
        columns=pd.MultiIndex.from_product([leftmat.columns, np.arange(len(model))])
    )
    # If model is a list of tuples (so rightdata has MultiIndex column)
    out = out.stack(level=0)
    if isinstance(rightdata.columns, pd.MultiIndex):
        out = out.set_axis(pd.MultiIndex.from_tuples(model), axis='columns')
    else:
        out = out.set_axis(model, axis='columns')
    out = out.swaplevel(i=1, j=0, axis='index') \
        .sort_index()
    return out



def _get_beta(endogmat: np.ndarray, exogmat: np.ndarray, window: tuple,
                   minobs: int, hasconst: bool
                   ) -> np.ndarray:
    """Calculate rolling-window beta

    Parameters
    ----------
    endogmat : 1-D numpy.ndarray
        A vector of returns indexed by time.
    exogmat : 2-D numpy.ndarray
        T-by-K matrix, Rows represent time periods and columns represent
        independent variables.
    window : tuple of 2 integers
        The period used to estimate beta. For example, `(-24,-1)` means using
        data between `t-24` and `t-1` to estimate beta at time `t`.
    minobs : int
        Minimum number of observations to estimate beta
    hasconst : bool
        If True, a constant term is added to the regression.

    Returns
    -------
    2-D numpy.ndarray
        T-by-K matrix. Rows represent time periods. Each column
        represent beta for each factor. The intercept is not returned.
    """
    n_factors = exogmat.shape[1]
    if hasconst:
        exogmat = np.hstack((exogmat, np.ones((exogmat.shape[0], 1))))
    return get_rolling_stats_njit(
        endogmat, exogmat, window, minobs,
        _estimate_beta_njit, (hasconst,), n_factors,
    )


@njit
def get_rolling_stats_njit(
        endogmat: np.ndarray, exogmat: np.ndarray, window: tuple, minobs: int,
        func: Callable, func_args: tuple, func_retsize: int) -> np.ndarray:
    """Calculate rolling-window statistics

    Parameters
    ----------
    endogmat : 1-D numpy.ndarray
        A vector of returns indexed by time.
    exogmat : 2-D numpy.ndarray
        T-by-K matrix, Rows represent time periods and columns represent
        independent variables.
    window : tuple of 2 integers
        The period used to calculate statistics. For example, `(-24,-1)` means using
        data between `t-24` and `t-1` to calculate statistics at time `t`.
    minobs : int
        Minimum number of observations to calculate statistics
    func : Callable
        Function to be applied for each period to calculate statistics
        in the rolling window. The function has signature (endogmat,exogmat,*args).
        The function returns a 1D vector with size func_retsize
    func_args : tuple
        Other arguments to be passed to func.
    func_retsize : int
        Size of the returned vector

    Returns
    -------
    2-D numpy.ndarray
        T-by-K matrix. Rows represent time periods. Each column
        represent beta for each factor. The intercept is not returned.
    """
    # Number of periods, variables,
    nper = endogmat.shape[0]
    # Initialization
    outstats = np.zeros((nper, func_retsize))
    outstats.fill(np.nan)
    # Loop
    for t in range(-window[0], nper):
        # Note that right boundary is not included in array slicing
        endogtmp = endogmat[t+window[0]:t+window[1]+1]
        exogtmp = exogmat[t+window[0]:t+window[1]+1, :]
        # Generate indicator
        nnanmask = np.isfinite(endogtmp)
        for j in range(window[1]-window[0]+1):
            nnanmask[j] = np.isfinite(exogtmp[j, :]).all() * nnanmask[j]
        X = exogtmp[nnanmask, :]
        Y = endogtmp[nnanmask]
        XX = X.T @ X
        if (len(Y) >= minobs) and (np.linalg.matrix_rank(XX) == X.shape[1]):
            outstats[t, :] = func(Y, X, *func_args)
    return outstats


@njit
def _estimate_beta_njit(endog_not_null, exog_not_null, hasconst):
    if hasconst:
        return _estimate_ols_coefs_njit(endog_not_null, exog_not_null)[:-1]
    else:
        return _estimate_ols_coefs_njit(endog_not_null, exog_not_null)


@njit
def _estimate_ols_coefs_njit(endog_not_null, exog_not_null):
    X, Y = exog_not_null, np.ascontiguousarray(endog_not_null)
    XX = X.T @ X
    return np.linalg.inv(XX) @ X.T @ Y


@njit
def njit_get_condition_number(endog, exog):
    # REF: Salmerón, R., García, C. B., & García, J. (2018). Variance Inflation
    # Factor and Condition Number in multiple linear regression. Journal of
    # Statistical Computation and Simulation, 88(12), 2365–2384.
    # "Standardize" columns of exog to have unit length
    exog_ = np.zeros(exog.shape)
    exog_.fill(np.nan)
    for i in range(exog.shape[1]):
        exog_[:, i] = exog[:, i] / np.linalg.norm(exog[:, i])
    # Calculate eigenvalues and condition number
    eigvals = np.linalg.eigvalsh(exog_.T @ exog_)
    eigvals = np.sort(eigvals)[::-1]
    return np.sqrt(eigvals[0]/eigvals[-1])


@njit
def njit_get_vifs(exog):
    k_vars = exog.shape[1]
    vifs = np.zeros(k_vars)
    vifs.fill(np.nan)
    for i in range(k_vars):
        x_i = exog[:, i]
        mask = np.arange(k_vars) != i
        x_noti = exog[:, mask]
        r2 = _njit_get_rsquared(x_i, x_noti)
        vifs[i] = 1. / (1. - r2)
    return vifs


@njit
def _check_has_const(arr):
    for i in range(arr.shape[1]):
        if np.max(arr[:,i]) == np.min(arr[:,i]):
            return True
    return False


@njit
def _njit_get_rsquared(endog, exog):
    coefs = _estimate_ols_coefs_njit(endog, exog)
    resid = endog - exog @ coefs
    ssr = resid.dot(resid)
    if _check_has_const(exog):
        endog_ = endog - endog.mean()
        return 1- ssr / endog_.dot(endog_)
    else:
        return 1 - ssr / endog.dot(endog)



# =============================================================================#
# Fama-MacBeth Regression
# =============================================================================#

def fmreg(leftdata: pd.DataFrame, rightdata: pd.DataFrame, models: list,
        maxlag: int, roworder: list, hasconst: bool, scale: float,
        getlambda: bool, winsorcuts: Union[tuple,None]=None,
        winsorindeponly: bool=True, estfmt: tuple=('.3f', '.2f')
          ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
    """Estimate FM regression for one model specification.

    Parameters
    ----------
    leftdata : pd.DataFrame
        Dataframe containing dependent variables. Its index is a MultiIndex,
        whose level 0 represents entities and its level 1 represents time
        periods.
    rightdata : pd.DataFrame
        Dataframe containing independent variables. Its index is a MultiIndex,
        whose level 0 represents entities and its level 1 represents time
        periods. The columns can be either a index or a MultiIndex with 2 levels.
    models : list of list
        A list of models. Each element represents a model containing names.
        For each model, the elements can be used to select columns in
        leftdata and rightdata, even when their columns are MultiIndex. The
        first element represents the dependent variable and the others
        represent independent variables
    maxlag : int
        Maximum number of lags for Newey-West estimator
    roworder : list of str
        The order of rows in the output. If the columns of `rightdata` is a
        MultiIndex with 2 levels, then `roworder` should match level 1 of the
        MultiIndex.
    hasconst : bool
        If True, a constant term is added in the model.
    scale : float
        The estimated coefficients will be multipled by this number.
    winsorcuts : tuple or None
        A tuple of lower and upper percentiles to winsorize variables in each
        cross-sectional regression. Both should be between 0 and 1 and the
        lower one should be less than the upper one.
    winsorindeponly : bool
        If True, only independent variables are winsorized.
    estfmt : tuple
        A 2-D tuple of str specifying the format of the coefficient and tstat,
        respectively.
    getlambda : bool
        If True,
    Return
    ------
    pd.DataFrame
        Formatted regression table
    pd.DataFrame, pd.Series
        If getlambda=True, then also return estimated lambda in a pandas.Series
    """
    # Check arguments
    if not isinstance(leftdata, pd.DataFrame):
        raise TypeError('leftdata is not a pandas.DataFrame')
    if not isinstance(rightdata, pd.DataFrame):
        raise TypeError('rightdata is not a pandas.DataFrame')
    if isinstance(leftdata.columns, pd.MultiIndex):
        if not leftdata.columns.nlevels == 2:
            raise ValueError('if leftdata has a MultiIndex column names, '
                             'its level must be 2')
    else:
        if not isinstance(leftdata.columns, pd.Index):
            raise ValueError('if leftdata does not have a MultiIndex column '
                             'names, it must be a pandas.Index')
    if isinstance(rightdata.columns, pd.MultiIndex):
        if not rightdata.columns.nlevels == 2:
            raise ValueError('if rightdata has a MultiIndex column names, '
                             'its level must be 2')
        if not set(roworder).issubset(set(rightdata.columns.levels[1])):
            raise ValueError('roworder must be a subset of level 1 of columns '
                             'of rightdata if its column names is a MultiIndex')
    else:
        if not isinstance(rightdata.columns, pd.Index):
            raise ValueError('if rightdata does not have a MultiIndex column '
                             'names, it must be a pandas.Index')
        if not set(roworder).issubset(set(rightdata.columns)):
            raise ValueError('roworder must be a subset of columns of rightdata')
    if not isinstance(hasconst, bool):
        raise ValueError('hasconst must be a boolean')
    if winsorcuts is not None:
        if not isinstance(winsorcuts, tuple):
            raise ValueError('winsorcuts must be a tuple or None')
        if not (len(winsorcuts) == 2):
            raise ValueError('winsorcuts must be a tuple of length 2')
        if not (0<=winsorcuts[0]<=1):
            raise ValueError('winsorcuts[0] must be between 0 and 1')
        if not (0<=winsorcuts[1]<=1):
            raise ValueError('winsorcuts[1] must be between 0 and 1')
        if not (winsorcuts[0]<winsorcuts[1]):
            raise ValueError('winsorcuts[0] must be less than winsorcuts[1]')
    if not isinstance(winsorindeponly, bool):
        raise ValueError('winsorindeponly must be a boolean')
    if not (isinstance(estfmt, tuple) and len(estfmt) == 2):
        raise ValueError('estfmt must be a tuple with length 2')
    for s in estfmt:
        if not isinstance(s, str):
            raise ValueError(f'{s} in estfmt is not a str')

    estout = FMResult()
    for model in models:
        # Check arguments
        if not (isinstance(model, list) and len(model) >= 2):
            raise ValueError(f'The model ({model}) is not a list of length 2')
        if model[0] not in leftdata.columns:
            raise ValueError(f'model[0] of the model ({model}) not in columns'
                             f' of leftdata')
        if not set(model[1:]).issubset(set(rightdata.columns.to_flat_index())):
            raise ValueError(f'model[1:] of the model ({model}) not in columns'
                             f' of rightdata')
        # ---------- Prepare LHS data set
        if isinstance(leftdata.columns, pd.MultiIndex):
            # Note that model[0] is not a list, so double brackets ensure that a
            # dataframe is returned
            leftdatatmp = leftdata[[model[0]]]
            depvar = leftdatatmp.columns.get_level_values(1)[0]
            leftdatatmp.columns = [depvar]
        else:
            depvar = model[0]
            leftdatatmp = leftdata[[model[0]]]
            # No need to assign leftdata.columns here
        # ---------- Prepare RHS data set
        rightdatatmp = rightdata.reindex(leftdata.index)[model[1:]]
        if isinstance(rightdata.columns, pd.MultiIndex):
            # Note that model[1:] is a list, so single brackets are enough to
            # return a dataframe
            # This is a MultiIndex
            indepvars = list(rightdatatmp.columns.get_level_values(1))
            rightdatatmp.columns = indepvars
        else:
            indepvars = model[1:]
            # No need to assign rightdata.columns here
        # Add Constant
        if hasconst:
            indepvars += ['Constant']
        # Get entity and time variable names
        indvar = leftdatatmp.index.levels[0].name
        timevar = leftdatatmp.index.levels[1].name
        # ----------- Convert dataframe into matrix form
        # Get matrix form of dependent variable
        endogmat = leftdatatmp.reset_index().pivot(
            index=timevar,
            columns=indvar,
            values=leftdatatmp.columns[0]
        ).to_numpy()
        # Get matrix form of independent variable
        exogmat = np.array([rightdatatmp[[v]].reset_index().pivot(
            index=timevar,
            columns=indvar,
            values=v
        ) for v in rightdatatmp.columns])
        # Swap axis
        exogmat = exogmat.swapaxes(0, 1)
        # ----------- Fama-MacBeth Regression
        # Get estimates
        coef, stderr, nobs, nobsavg, nper, ar2 = _fmreg_njit(
            endogmat,
            exogmat,
            maxlag,
            hasconst,
            winsorcuts,
            winsorindeponly
        )
        # ----------- Format output
        estout.add(coef, stderr, indepvars, nper-1,
                   stat={
                       'N': nobs,
                       'Periods': nper,
                       'N/period': nobsavg,
                       'Adjusted $R^2$': ar2
                   })
    estouttab = estout.output(
        type='tstat',
        estfmt=estfmt,
        statfmt=(',.0f', ',.0f', ',.1f', '.3f'),
        roworder=roworder + ['Constant'] if hasconst else roworder,
        scale=scale
    )
    if not getlambda:
        return estouttab
    else:
        outlambda = pd.concat([
            pd.Series({v.label: v.param for v in eststo['est']})
            for eststo in estout.eststo
        ], axis=1)
        outlambda.reindex(roworder + ['Constant'] if hasconst else roworder)
        return estouttab, outlambda


@njit
def _fmreg_njit(endogmat, exogmat, maxlag, hasconst, winsorcuts, winsorindeponly):
    # Verify parameters
    assert endogmat.ndim == 2, 'The number of dimensions of endog is not 2'
    assert exogmat.ndim == 3, 'The number of dimensions of endog is not 3'
    # Degress of freedom
    if hasconst:
        K = exogmat.shape[1] + 1
    else:
        K = exogmat.shape[1]
    # Total number of periods (including periods with missing variables)
    nper = exogmat.shape[0]
    # ----------- First stage
    # Lambdas
    lambdas = np.zeros((nper, K))
    lambdas.fill(np.nan)
    # Adjusted R2
    ar2 = np.zeros(nper)
    ar2.fill(np.nan)
    # Number of observations
    nobs = np.zeros(nper)
    nobs.fill(np.nan)
    # Loop over each row
    for t in range(nper):
        exogtmp = exogmat[t, :].T
        endogtmp = endogmat[t, :].T
        # Get nnanmask
        nnanmask = np.isfinite(endogtmp)
        for i in range(endogtmp.shape[0]):
            nnanmask[i] = nnanmask[i] * np.isfinite(exogtmp[i, :]).all()
        if hasconst:
            X = np.hstack((exogtmp, np.ones((endogtmp.shape[0], 1))))
        else:
            X = exogtmp
        X = X[nnanmask, :]
        Y = endogtmp[nnanmask]
        if X.size > 0 and Y.size > 0:
            # Winsor variables
            if winsorcuts is not None:
                for i in range(K):
                    X[:, i] = winsor_njit(X[:, i], winsorcuts, 'inner')
                if not winsorindeponly:
                    Y = winsor_njit(Y, winsorcuts, 'inner')
            XX = X.T @ X
            if X.shape[0] > X.shape[1] and np.linalg.matrix_rank(XX) == X.shape[1]:
                beta = np.linalg.inv(XX) @ X.T @ Y
                resid = Y - X @ beta
                tss = (Y - Y.mean()).dot(Y - Y.mean())
                rss = resid.dot(resid)
                ar2[t] = 1 - rss / tss * (Y.shape[0] - 1*hasconst) / (Y.shape[0] - K)
                lambdas[t, :] = beta.T
                nobs[t] = Y.shape[0]
    # ----------- Second stage
    # Calculate mean
    coef = np.zeros(K)
    for i in range(K):
        coef[i] = np.nanmean(lambdas[:, i])
    # Calculate standard error
    stderr = np.zeros(K)
    for i in range(K):
        stderr[i] = _newey_njit(lambdas[:, i]-coef[i], maxlag)
    # Calculate average adjusted R2
    adjr2 = np.nanmean(ar2)
    # Calculate total number of observations
    N = np.nansum(nobs)
    # Calculate average number of observations per period
    Navg = np.nanmean(nobs)
    # Number of periods (with value estimates)
    Nperout = np.isfinite(ar2).sum()
    return coef, stderr, N, Navg, Nperout, adjr2


@njit
def _newey_njit(xe: np.ndarray, maxlag: int):
    """Calculate Newey-west estimator of standard error given demeaned series

    Parameters
    ----------
    xe: pd.np.ndarray
        Demeaned series of float
    maxlag: int
        Maximum lag in Newey-West estimator
    Return
    ------
    float
        Newey-West estimator of standard error
    """
    # Remove NaN
    xe = xe[np.isfinite(xe)]
    T = xe.shape[0]
    if T <= maxlag:
        return np.nan
    # Calculate covariances
    cov = np.zeros(maxlag+1)
    cov[0] = xe.dot(xe) / T
    for i in range(1, maxlag+1):
        cov[i] = xe[:-i].dot(xe[i:]) / T
    # The estimator (Positive and negative parts are combined)
    out = cov[0] + 2 * (1-np.arange(1, maxlag+1)/(maxlag+1)).dot(cov[1:])
    return np.sqrt(out / (T-1))


@njit
def winsor_njit(data: np.ndarray, cuts: tuple, interpolation: str):
    """Winsorize a numpy array (faster using numba)

    Parameters
    ----------
    data : numpy.ndarray
        A 1D numpy array
    cuts: tuple
        A tuple of two elements. The first element is the lower percentile and
        the second element is the upper percentile. Both should be between 0 and 1
    interpolation: str
        Either 'inner' or 'outer'. In the case that the exact percentile is between
        two values, 'inner' will choose the one closer to the center and 'outer'
        will choose the one closer to the tail

    Returns
    -------
    numpy.ndarray
        Winsorized data
    """
    # Select non-missing values
    datatmp = data[np.isfinite(data)]
    N = datatmp.shape[0]
    datatmp.sort()
    # Get quantile
    if interpolation == 'inner':
        lowerbound = datatmp[int(np.ceil((N-1)*cuts[0]))]
        upperbound = datatmp[int(np.floor((N-1)*cuts[1]))]
    elif interpolation == 'outer':
        lowerbound = datatmp[int(np.floor((N-1)*cuts[0]))]
        upperbound = datatmp[int(np.ceil((N-1)*cuts[1]))]
    else:
        raise ValueError('Invalid interpolation type.')
    out = np.zeros(data.shape)
    for i in range(data.shape[0]):
        if np.isfinite(data[i]) and data[i] < lowerbound:
            out[i] = lowerbound
        elif np.isfinite(data[i]) and data[i] > upperbound:
            out[i] = upperbound
        else:
            out[i] = data[i]
    return out


def tscssum(data: pd.DataFrame, by: list, subset: Union[list,None]=None,
            percentiles: tuple=(.01, .05, .50, .95, .99)) -> pd.DataFrame:
    """Print time-series average of cross-sectional summary statistics for numeric columns

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to generate summary statistics
    by: list
        List of columns representing time periods to group by
    subset : list of str, default: None
        The list of column names to generate summary statistics. If None, all
        columns are used.
    percentiles : tuple of float, default: (.01, .05, .25, .50, .75, .95, .99)
        The list of percentiles in the table.

    Returns
    -------
    pandas.DataFrame
        The summary statistics. Each row represents one variable and each
        column represents one statistic.
    """
    if not isinstance(by, list):
        raise TypeError('by must be a list')
    if not isinstance(subset, list):
        raise TypeError('subset must be a list')
    # Select columns (only numeric ones)
    if subset is None:
        subset = list(data.select_dtypes(include=['number']).columns)
    else:
        if not set(subset).issubset(set(data.columns)):
            raise ValueError('subset is not in columns of data')
        subset = list(data[subset].select_dtypes(include=['number']).columns)
    # Sort


    @njit
    def _sumstat_wrapper(arrlist, percentiles):
        data = arrlist[0]
        nstats = len(percentiles) + 5
        nvars = data.shape[1]
        out = np.zeros((1, nvars * nstats))
        out.fill(np.nan)
        for i in range(nvars):
            out[0,nstats*i+0] = np.isfinite(data[:,i]).sum()
            out[0,nstats*i+1] = np.nanmean(data[:,i])
            out[0,nstats*i+2] = np.nanstd(data[:,i])
            out[0,nstats*i+3] = np.nanmin(data[:,i])
            for j in range(len(percentiles)):
                out[0,nstats*i+j+4] = np.nanquantile(data[:,i], percentiles[j])
            out[0,nstats*i+len(percentiles)+4] = np.nanmax(data[:,i])
        return out


    statslist = ['N', 'Mean', 'Std', 'Min'] + [
        'p' + str(int(i * 100)) for i in percentiles
    ] + ['Max']
    colout = [(x, y) for x in subset for y in statslist]
    dftmp = groupby_apply(
        data=data[by + subset].sort_values(by),
        by=by,
        func=_sumstat_wrapper,
        colargs=[subset],
        otherargs=(percentiles,),
        colout=colout,
    )
    dftmp.drop(columns=by, inplace=True)
    dftmp.columns = pd.MultiIndex.from_tuples(dftmp.columns)
    nrows = dftmp.shape[0]
    rowlist = dftmp.columns.get_level_values(0).unique()
    collist = dftmp.columns.get_level_values(1).unique()
    out = dftmp.mean().unstack(1).loc[rowlist, collist]
    # Calculate the number of observations
    out['N'] = (out['N'] * nrows).astype('int')
    return out


def format_table(data, float_digit=3):
    """Format data in table"""
    intcols = data.select_dtypes(include='int').columns
    floatcols = data.select_dtypes(include='float').columns
    out = []
    for col in data.columns:
        if col in intcols:
            out.append(data[col].apply(lambda x: f'{x:,d}'))
        elif col in floatcols:
            out.append(data[col].apply(lambda x: f'{x:.{float_digit}f}'))
        else:
            ValueError(f'Column {col} is not of type int or float')
    return pd.concat(out, axis=1)



# =============================================================================#
# Classes related to reporting output tables
# =============================================================================#

class FMResult:
    """The estimation result from a linear regression model

    Properties
    ----------
    eststo : list of dict

    Methods
    -------
    add
        Add regression results and statistics
    output
        Print regression table
    """

    def __init__(self):
        self._eststo = []

    def __getitem__(self, key):
        if key >= len(self._eststo):
            raise IndexError('list index out of range')
        else:
            return self._eststo[key]

    @property
    def eststo(self):
        """A list of dict containing estimates and stats"""
        return self._eststo

    def add(self, params, stderrs, labels, doff, stat):
        if not isinstance(params, np.ndarray):
            raise TypeError('params is not of type numpy.ndarray')
        if not isinstance(stderrs, np.ndarray):
            raise TypeError('stderrs is not of type numpy.ndarray')
        if not params.ndim == 1:
            raise TypeError('params is not 1-D')
        if not stderrs.ndim == 1:
            raise TypeError('stderrs is not 1-D')
        if not params.shape == stderrs.shape:
            raise ValueError('params and stderrs have different shapes.')
        if not isinstance(labels, list):
            raise TypeError('labels is not a list')
        for s in labels:
            if not isinstance(s, str):
                raise TypeError(f'element {s} in labels is not str')
        if not isinstance(stat, (dict, pd.Series)):
            raise TypeError('stat is not a dict or pandas.Series')
        self._eststo.append({
            'est': [PointEstimate(params[i],
                                  stderrs[i],
                                  labels[i],
                                  doff)
                    for i in range(params.shape[0])],
            'stat': stat if isinstance(stat, pd.Series) else pd.Series(stat)
        })

    def output(self, type: str, estfmt: tuple, statfmt: tuple, roworder: list,
               tex: bool=False, scale: float=1) -> pd.DataFrame:
        """Print the regression table.

        Parameters
        ----------
        type : str, either 'tstat' or 'pval'
        estfmt : 2-d tuple
        statfmt : tuple
        roworder : list
        tex : bool, default: False
        scale : float, default: 1

        Returns
        -------
        pandas.DataFrame
        """
        # ----------- Coefficients
        # Get Series-format table
        uptbl = pd.concat([
            pd.concat([x.output(type, tex, estfmt, scale)
                       for x in y['est']])
            for y in self._eststo
        ], axis=1).reindex(roworder, level=0).fillna('')
        # Further formatting
        uptbl.index.names = [None, None]
        uptbl = uptbl.reset_index()
        uptbl.loc[lambda x: x['level_1'] == type, 'level_0'] = ''
        uptbl = uptbl.drop(columns=['level_1']) \
            .rename(columns={'level_0': ''})
        # ----------- Statistics
        # Get the table
        lotbl = pd.concat([
            x['stat'] for x in self.eststo
        ], axis=1)
        assert len(statfmt) == len(lotbl.index)
        # Format numbers
        lotbl = pd.DataFrame(
            [
                [
                    ('{:'+statfmt[i]+'}').format(v)
                    for v in lotbl.iloc[i].to_numpy()
                ]
                for i in range(lotbl.shape[0])
            ],
            index=lotbl.index
        )
        lotbl.index.name = ''
        lotbl.reset_index(inplace=True)
        tbl = pd.concat([uptbl, lotbl], axis=0)
        return tbl

class PointEstimate:
    """Point estimate of one coefficient

    Properties
    ----------
    param : float
        Point estimate
    stderr : float
        Standard error of the point estimate
    label : str
        Label of the variable
    doff : int
        Degrees of freedom, used to calculate p-value
    tstat : float
        t-statistic
    pval : float
        p-value based on Student's t distribution with `doff` degrees of freedom
    nstars : int
        Number of stars
    starlabel : str
        A string containing `nstars` stars.
    """

    def __init__(self, param, stderr, label, doff):
        if not isinstance(param, float):
            raise TypeError('param is not float')
        if not isinstance(stderr, float):
            raise TypeError('stderr is not float')
        if not isinstance(label, str):
            raise TypeError('label is not float')
        if not isinstance(doff, (int, float, np.number)):
            raise TypeError('doff is not of type int, float, or numpy.number')
        self._param = param
        self._stderr = stderr
        self._label = label
        self._doff = doff
        self._tstat = param / stderr
        self._pval = 2 * (1 - tdist.cdf(np.abs(self._tstat), self._doff))
        if 0 <= self._pval <= 0.01:
            self._nstars = 3
        elif 0.01 < self._pval <= 0.05:
            self._nstars = 2
        elif 0.05 < self._pval <= 0.1:
            self._nstars = 1
        else:
            self._nstars = 0
        self._starlabel = '*' * self._nstars

    @property
    def param(self):
        """Point estimate"""
        return self._param

    @property
    def stderr(self):
        """Standard error of the point estimate"""
        return self._stderr

    @property
    def label(self):
        """Label of the variable"""
        return self._label

    @property
    def doff(self):
        """Degrees of freedom to calculate p-value"""
        return self._doff

    @property
    def tstat(self):
        """t-statistic"""
        return self._tstat

    @property
    def pval(self):
        return self._pval

    def output(self, stat, tex, outputfmt, scale):
        """Return a pandas.Series of formatted output"""
        if not (isinstance(outputfmt, tuple) and len(outputfmt) == 2):
            raise TypeError('outputfmt must be tuple of length 2')
        out = pd.Series(
            ['', ''],
            index=pd.MultiIndex.from_product(
                [[self._label], ['param', stat]])
        )
        # Parameter estimate
        paramstr = ('{:' + outputfmt[0] + '}').format(self._param*scale)
        if self._nstars > 0:
            if tex:
                paramstr += r'\sym{' + self._starlabel + '}'
            else:
                paramstr += self._starlabel
        if tex and self._param < 0:
            paramstr = paramstr.replace('-', '$-$')
        out[(self._label, 'param')] = paramstr
        # Statistic
        if stat == 'tstat':
            statstr = ('({:' + outputfmt[1] + '})').format(self._tstat)
        elif stat == 'pval':
            statstr = ('({:' + outputfmt[1] + '})').format(self._pval)
        else:
            statstr = '()'
        if tex:
            statstr = statstr.replace('-', '$-$')
        out[(self._label, stat)] = statstr
        return out


def regtablatex(data: pd.DataFrame) -> str:
    """Generate tex table from regression table.

    Parameters:
    ----------
    data : DataFrame
        The regression table, similar to the output of fmreg()

    Returns:
    ----------
    output : string
        The tex code of the table
    """
    # Check duplicated column headers
    if not len(data.columns) == len(data.columns.to_flat_index().unique()):
        raise ValueError('Duplicated columns headers detected!')
    # ---- Preparation
    data = data.reset_index(drop=True).copy()
    # Number of columns
    nrow = data.shape[0]
    ncol = data.shape[1]
    # column number of last line with coefficients or t-stats
    coefendflag = (data.iloc[:, 0] == '')[::-1].idxmax()
    # Add modifiers to negative signs and stars
    starpat = re.compile(r'\*{1,3}')
    negpat = re.compile(r'\-{1}')
    for i in range(nrow):
        for j in range(1, ncol):
            targetstr = data.iloc[i, j]
            # Add \sym{ } around stars
            matched = starpat.search(targetstr)
            if matched is not None:
                targetstr = re.sub(
                    starpat,
                    "\\\\sym{" +
                    targetstr[matched.start():matched.end()] + "}",
                    targetstr
                )
            # Add $ $ around -
            matched = negpat.search(targetstr)
            if matched is not None:
                targetstr = re.sub(negpat, "$-$", targetstr)
            data.iloc[i, j] = targetstr
    # Get max width for each column
    maxwidth = data.apply(lambda x: x.apply(lambda x: len(x)).max()).tolist()
    # Add spaces to each cell
    for j in range(ncol):
        data.iloc[:, j] = data.iloc[:, j].apply(
            lambda x: ("{:" + str(maxwidth[j]) + "s}").format(x)
        )
    # ---- Output
    strout = ""
    # Header
    strout += "{\n\\def\\sym#1{\\ifmmode^{#1}\\else\\(^{#1}\\)\\fi}\n"
    strout += "\\begin{tabular}{l*{" + str(ncol-1) + "}{c}}\n"
    # Header row
    strout += "\\toprule\n"
    nlevels = data.columns.nlevels
    # Loop over each level
    for i in range(nlevels):
        colnames = data.columns.get_level_values(i)
        # Loop over each value
        j = 0
        while j < ncol:
            # Get the number of repetitions
            k = 1
            while j+k < ncol and colnames[j+k] == colnames[j]:
                k += 1
            if j > 0:
                strout += " & "
            strout += " \\multicolumn{" + str(k) + "}{c}{" + colnames[j] + "} "
            j += k
        strout += " \\\\\n"
    strout += "\\midrule\n"
    # Coefficients and t-stats
    for i in range(nrow):
        for j in range(ncol):
            strout += data.iloc[i, j]
            if j < ncol - 1:
                strout += " & "
            else:
                strout += " \\\\\n"
        if data.iloc[i, 0].strip() == "" and i <= coefendflag:
            strout += "\\addlinespace\n"
        if i == coefendflag:
            strout += "\\midrule\n"
    # Bottom
    strout += "\\bottomrule\n\\end{tabular}\n}"

    return strout


def regtabtext(data: pd.DataFrame, showindex: bool=False, colborder: bool=False,
               title: Union[str,None]=None) -> str:
    """Generate tex table from regression table.

    Parameters:
    ----------
    data : DataFrame
        The regression table, similar to the output of get_fmreg_all(). All
        columns are str. Allow for MultiIndex up to 2 levels in columns.
    showindex : bool, default: True
        Whether or not to show index in the result.
    colborder : bool, default: False
        Whether or not to show column borders in the result.
    title : str, default: None
        Title of the table

    Returns:
    ----------
    output : string
        The text code of the table
    """
    # ---------- Check arguments
    # Check duplicated column headers
    if not len(data.columns) == len(data.columns.to_flat_index().unique()):
        raise ValueError('Duplicated columns headers detected!')
    # Columns only contain string and only two levels for MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        if not data.columns.nlevels == 2:
            raise ValueError('Columns of data does not have level 2')
        if not all(isinstance(s, str) for s in data.columns.levels[0]):
            raise TypeError('Some element of level 0 of data.columns is not str')
        if not all(isinstance(s, str) for s in data.columns.levels[1]):
            raise TypeError('Some element of level 1 of data.columns is not str')
    else:
        if not all(isinstance(s, str) for s in data.columns):
            raise TypeError('Some element of data.columns is not str')
    # If showindex, then it only contains string
    if showindex:
        if not data.index.dtype.kind == 'O':
            raise ValueError('Index is not of str type.')
    # ---- Preparation
    data = data.reset_index(drop=not showindex).copy()
    # Number of columns and rows
    nrow = data.shape[0]
    ncol = data.shape[1]
    # column number of last line with coefficients or t-stats
    coefendflag = (data.iloc[:, 0] == "")[::-1].idxmax()
    # Add space
    data = data.applymap(unstrip)
    # Get max width for each column
    maxwidth = getmaxwidth(data)
    # Add space for the first time
    data = aligntable(data, maxwidth)
    # Update max width
    maxwidth = getmaxwidth(data)
    # If columns is a MultiIndex, adjust the maxwidth
    if isinstance(data.columns, pd.MultiIndex):
        # Loop over each high-level index
        for s in data.columns.levels[0]:
            if s.strip() != '':
                s2 = unstrip(s)
                # If the width of the multicolumn text exceeds the sum of width
                # of columns exceeds the multicolumn text
                # The width of column borders is also added
                if maxwidth[s].sum() + maxwidth[s].shape[0] - 1 < len(s2):
                    # Proportionally increase all columns
                    maxwidth[[s]] += np.ceil((len(s2) -
                                              (maxwidth[s].sum() +
                                               maxwidth[s].shape[0] - 1)
                                              ) / maxwidth[s].shape[0]).astype('int')
        # Add space again
        data = aligntable(data, maxwidth)
    # ---------- Output
    strout = ""
    totalwidth = (maxwidth.sum() + data.shape[1] + 1)
    # Title
    if title is not None:
        strout += title
        strout += '\n'
    # Header
    strout += '=' * totalwidth + '\n'
    if isinstance(data.columns, pd.MultiIndex):
        # Top level
        strout += '|' + '|'.join(
            [aligntext(x,
                       maxwidth[x].sum() + maxwidth[x].shape[0] - 1,
                       'center')
             for x in data.columns.levels[0]]
        ) + '|' + '\n'
        # Bottom level
        strout += '|' + '|'.join(
            [aligntext(data.columns.get_level_values(1)[i],
                       maxwidth.iloc[i],
                       'center') for i in range(data.shape[1])]
        ) + '|' + '\n'
    else:
        strout += '|' + '|'.join(data.columns) + '|' + '\n'
    strout += '-' * totalwidth + '\n'
    # Coefficients and t-stats
    for i in range(nrow):
        strout += '|' + '|'.join(
            [data.iloc[i, j] for j in range(ncol)]
        ) + '|' + '\n'
        # Add line separating coefficients and other statistics
        if i == coefendflag:
            strout += '-' * totalwidth + '\n'
    strout += '=' * totalwidth + '\n'
    if not colborder:
        strout = strout.replace('|', ' ')
    return strout


def aligntable(data, maxwidth):
    ncol = data.shape[1]
    # Add spaces to each cell
    for j in range(ncol):
        if j == 0:
            # Align left for the first column
            data.iloc[:, j] = data.iloc[:, j].apply(
                lambda x: aligntext(x, maxwidth.iloc[j], 'left')
            )
        else:
            # Align centered for the others
            data.iloc[:, j] = data.iloc[:, j].apply(
                lambda x: aligntext(x, maxwidth.iloc[j], 'center')
            )
    return data


def aligntext(text, width, align):
    """Align text with specific width and direction
    """
    if not isinstance(text, str):
        raise ValueError('text is not of type str.')
    if not int(width) == width:
        raise ValueError('width is not an integer')
    if not width >= len(text):
        raise ValueError(f'width ({width}) is smaller than the width of text ({text})')
    if text == '':
        return ' ' * width
    if align == 'left':
        alignsym = '<'
    elif align == 'right':
        alignsym = '>'
    elif align == 'center':
        alignsym = '^'
    else:
        raise ValueError(f'Invalid align type: {align}')
    outstr = ('{:' + alignsym + str(width) + 's}').format(text)
    return outstr


def unstrip(text):
    if len(text) == 0:
        return ''
    if text[0] != ' ':
        text = ' ' + text
    if text[-1] != ' ':
        text = text + ' '
    return text


def getmaxwidth(data):
    return data.apply(
        lambda col: col.apply(lambda cell: len(cell)).max()
    )


# cut portfolios into groups
@njit
def qcut_jit(x, q):
    return np.floor(np.searchsorted(np.sort(x), x, side='right') * q / len(x) - 1e-12) + 1

# # Univariate portfolio sorting
# def uniportsort(
#         data: pd.DataFrame, retvar, sortvar, ngroups: int, weight, maxlag: int,
#         fdata=None, fmodel=None, scale=1
# ):
#     """Generate tex table from regression table.
#
#     Parameters:
#     ----------
#     data : pandas.DataFrame
#         Panel data containing all variables needed, e.g. returns, weights, and
#         variables to be sorted. Its index must be a MultiIndex with level 0
#         representing entities and level 1 representing time periods
#     retvar:
#         Column of data as the return
#     sortvar:
#         Column of data to be sorted on
#     ngroups: int
#         Number of groups (positive integer)
#     weight:
#         Column of data to calculate weighted average returns
#     maxlag: int
#         Maximum lag for Newey-West correction; negative for non-adjustment
#     fdata: pd.DataFrame, default: None
#         Table containing factors to adjusted return. Its index must be
#         consistent with level 1 of the data.index. The user should guarantee
#         that factors in fdata and returns in the column retvar of data are
#         consistent (both are in decimal or percentage). None if no adjustment
#         to be reported. fdata and fmodel must either both be None or both be
#         of the respective types.
#     fmodel: dict, default: None
#         A dict of keys (model names) and values (list of columns of fdata as
#         factors) to adjust return. None if no adjustment to be reported.
#     scale: float, default 1
#         Scale to be multiplied by the return and factors.
#     reverse: bool, default False
#         By default, the
#     Returns:
#     ----------
#
#     """
#     # Check arguments
#     if not isinstance(data, pd.DataFrame):
#         raise TypeError('data is not a pandas.DataFrame')
#     if retvar not in data.columns:
#         raise ValueError('retvar is not a column of data')
#     if sortvar not in data.columns:
#         raise ValueError('sortvar is not a column of data')
#     if not (isinstance(ngroups, int) and ngroups > 0):
#         raise TypeError('ngroups must be a positive integer')
#     if not (isinstance(ngroups, int) and ngroups > 0):
#         raise TypeError('ngroups must be a positive integer')
#     if weight not in data.columns:
#         raise ValueError('weight is not a column of data')
#     if not isinstance(maxlag, int):
#         raise TypeError('maxlag must be an integer')
#     if not ((fmodel is None and fdata is None) or (
#             isinstance(fdata, pd.DataFrame) and isinstance(fmodel, dict)
#     )):
#         raise TypeError('fdata and fmodel must either both be None or not be None')
#     shortlong = modelsetup['shortlong']
#     maxlag = modelsetup['maxlag']
#     fmodel = modelsetup['fmodel']
#     fdata = modelsetup['fdata']
#     pct = modelsetup['pct']
#     # Long-short or short-long
#     if shortlong:
#         lsvar = f'P1-P{ngroups}'
#     else:
#         lsvar = f'P{ngroups}-P1'
#     # Merge sort variable with returns
#     if weight is None:
#         dfr = pd.concat(
#             [rdata[retvar], sdata[sortvar].rename(sortvar)], axis=1
#         ).dropna(axis=0, subset=[sortvar])
#     else:
#         dfr = pd.concat(
#             [rdata[[retvar, weight]], sdata[sortvar].rename(sortvar)], axis=1
#         ).dropna(axis=0, subset=[sortvar])
#     # The user should ensure that the correct model is sent to the function
#     # if weight is None:
#     #     if modelname is None:
#     #         dfr = pd.concat(
#     #             [rdata[retvar], sdata[sortvar].rename(sortvar)],
#     #             axis=1
#     #         ).dropna(axis=0, subset=[sortvar])
#     #     else:
#     #         dfr = pd.concat(
#     #             [rdata[retvar], sdata[(modelname, sortvar)].rename(sortvar)],
#     #             axis=1
#     #         ).dropna(axis=0, subset=[sortvar])
#     # else:
#     #     if modelname is None:
#     #         dfr = pd.concat(
#     #             [rdata[[retvar,weight]], sdata[sortvar].rename(sortvar)],
#     #             axis=1
#     #         ).dropna(axis=0, subset=[sortvar])
#     #     else:
#     #         dfr = pd.concat(
#     #             [rdata[[retvar,weight]], sdata[(modelname, sortvar)].rename(sortvar)], axis=1
#     #         ).dropna(axis=0, subset=[sortvar])
#     # Get ranks
#     dfr['rnk'] = dfr.groupby(timevar)[sortvar].transform(lambda x: qcut_jit(x.values, q=ngroups)).astype('int')
#     # Calculate portfolio return
#     if weight is None:
#         dfp = dfr.reset_index().groupby([timevar, 'rnk']).apply(
#             lambda x: np.mean(x[retvar])
#         ).to_frame(name=retvar)
#     else:
#         dfp = dfr.reset_index().groupby([timevar, 'rnk']).apply(
#             lambda x: np.average(x[retvar], weights=x[weight])
#         ).to_frame(name=retvar)
#     # Calculate average sorted variable
#     dfs = dfr.reset_index().groupby([timevar, 'rnk']).apply(
#         lambda x: np.mean(x[sortvar])
#     ).to_frame(name=sortvar)
#     del dfr
#     # Transpose
#     dfp = dfp.reset_index().pivot(index=timevar, columns='rnk', values=retvar).rename(
#         columns=lambda x: 'P' + str(x)
#     )
#     dfs = dfs.reset_index().pivot(index=timevar, columns='rnk', values=sortvar).rename(
#         columns=lambda x: 'P' + str(x)
#     )
#     # Average sort variable
#     res_sortvar = dfs.mean().apply(lambda x: "{:5.4f}".format(x)).rename('fconst').to_frame().assign(ftstat="").stack().rename(r'$\beta_{\mathrm{'+sortvar+r'}}$')
#     # Long-short portfolio return
#     if shortlong:
#         dfp[lsvar] = dfp['P1'] - dfp[f'P{ngroups}']
#     else:
#         dfp[lsvar] = dfp[f'P{ngroups}'] - dfp['P1']
#     # Calculate average raw return and t-stat for each portfolio
#     dfavgp = pd.concat([_get_ave_ret(x[1], maxlag) for x in dfp.items()], axis=1, keys=dfp.columns)
#     dfavgp = dfavgp.T
#     # Format average return and t-stat
#     dfavgp['fconst'] = dfavgp[['const','t','pval']].apply(lambda x: _format_const(x, 'const', 'pval', pct), axis=1)
#     dfavgp['ftstat'] = dfavgp[['t']].apply(lambda x: '({:4.2f})'.format(x['t']), axis=1)
#     # Generate output column for average return
#     res_avgret = dfavgp[['fconst','ftstat']].stack().rename('Average return')
#     # Calculate alpha or not
#     if fmodel is None or fdata is None:
#         res_out = pd.concat([res_sortvar, res_avgret], axis=1).fillna(value="").reindex(res_avgret.index).reset_index().drop(columns='level_1')
#     else:
#         # Calculate and format alpha
#         res_alpha = pd.concat([_get_formated_alpha(dfp, fdata.reindex(dfp.index), fvars, maxlag, pct) for fname, fvars in fmodel.items()], axis=1, keys=fmodel.keys())
#         res_out = pd.concat([res_sortvar, res_avgret, res_alpha], axis=1).fillna(value="").reindex(res_avgret.index).reset_index().drop(columns='level_1')
#     # Output
#     res_out['rnk'] = res_out['rnk'].where(np.arange((ngroups+1)*2) % 2 == 0, other="")
#     res_out.rename(columns={'rnk':'Portfolio'}, inplace=True)
#     return res_out
#
#
# def biportsortall(modelsetup, cdata, ctrlvarlst, fvars):
#     res_out = pd.concat(
#         map(lambda x: biportsort1(modelsetup, cdata, x, fvars), ctrlvarlst), axis=1, keys=ctrlvarlst
#     ).reset_index().drop(columns='level_1')
#     res_out['rnk_sort'] = res_out['rnk_sort'].where(np.arange((modelsetup['ngroups']+1)*2) % 2 == 0, other="")
#     res_out.rename(columns={'rnk_sort':'Portfolio'}, inplace=True)
#     return res_out
#
#
# def biportsort1(modelsetup, cdata, ctrlvar, fvars):
#     rdata = modelsetup['rdata']
#     sdata = modelsetup['sdata']
#     timevar = modelsetup['timevar']
#     # modelname = modelsetup['modelname']
#     sortvar = modelsetup['sortvar']
#     retvar = modelsetup['retvar']
#     ngroups = modelsetup['ngroups']
#     weight = modelsetup['weight']
#     shortlong = modelsetup['shortlong']
#     maxlag = modelsetup['maxlag']
#     fdata = modelsetup['fdata']
#     pct = modelsetup['pct']
#     # Long-short or short-long
#     if shortlong:
#         lsvar = f'P1-P{ngroups}'
#     else:
#         lsvar = f'P{ngroups}-P1'
#     # Merge sort variable with returns
#     if weight is None:
#         dfr = pd.concat(
#             [rdata[retvar], cdata[ctrlvar], sdata[sortvar].rename(sortvar)], axis=1
#         ).dropna(axis=0, subset=[ctrlvar, sortvar])
#     else:
#         dfr = pd.concat(
#             [rdata[[retvar, weight]], cdata[ctrlvar], sdata[sortvar].rename(sortvar)], axis=1
#         ).dropna(axis=0, subset=[ctrlvar, sortvar])
#     # if weight is None:
#     #     if modelname is None:
#     #         dfr = pd.concat(
#     #             [rdata[retvar], cdata[ctrlvar], sdata[sortvar].rename(sortvar)],
#     #             axis=1
#     #         ).dropna(axis=0, subset=[ctrlvar, sortvar])
#     #     else:
#     #         dfr = pd.concat(
#     #             [rdata[retvar], cdata[ctrlvar], sdata[(modelname, sortvar)].rename(sortvar)],
#     #             axis=1
#     #         ).dropna(axis=0, subset=[ctrlvar, sortvar])
#     # else:
#     #     if modelname is None:
#     #         dfr = pd.concat(
#     #             [rdata[[retvar,weight]], cdata[ctrlvar], sdata[sortvar].rename(sortvar)],
#     #             axis=1
#     #         ).dropna(axis=0, subset=[ctrlvar, sortvar])
#     #     else:
#     #         dfr = pd.concat(
#     #             [rdata[[retvar,weight]], cdata[ctrlvar], sdata[(modelname, sortvar)].rename(sortvar)], axis=1
#     #         ).dropna(axis=0, subset=[ctrlvar, sortvar])
#     # Get ctrl variable rank
#     dfr['rnk_ctrl'] = dfr.groupby(timevar)[ctrlvar].transform(
#         lambda x: qcut_jit(x.values, q=ngroups)
#     ).astype('int')
#     # Get sort variable rank controlling for ctrl variable
#     # dfr['rnk_sort'] = dfr.groupby([timevar, 'rnk_ctrl'])[sortvar].transform(
#     #     lambda x: pd.qcut(x, q=ngroups, labels=np.arange(1, ngroups+1))
#     # ).astype('int')
#     dfr['rnk_sort'] = dfr.groupby([timevar, 'rnk_ctrl'])[sortvar].transform(
#         lambda x: qcut_jit(x.values, q=ngroups)
#     ).astype('int')
#     # Calculate portfolio return
#     if weight is None:
#         dfp = dfr.reset_index().groupby([timevar, 'rnk_ctrl', 'rnk_sort']).agg({retvar: 'mean'})
#     else:
#         dfr['__rw__'] = dfr[retvar].values * dfr[weight].values
#         dfr = dfr.reset_index().groupby([timevar, 'rnk_ctrl', 'rnk_sort']).agg({weight: 'sum', '__rw__': 'sum'})
#         dfp = pd.DataFrame(dfr['__rw__'].values / dfr[weight].values, index=dfr.index, columns=[retvar])
#     del dfr
#     #-------- Output 1: Average across ctrl ranks
#     # Calculate average portfolio return across all ctrl ranks
#     dfp = dfp.groupby(['date_mn', 'rnk_sort'])[[retvar]].agg('mean')
#     # Transpose
#     dfp = dfp.reset_index().pivot(index=timevar, columns='rnk_sort', values=retvar).rename(
#         columns=lambda x: 'P' + str(x)
#     )
#     # Long-short portfolio return
#     if shortlong:
#         dfp[lsvar] = dfp['P1'] - dfp[f'P{ngroups}']
#     else:
#         dfp[lsvar] = dfp[f'P{ngroups}'] - dfp['P1']
#     # Calculate average return or alpha
#     if fvars is None:
#         # Calculate average raw return and t-stat for each portfolio
#         dfavgp = pd.concat([_get_ave_ret(x[1], maxlag) for x in dfp.items()], axis=1, keys=dfp.columns)
#         dfavgp = dfavgp.T
#         # Format average return and t-stat
#         dfavgp['fconst'] = dfavgp[['const','t','pval']].apply(lambda x: _format_const(x, 'const', 'pval', pct), axis=1)
#         dfavgp['ftstat'] = dfavgp[['t']].apply(lambda x: '({:4.2f})'.format(x['t']), axis=1)
#         # Generate output column for average return
#         res_out = dfavgp[['fconst','ftstat']].stack().rename(ctrlvar)
#     else:
#         res_out = _get_formated_alpha(dfp, fdata.reindex(dfp.index), fvars, maxlag).rename(ctrlvar)
#     return res_out


def groupby_wavg(data: pd.DataFrame, bys: list, var, weight) -> pd.Series:
    """Calculate weighted average of varlist by groups

    Parameters
    ----------
    data: pd.DataFrame
        The input table
    bys: list
        Columns of data to be grouped by
    var:
        Column of data to calculate weighted average
    weight:
        Column of data as weight

    Returns
    -------
    pd.Series
        With index same as bys, the value is the by-group average of var
        weighted by weight
    """
    if isinstance(var, str):
        xyvar = var + '_xy'
    else:
        xyvar = 'xy'
    # Remove missing values
    datatmp = data.dropna(subset=[var, weight])
    outvw = datatmp.assign(
        **{xyvar: lambda x: x[var]*x[weight]}
    ).groupby(bys)[[xyvar, weight]].sum()
    outvw[var] = outvw[xyvar] / outvw[weight]
    return outvw[var]


# Estimate factor alpha given factor data (fdata), factor variables (fvars)
def _get_alpha(x, fdata, fvars, maxlag):
    exog = fdata[fvars].dropna(how='any').assign(const=1)
    endog = x.reindex(exog.index)
    mod = OLS(endog, exog).fit()
    res = pd.Series([mod.params['const'],
                     np.sqrt(cov_hac(mod, nlags=maxlag)[-1,-1]),
                     mod.df_resid], index=['const','se','df'])
    res['t'] = res['const'] / res['se']
    res['pval'] = 2 * (1 - scipy.stats.t.cdf(abs(res['t']), res['df']))
    return res


# Calculate alpha and format columns for each factor model (prdata: portfolio return data)
def _get_formated_alpha(prdata, fdata, fvars, maxlag, pct):
    dfalpha = pd.concat([_get_alpha(x[1], fdata, fvars, maxlag) for x in prdata.items()], axis=1, keys=prdata.columns)
    dfalpha = dfalpha.T
    # Format alpha
    dfalpha['fconst'] = dfalpha[['const','t','pval']].apply(lambda x: _format_const(x, 'const', 'pval', pct), axis=1)
    dfalpha['ftstat'] = dfalpha[['t']].apply(lambda x: '({:4.2f})'.format(x['t']), axis=1)
    return dfalpha[['fconst','ftstat']].stack()


# Format point estimates with stars based on pval
def _format_const(estseries, paramvar, pvalvar, scale, format='9.2f'):
    """Format point estimates with stars based on p-value.

    Parameters
    ----------
    estseries : pandas.Series
        A series of parameter estimates and p-values, such that
        estseries[parameter] is the point estimate and estseries[pvalvar] is
        the corresponding p-value.
    paramvar : str
    pvalvar : str
    scale : int
        Multiplier to apply to the point estimate.
    format : str, default: '9.2f'
        Format of the point estimate

    Returns
    -------
    str
    """
    if estseries[pvalvar] > 0.1:
        return ('{:'+format+'}').format(estseries[paramvar] * scale)
    elif (0.05 < estseries[pvalvar]) and (estseries[pvalvar] <= 0.1):
        return ('{:'+format+'}*').format(estseries[paramvar] * scale)
    elif (0.01 < estseries[pvalvar]) and (estseries[pvalvar] <= 0.05):
        return ('{:'+format+'}**').format(estseries[paramvar] * scale)
    elif (0 <= estseries[pvalvar]) and (estseries[pvalvar] <= 0.01):
        return ('{:'+format+'}***').format(estseries[paramvar] * scale)


# Estimate average return and t-stat for each column (portfolio) with Newey-West standard error
def _get_ave_ret(x, maxlag):
    exog = np.ones(x.shape)
    mod = OLS(x, exog).fit()
    res = pd.Series(
        [mod.params.values[0], np.sqrt(cov_hac(mod, nlags=maxlag))[0][0], mod.df_resid],
        index=['const', 'se', 'df']
    )
    res['t'] = res['const'] / res['se']
    res['pval'] = 2 * (1 - tdist.cdf(abs(res['t']), res['df']))
    return res


def get_port_ret(
        data: pd.DataFrame,
        nq: Union[int, List[int]],
        timevar,
        retvar,
        rnkvars: list,
        rnkvarnames: list,
        wvar=None,
        dep=False,
        retfull=False,
):
    """Construct equal-weighted and/or value-weighted portfolio returns sorted
    on different variables, dependently or independently, from a DataFrame
    containing panel returns of securities for different times.

    Parameters
    ----------
    data : DataFrame
        A panel (long-type or stacked) data of returns for different securities
        at different times.
    nq : int or list of int
        Number(s) of portfolios for each rank variable. The order should be
        consistent with `rnkvars`. If it is an integer, the number is applied
        for all rank variables.
    timevar : str
        Name of the variable representing time. This variable determines the
        time variable in the output.
    retvar : str
        Name of the variable representing security returns.
    rnkvars : list of str
        Name(s) of variables to rank on.
    rnkvarnames : list of str
        Name(s) of the created variable representing different portfolios
    wvar : str or None, default None
        Name of the variable representing portfolio weights. If it is None,
        only equal-weighted returns are calcualted
    dep : bool, default False
        Whether or not variable sorts are dependent. If `dep=True`, the order
        of variable names in `rnkvars` indicates the order of dependent sorting.
    retfull : bool, default False
        If True, a tuple will be returned from the function. The first element
        contains portfolios while the second one contains the full input data
        set with rank variables.

    Returns:
    ----------
    output : DataFrame
        A DataFrame containing a time-series of returns for different portfolios
    data : DataFrame
        The input data adding portfolio numbers
    """
    data = data.copy()
    # If nq is an integer, equivalent to nq <- [nq, ..., nq] with len(rnkvars)
    if np.ndim(nq) == 0:
        nq = [nq] * len(rnkvars)
    # Remove NaN
    data = data.dropna(subset=rnkvars+[retvar], how='any')
    # Generate group-by variable list
    by_list = [[timevar]]
    if dep:
        # Successively add each rank variable
        for v in rnkvarnames:
            by_list.append(by_list[-1] + [v])
        del by_list[len(rnkvars)]
    else:
        by_list = by_list * len(rnkvars)
    # Obtain portfolio ranks
    for i in range(len(by_list)):
        rnkvar = rnkvars[i]
        rnkvarname = rnkvarnames[i]
        data = data.groupby(by_list[i]).apply(
            lambda x: _get_qcut(x, nq[i], rnkvar, rnkvarname)
        ).reset_index(drop=True)
    # Equal-weighted return
    out = data.groupby(
        [timevar] + rnkvarnames
    )[retvar].mean().to_frame('ew_ret')
    # Value-weighted return
    if wvar is not None:
        outvw = groupby_wavg(
            data, [timevar] + rnkvarnames, retvar, wvar
        ).to_frame('vw_ret')
        out = out.join(outvw)
    # Nobs
    out = out.join(
        data.groupby([timevar] + rnkvarnames)[retvar].count().to_frame('cnt')
    ).reset_index(drop=False)
    if retfull:
        return out, data
    else:
        return out


def get_qcut(x, q, rnkvar, rnkvarname):
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
    return _get_qcut(x, q, rnkvar, rnkvarname)

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
