import numpy as np
import pandas as pd
from numba import njit
from joblib import Parallel, delayed, cpu_count
from scipy.stats import t as tdist
from typing import Union
import re
from functools import partial

# =============================================================================#
# Functions to estimate beta
# =============================================================================#

def estbeta(leftdata: pd.DataFrame, rightdata: pd.DataFrame, models: dict,
            window: tuple, minobs: int, hasconst: bool, ncore: int=cpu_count()
            ) -> pd.DataFrame:
    """Estimate beta given a list of models.

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
    """Estimate beta for a single model

    Parameters
    ----------
    leftdata : pandas.Series
        The index should be a MultiIndex whose level 0 represents assets and
        level 1 represents time periods
    rightdata : pandas.DataFrame
        The index represents time periods and is the same as the level 1 of
        the index of `leftdata`.
    model : list
        The elements should be in the columns of rightdata.
    window : tuple of 2 integers
        The period used to estimate beta. For example, `(-24,-1)` means using
        data between `t-24` and `t-1` to estimate beta at time `t`.
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
    assert isinstance(leftdata, pd.Series)
    assert isinstance(rightdata, pd.DataFrame)
    assert set(model).issubset(set(rightdata.columns))
    # Remove NaN
    leftdata = leftdata.dropna()
    rightdata = rightdata.dropna(how='all')
    # Construct matrix
    leftmat = leftdata.unstack(level=0).sort_index()
    rightdata = rightdata.reindex(leftmat.index)
    # Construct X
    exog = rightdata[model].to_numpy()
    out = pd.DataFrame(
        np.concatenate(
            Parallel(n_jobs=ncore)(
                delayed(
                    lambda x: _get_beta_njit(
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



@njit
def _get_beta_njit(endogmat: np.ndarray, exogmat: np.ndarray, window: tuple,
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
    # Number of periods, variables,
    nper = endogmat.shape[0]
    nvar = exogmat.shape[1]
    # Add constant
    if hasconst:
        exogmat = np.hstack((exogmat, np.ones((nper, 1))))
    # Initialization
    estbeta = np.zeros((nper, nvar))
    estbeta.fill(np.nan)
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
            if hasconst:
                estbeta[t, :] = (np.linalg.inv(XX) @ X.T @ Y)[:-1]
            else:
                estbeta[t, :] = np.linalg.inv(XX) @ X.T @ Y
    return estbeta


# =============================================================================#
# Fama-MacBeth Regression
# =============================================================================#

def fmreg(leftdata: pd.DataFrame, rightdata: pd.DataFrame, models: list,
        maxlag: int, roworder: list, hasconst: bool, scale: float,
        getlambda: bool, winsorcuts: Union[tuple,None]=None,
        winsorindeponly: bool=True, estfmt: tuple=('.3f', '.2f')) -> pd.DataFrame:
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
        NOT IMPLEMENTED YET
    """
    # Check arguments
    assert isinstance(leftdata, pd.DataFrame)
    assert isinstance(rightdata, pd.DataFrame)
    if isinstance(leftdata.columns, pd.MultiIndex):
        assert leftdata.columns.nlevels == 2
    else:
        assert isinstance(leftdata.columns, pd.Index)
    if isinstance(rightdata.columns, pd.MultiIndex):
        assert rightdata.columns.nlevels == 2
        assert set(roworder).issubset(set(rightdata.columns.levels[1]))
    else:
        assert isinstance(rightdata.columns, pd.Index)
        assert set(roworder).issubset(set(rightdata.columns))
    assert isinstance(hasconst, bool)
    assert (winsorcuts is None) or (
            isinstance(winsorcuts, tuple) and
            (len(winsorcuts) == 2) and (0<=winsorcuts[0]<=1) and
            (0<=winsorcuts[1]<=1) and (winsorcuts[0]<winsorcuts[1]))
    assert isinstance(winsorindeponly, bool)
    assert isinstance(estfmt, tuple) and len(estfmt) == 2
    for s in estfmt:
        assert isinstance(s, str)

    estout = FMResult()
    for model in models:
        # Check arguments
        assert isinstance(model, list) and len(model) >= 2
        assert model[0] in leftdata.columns, 'model[0] not in columns of leftdata'
        assert set(model[1:]).issubset(set(rightdata.columns.to_flat_index())), \
            'model[1:] not in columns of rightdata'
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
    return estouttab


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
                    X[:, i] = _winsor_njit(X[:, i], winsorcuts)
                if not winsorindeponly:
                    Y = _winsor_njit(Y, winsorcuts)
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
def _winsor_njit(data: np.ndarray, cuts: tuple, interpolation: str):
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


def tscssum(data: pd.DataFrame, subset: Union[list,None]=None,
            percentiles: tuple=(.01, .05, .50, .95, .99)) -> pd.DataFrame:
    """Print time-series average of cross-sectional summary statistics

    Parameters
    ----------
    data : pandas.DataFrame
        The data table to generate summary statistics
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
    if subset is None:
        subset = data.columns
    else:
        assert set(subset).issubset(set(data.columns))
    # Construct functions
    funclist = [
       lambda x: np.isfinite(x).sum() if np.isfinite(
           x).sum() > 0 else np.nan,
       lambda x: x[np.isfinite(x)].mean() if np.isfinite(
           x).sum() > 0 else np.nan,
       lambda x: x[np.isfinite(x)].std() if np.isfinite(
           x).sum() > 0 else np.nan,
       lambda x: x[np.isfinite(x)].min() if np.isfinite(
           x).sum() > 0 else np.nan,
       ] + [
        partial(
            lambda x, i: np.percentile(x[np.isfinite(x)], q=i * 100)
            if np.isfinite(x).sum() > 0 else np.nan, i=i)
        for i in percentiles
    ] + [lambda x: x[np.isfinite(x)].max() if np.isfinite(
        x).sum() > 0 else np.nan]
    # Calculate the number of observations
    resN = data.groupby(level=1)[subset].agg(funclist[0]).sum().astype('int')
    # Calculate other summary statistics
    res = data.groupby(level=1)[subset] \
        .agg(funclist) \
        .mean() \
        .unstack(level=-1) \
        .set_axis(['N', 'Mean', 'Std', 'Min'] +
                  ['p' + str(int(i * 100)) for i in percentiles] + ['Max'],
                  axis='columns'
                  )
    res['N'] = resN
    return res



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
        assert isinstance(params, np.ndarray), \
            'params is not of type numpy.ndarray'
        assert isinstance(stderrs, np.ndarray), \
            'stderrs is not of type numpy.ndarray'
        assert params.ndim == 1, 'params is not 1-D'
        assert stderrs.ndim == 1, 'stderrs is not 1-D'
        assert params.shape == stderrs.shape, \
            'params and stderrs have different shapes.'
        assert isinstance(labels, list), 'labels is not a list'
        for s in labels:
            assert isinstance(s, str), f'element {s} in labels is not str'
        assert isinstance(stat, (dict, pd.Series))
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
        assert isinstance(param, float)
        assert isinstance(stderr, float)
        assert isinstance(label, str)
        assert isinstance(doff, (int, float, np.number))
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
        assert isinstance(outputfmt, tuple) and len(outputfmt) == 2
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
    assert len(data.columns) == len(data.columns.to_flat_index().unique()), 'Duplicated columns headers detected!'
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
    assert len(data.columns) == len(data.columns.to_flat_index().unique()), \
        'Duplicated columns headers detected!'
    # Columns only contain string and only two levels for MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        assert data.columns.nlevels == 2
        assert all(isinstance(s, str) for s in data.columns.levels[0])
        assert all(isinstance(s, str) for s in data.columns.levels[1])
    else:
        assert all(isinstance(s, str) for s in data.columns)
    # If showindex, then it only contains string
    if showindex:
        assert data.index.dtype.kind == 'O', 'Index is not of str type.'
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
    assert isinstance(text, str), 'text is not of type str.'
    assert int(width) == width, 'width is not an integer'
    assert align in ['left', 'right', 'center'], 'Invalid align type.'
    assert width >= len(text), \
        f'width ({width}) is smaller than the width of text ({text})'
    if text == '':
        return ' ' * width
    if align == 'left':
        alignsym = '<'
    elif align == 'right':
        alignsym = '>'
    elif align == 'center':
        alignsym = '^'
    else:
        pass
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

