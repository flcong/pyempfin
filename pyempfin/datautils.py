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


def trunc(data: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Truncate data at specific percentiles
        E.g. _trunc(data, threshold=(0.01, 0.99)) truncate data
        at 1% and 99%.    

    Parameters
    ----------
    data : np.ndarray
    lower : float
        Lower percentile to truncate. The value should be between 0 and 1
    upper : float
        Upper percentile to truncate. The value should be between 0 and 1

    Returns
    -------
    np.ndarray
    """    
    assert 0 <= lower <= 1, 'Lower percentile is not between 0 and 1'
    assert 0 <= upper <= 1, 'Upper percentile is not between 0 and 1'
    # Get mask for non-missing values
    nnamask = ~np.isnan(data)
    # Get lower and upper threshold
    lo = np.quantile(data[nnamask], lower)
    up = np.quantile(data[nnamask], upper)
    # Get values between lo and up
    res = np.nan * np.ones(len(data))
    mask = (lo <= data) & (data <= up)
    res[mask] = data[mask]
    return res


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



def getnodup(data: pd.DataFrame, subset: Union[list,None]=None) -> pd.DataFrame:
    """Return rows without duplicated values over the subset of variables

    Parameters
    ----------
    data : pandas.DataFrame
    subset : list of str, default: None

    Returns
    -------
    pandas.DataFrame
    """
    return data.loc[lambda x: ~x.duplicated(subset=subset, keep=False)]



# Count missing values
def desmiss(data: pd.DataFrame,
            subset: Union[list,None]=None,
            not_missing: bool=False,
            sort: Union[str,None]='descending') -> pd.DataFrame:
    """Return a description of missing (or non-missing) observations in the data set

    Parameters
    ----------
    data : pandas.DataFrame
    subset : list of str, default: None
    not_missing : bool, default: False
    sort : {'descending', 'ascending', None}, default: 'descending'

    Returns
    -------
    pandas.DataFrame
    """
    if subset is None:
        subset = data.columns
    if not_missing:
        out = pd.DataFrame({
            '# not NaN': data[subset].apply(lambda x: (~x.isna()).sum())
        })
        out['% not NaN'] = out['# not NaN'].apply(
            lambda x: '{:.2f}%'.format(x / data.shape[0] * 100)
        )
        sort_var = '# not NaN'
    else:
        out = pd.DataFrame({
            '# NaN': data[subset].apply(lambda x: x.isna().sum())
        })
        out['% NaN'] = out['# NaN'].apply(
            lambda x: '{:.2f}%'.format(x / data.shape[0] * 100)
        )
        out = out.sort_values('# NaN', ascending=False)
        sort_var = '# NaN'
    # Sort
    if sort == 'descending':
        out = out.sort_values(sort_var, ascending=False)
    elif sort == 'ascending':
        out = out.sort_values(sort_var, ascending=True)
    elif sort is None:
        pass
    else:
        raise ValueError('Invalid value for argument sort')
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


def weighted_mean(data: pd.DataFrame,
                  by: list,
                  value: str,
                  weight: str,
                  wavg: str) -> pd.DataFrame:
    """Calculate weighted average

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame
    by : list of str
        List of column names to group by
    value : str
        Column name of the variable to calculate average
    weight : str
        Column name of the variable as weight
    wavg : str
        Column name of the calculated weighted average

    Returns
    -------
    pd.DataFrame
        Output DataFrame with column names in by and wavg.
    """
    # Dict to encode column names
    encodedict = {by[i]: f'by_{i}' for i in range(len(by))}
    encodedict[value] = 'value'
    encodedict[weight] = 'weight'
    # Dict to decode column names
    decodedict = {f'by_{i}': by[i] for i in range(len(by))}
    decodedict['value'] = value
    decodedict['weight'] = weight
    decodedict['wavg'] = wavg
    # Assign encoded names
    by_ = [f'by_{i}' for i in range(len(by))]
    # Select useful columns
    data1 = data[by + [value, weight]]
    # Encode column names
    data1.columns = data1.columns.to_series().replace(encodedict).tolist()
    # Remove missing weights
    data1 = data1.dropna(subset=['value'], how='any')
    # Calculate val*weight
    data1['val*w'] = data1['value'] * data1['weight']
    # Calculate sum(val*weight)
    data1['sum(val*w)'] = data1.groupby(by_)['val*w'].transform('sum')
    # Calculate sum(weight)
    data1['sum(w)'] = data1.groupby(by_)['weight'].transform('sum')
    # Calculate sum(val*weight)/sum(weight)
    data1['wavg'] = data1['sum(val*w)'] / data1['sum(w)']
    # Drop duplicates
    data1 = data1.drop_duplicates(by_)
    # Keep columns
    data1 = data1[by_ + ['wavg']]
    # Decode variable names
    data1.columns = data1.columns.to_series().replace(decodedict).tolist()
    return data1


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


def des(data: Union[pd.Series, pd.DataFrame],
        dropna: bool=False,
        format: Union[None,str,list]=',.2f',
        columns: str='stats',
        html: bool=False,
        **kwargs):
    """Enhanced pandas describe() with NA counts and better formating

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Input data
    dropna : bool, default: False
        If True, NA counts and percentages are displayed.
    format : None or str or list, default: None
        Format of statistics for each variable. If data is pd.Series, format
        must be either a str or a list of a single str.
    """
    assert isinstance(data, (pd.Series, pd.DataFrame)), 'data is not pandas.Series or pandas.DataFrame'
    assert columns in ['stats', 'vars'], 'Invalid value for columns'
    out = data.describe(**kwargs)
    # Change format
    if isinstance(data, pd.Series):
        if pd.api.types.is_list_like(format) and len(format) == 1:
            outstr_wocnt = out.drop(index='count').apply(
                lambda x: _format_to_str(x, format[0])
            )
        elif isinstance(format, str):
            outstr_wocnt = out.drop(index='count').apply(
                lambda x: _format_to_str(x, format)
            )
        else:
            raise TypeError('format is invalid for pandas.Series')
        outstr_cnt = pd.Series(
            [_format_to_str(out.loc['count'], ',.0f')], index=['count']
        )
    elif isinstance(data, pd.DataFrame):
        if isinstance(format, str):
            outstr_wocnt = out.drop(index='count').apply(
                lambda x: _format_to_str(x, format)
            )
        elif pd.api.types.is_list_like(format):
            assert len(format) == len(data.columns), \
                'format does not have the same length as data'
            outstr_wocnt = pd.concat([
                _format_to_str(out.drop(index='count').iloc[:, i], format[i])
                for i in range(len(out.columns))
            ], axis=1)
        else:
            raise TypeError('format is not list-like or a str')
        outstr_cnt = _format_to_str(out.loc['count'], ',.0f').to_frame().T
    else:
        raise TypeError('data should be pandas.Series or pandas.DataFrame')
    outstr = pd.concat([outstr_cnt, outstr_wocnt], axis=0)
    # Add NA counts
    if not dropna:
        na_count = data.isna().sum(axis=0)
        na_pct = na_count / out.loc['count'] * 100
        outstr.loc['#NA'] = _format_to_str(na_count, ',d')
        outstr.loc['%NA'] = _format_to_str(na_pct, '.2f', suffix='%')
    if isinstance(data, pd.Series):
        outstr = outstr.to_frame(data.name)
    if columns == 'stats':
        outstr = outstr.T
    # Output
    if html:
        return outstr.style.set_table_styles(
            [
                dict(selector='td', props=[('text-align', 'right')]),
                dict(selector='thead>tr>th', props=[('text-align', 'center')])
            ]
        )
    else:
        return outstr



def groupby_apply(
        data: pd.DataFrame,
        by: list,
        func: callable,
        colargs: list,
        otherargs: tuple,
        colout: list
):
    """
    Fast groupby-apply using numba

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe
    by : list of str
        Names of columns to group by
    func : callable numba function
        The numba function to be applied to each group. The first argument is
        a (2d) numpy array whose columns match colargs. Other arguments are
        given via otherargs.
    colargs : list of str
        Names of columns to be sent to the numba function as a 2d numpy array
    otherargs : tuple
        Other arguments (e.g. constant) to be sent to the numba function. Its
        order should match the arguments of the numba function
    colout : list of str
        Names of output columns

    Returns
    -------
    pd.DataFrame
    """
    assert isinstance(data, pd.DataFrame), 'data is not a pandas.DataFrame'
    assert isinstance(by, list), 'by is not a list'
    for i, s in enumerate(by):
        assert isinstance(s, str), f'The {i+1}th element in by is not a str'
    assert isinstance(colargs, list), 'colargs is not a list'
    for i, s in enumerate(colargs):
        assert isinstance(s, str), f'The {i+1}th element in colargs is not a str'
    assert isinstance(otherargs, tuple), 'otherargs is not a tuple'
    assert isinstance(colout, list), 'colout is not a list'
    for i, s in enumerate(colout):
        assert isinstance(s, str), f'The {i+1}th element in colout is not a str'

    # Match variable names
    by_name2id = {x: f'by{i}' for i, x in enumerate(by)}
    by_id2name = {f'by{i}': x for i, x in enumerate(by)}
    colargs_name2id = {x: f'colargs{i}' for i, x in enumerate(colargs)}
    # New column names
    by_id = [f'by{i}' for i, x in enumerate(by)]
    colargs_id = [f'colargs{i}' for i, x in enumerate(colargs)]
    # # Extract data
    # datatmp = data[by + colargs].sort_values(by)
    # # Rename
    # datatmp = datatmp.rename(columns=by_name2id).rename(columns=colargs_name2id)
    # Extract data and rename columns
    datatmp = pd.concat([
        data[by].rename(columns=by_name2id),
        data[colargs].rename(columns=colargs_name2id)
    ], axis=1).sort_values(by_id)
    # Get group id
    datatmp['grpid'] = datatmp.groupby(by_id).ngroup()
    # Create a link between by variables and group id
    grplk = datatmp[by_id + ['grpid']].drop_duplicates()
    # Apply the lower-level numba groupby function
    funcresnp = _groupby_apply_np(
        datatmp[['grpid'] + colargs_id].to_numpy(),
        func,
        otherargs
    )
    funcres = pd.DataFrame(funcresnp, columns=['grpid'] + colout)
    funcres['grpid'] = funcres['grpid'].astype('int')
    # Merge by-vars
    dataout = grplk.merge(funcres, on='grpid', how='left', validate='1:m')
    dataout = dataout.rename(columns=by_id2name)
    dataout = dataout.drop(columns=['grpid'])
    return dataout



@njit
def _groupby_apply_np(data, func, otherargs):
    """
    Groupby-Apply using numba given group id

    Parameters
    ----------
    data : np.ndarray
        The first column is group id. Other columns are sent to func as a 2d
        numpy array.
    func : callable
    otherargs: tuple

    Returns
    -------
    np.ndarray
    """
    ngroups = int(data[-1,0])+1   # Number of groups
    nrows = data.shape[0]    # Number of rows
    reslist = []
    istart = 0
    for k in range(ngroups):
        # Find start and end rows of the group
        # (istart point to the start and iend-1 point to the end
        iend = istart + 1
        while iend < nrows and data[iend-1,0] == data[iend,0]:
            iend += 1
        res = func(data[istart:iend,1:], *otherargs)
        reslist.append(np.hstack((np.array([k]), res)))
        # Move to the next group
        istart = iend
    assert len(reslist) == ngroups
    return reslist



def _format_to_str(data: Union[float, pd.Series],
                   format: str,
                   prefix: str='',
                   suffix: str='') -> Union[str, pd.Series]:
    """Format data (series or number) into str.

    Parameters
    ----------
    data : float or pandas.Series
    format : str
    prefix : str, default: ''
    suffix : str, default: ''

    Returns
    -------
    str or pd.Series
    """
    formatfunc = lambda x: (prefix + '{:' + format + '}' + suffix).format(x)
    if isinstance(data, pd.Series):
        return data.apply(formatfunc)
    elif pd.api.types.is_scalar(data):
        return formatfunc(data)