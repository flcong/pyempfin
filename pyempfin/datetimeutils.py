import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from typing import Union
from numba import njit
import datetime

def intck(freq: str, fdate: Union[pd.Series,pd.Timestamp], ldate: Union[pd.Series,pd.Timestamp]) -> Union[pd.Series,pd.Timestamp]:
    """Return the number of months or days between two dates, similar to SAS intck function
    
    Parameters
    ----------
    freq : str
        Allowed values are "month" and "day"
    fdate : pd.Series or pd.Timestamp
        Start date or a series of start dates.
    ldate : pd.Series or pd.Timestamp
        End date or a series of end dates.
    """    
    if isinstance(fdate, pd.Series) and isinstance(ldate, pd.Series):
        if freq == 'month':
            return (ldate.dt.year - fdate.dt.year) * 12 + (ldate.dt.month - fdate.dt.month)
        elif freq == 'day':
            return (ldate - fdate).dt.days

    elif isinstance(fdate, pd.Series) and isinstance(ldate, pd.Timestamp):
        if freq == 'month':
            return (ldate.year - fdate.dt.year) * 12 + (ldate.month - fdate.dt.month)
        elif freq == 'day':
            return (ldate - fdate).dt.days

    elif isinstance(fdate, pd.Timestamp) and isinstance(ldate, pd.Series):
        if freq == 'month':
            return (ldate.dt.year - fdate.year) * 12 + (ldate.dt.month - fdate.month)
        elif freq == 'day':
            return (ldate - fdate).dt.days
        
    elif isinstance(fdate, pd.Timestamp) and isinstance(ldate, pd.Timestamp):
        if freq == 'month':
            return (ldate.year - fdate.year) * 12 + (ldate.month - fdate.month)
        elif freq == 'day':
            return (ldate - fdate) / np.timedelta64(1, 'D')
        
    elif isinstance(fdate, datetime.date) and isinstance(ldate, datetime.date):
        if freq == 'month':
            return (ldate.year - fdate.year) * 12 + (ldate.month - fdate.month)
        elif freq == 'day':
            return (ldate - fdate) / datetime.timedelta(days=1)

def sas2date(d: pd.Series) -> pd.Series:
    """Convert a pandas series of SAS date to pandas datetime

    Parameters
    ----------
    d : pandas.Series
        The series containing SAS date in numeric form, i.e. the number of days
        from 1960-01-01. This is typically from a pandas dataframe that is read
        using the pyreadstat.read_sas7bdat function.
    
    Returns
    -------
    pandas.Series
        A pandas series of pandas datetime.
    """
    return pd.to_datetime('1960-01-01') + pd.to_timedelta(d, unit='D')

def date2sas(d: pd.Series) -> pd.Series:
    """Convert a pandas series of datetime to SAS date in int

    Parameters
    ----------
    d : pandas.Series
        Pandas datetime type.

    Returns
    -------
    pandas.Series
        Integer representing number of days from 1960-01-01.
    """
    return (d - pd.Timestamp('1960-01-01')).dt.days

def ymd2date(d: pd.Series) -> pd.Series:
    """Convert a pandas series of yyyymmdd date to pandas datetime

    Parameters
    ----------
    d : pandas.Series
        The series containing date in the form of 8-digit integer yyyymmdd.
    
    Returns
    -------
    pandas.Series
        A pandas series of pandas datetime.
    """
    return pd.to_datetime(d, format='%Y%m%d')

# Pandas datetime to YYYYMMDD
def date2ymd(d: Union[pd.Series,pd.Timestamp], type: str='int') -> Union[pd.Series,int,str]:
    """Convert a pandas series of pandas datetime into yyyymmdd format, either
    string or integer type.

    Parameters
    ----------
    d : pandas.Series or pandas.Timestamp
        A pandas series of pandas datatime or a single pandas.Timestamp
    type : str, default: 'int'
        Either 'int' or 'str' specifying the output type.

    Returns
    -------
    pandas.Series
        A pandas series of type str or int depending on the parameter `type` or
        a single str or int if the input is a pandas.Timestamp.
    """
    if isinstance(d, pd.Series) and pd.api.types.is_datetime64_ns_dtype(d):
        if type == 'str':
            return (d.dt.year * 10000 + d.dt.month * 100 + d.dt.day).astype(type).str[:8]
        else:
            return (d.dt.year * 10000 + d.dt.month * 100 + d.dt.day).astype(type)
    elif isinstance(d, pd.Timestamp):
        return pd.Series([d.year * 10000 + d.month * 100 + d.day]).astype(type).iloc[0]
    else:
        raise ValueError("Invalid argument: Neither a pandas Timestamp nor a Series of pandas Timestamp")
    
def monthstart(d: pd.Series) -> pd.Series:
    """Convert a pandas series of pandas datetime into the first dates of month

    Parameters
    ----------
    d : pandas.Series
        A pandas series of pandas datetime
    
    Returns
    -------
    pandas.Series
        A pandas series of pandas datetime containing the first dates of each
        month.
    """
    return d.astype('datetime64[M]')


# Return month end
def monthend(d: pd.Series) -> pd.Series:
    """Convert a pandas series of pandas datetime into the last dates of month

    Parameters
    ----------
    d : pandas.Series
        A pandas series of pandas datetime
    
    Returns
    -------
    pandas.Series
        A pandas series of pandas datetime containing the last dates of each
        month.
    """
    return d.dt.to_period('M').dt.to_timestamp('M')


# def yrdif(start: Union[pd.Timestamp,datetime.date,pd.Series],
#           end: Union[pd.Timestamp,datetime.date,pd.Series],
#           basis: str='ACT/ACT') -> Union[float,np.array]:
#     """Mimic the yrdif function in SAS to calculate number of years between two
#     dates (assuming end-of-day).
#
#     Parameters
#     ----------
#     start : pandas.Timestamp, datetime.date, or a pandas.Series of them
#         The start date, i.e. the time 00:00.00 on the beginning of the date
#     end : pandas.Timestamp, datetime.date, or a pandas.Series of them
#         The end date, i.e. the time 00:00.00 on the beginning of the date
#     basis : str, default: 'ACT/ACT'
#         The day count convention to calculate the year fraction according to
#         the SAS function. Currently only the 'ACT/ACT' convention is
#         implemented.
#
#     Returns
#     -------
#     float or a pandas series of float
#         The number of years between the two days
#     """
#     if isinstance(start, pd.Timestamp):
#         start = np.datetime64(start)
#     if isinstance(end, pd.Timestamp):
#         end = np.datetime64(end)
#     if isinstance(start, np.datetime64) and isinstance(end, np.datetime64):
#         return _yrdif([start], [end], basis)[0]
#     elif isinstance(start, pd.Series) and isinstance(end, pd.Series) and start.shape[0] == end.shape[0]:
#         # TODO: Speed up this part using numba?
#         return pd.Series(
#             Parallel(n_jobs=4)(delayed(_yrdif)(
#                 start[i], end[i], basis) for i in range(start.shape[0])),
#             index=start.index)
#

@njit
def is_leap_year(data: int) -> bool:
    """Return if a year is a leap year"""
    return (np.mod(data, 400) == 0) | \
           ((np.mod(data, 4) == 0) & (np.mod(data, 100) != 0))


@njit
def _datdif_diff_month(start, end):
    """Return number of days between two dates in different months (assume end-of-day)"""
    yid = 0
    mid = 1
    did = 2
    dinm_noleap = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    dinm_leap = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if is_leap_year(start[yid]):
        dinm = dinm_leap
    else:
        dinm = dinm_noleap
    outdays = 0
    # Days in the first month
    outdays += dinm[start[mid]] - start[did]
    # Days in the last month
    outdays += end[did]
    # Days in between
    for m in range(start[mid]+1, end[mid]):
        outdays += dinm[m]
    return outdays


@njit
def _datdif_same_year(start: Union[list,tuple,np.array], end: Union[list,tuple,np.array]):
    """Return number of days between two dates in the same year (assume end-of-day)"""
    yid = 0
    mid = 1
    did = 2
    if start[mid] == end[mid]:
        # The two days are in the same month
        return end[did] - start[did]
    elif start[mid] < end[mid]:
        return _datdif_diff_month(start, end)
    elif start[mid] > end[mid]:
        return -_datdif_diff_month(end, start)


@njit
def _yrdif_same_year(start: Union[list,tuple,np.array], end: Union[list,tuple,np.array]):
    """Return year fraction between two dates in the same year (assume end-of-day)"""
    yid = 0
    mid = 1
    did = 2
    ndays = _datdif_same_year(start, end)
    if is_leap_year(start[yid]):
        return ndays / 366
    else:
        return ndays / 365


@njit
def _datdif_diff_year(start, end):
    """Return number of days between two dates in different year (assume end-of-day)"""
    yid = 0
    outdays = 0
    # Days in the first year
    outdays += _datdif_same_year(start, (start[yid], 12, 31))
    # Days in the last year
    outdays += _datdif_same_year((end[yid], 1, 1), end) + 1
    # Days in between
    for y in range(start[yid] + 1, end[yid]):
        if is_leap_year(y):
            outdays += 366
        else:
            outdays += 365
    return outdays


@njit
def _yrdif_diff_year(start, end):
    """Return year fraction between two dates in different year (assume end-of-day)"""
    yid = 0
    yrfrac = 0
    # Days in the first year
    yrfrac += _datdif_same_year(start, (start[yid], 12, 31)) / \
              (366 if is_leap_year(start[yid]) else 365)
    # Days in the last year
    yrfrac += (_datdif_same_year((end[yid], 1, 1), end) + 1) / \
              (366 if is_leap_year(end[yid]) else 365)
    # Days in between
    yrfrac += end[yid] - start[yid] - 1
    return yrfrac


@njit
def _datdif(start: Union[list,tuple,np.array], end: Union[list,tuple,np.array]):
    """Return the number of days between two dates (assume end-of-day)"""
    yid = 0
    mid = 1
    did = 2
    if start[yid] == end[yid]:
        return _datdif_same_year(start, end)
    elif start[yid] < end[yid]:
        return _datdif_diff_year(start, end)
    else:
        return -_datdif_diff_year(end, start)


@njit
def _yrdif(start: Union[list,tuple,np.array], end: Union[list,tuple,np.array]):
    """Return the year fraction between two dates (assume end-of-day)"""
    yid = 0
    mid = 1
    did = 2
    if start[yid] == end[yid]:
        return _yrdif_same_year(start, end)
    elif start[yid] < end[yid]:
        return _yrdif_diff_year(start, end)
    elif start[yid] > end[yid]:
        return -_yrdif_diff_year(end, start)


@njit
def _datdif_arr(start: np.array, end: np.array):
    """Same as _datdif but input is an array of dates"""
    N = start.shape[0]
    out = np.zeros(N)
    out.fill(np.nan)
    for i in range(N):
        out[i] = _datdif(start[i,:], end[i,:])
    return out


@njit
def _yrdif_arr(start: np.array, end: np.array):
    """Same as _yrdif but input is an array of dates"""
    N = start.shape[0]
    out = np.zeros(N)
    out.fill(np.nan)
    for i in range(N):
        out[i] = _yrdif(start[i,:], end[i,:])
    return out


def datdif(
        start: Union[pd.Series,pd.Timestamp,datetime.date],
        end: Union[pd.Series,pd.Timestamp,datetime.date],
        basis: str='ACT/ACT'
):
    """Mimic the datdif function (ACT/ACT) in SAS to calculate number of days
    between two dates (assuming end-of-day).

    Parameters
    ----------
    start : pandas.Timestamp, datetime.date, or a pandas.Series of them
        The start date, i.e. the time 00:00.00 on the beginning of the date
    end : pandas.Timestamp, datetime.date, or a pandas.Series of them
        The end date, i.e. the time 00:00.00 on the beginning of the date
    basis : str, default: 'ACT/ACT'
        The day count convention to calculate the year fraction according to
        the SAS function. Currently only the 'ACT/ACT' convention is
        implemented.

    Returns
    -------
    float or a pandas series of float
        The number of years between the two days
    """
    singledatetypes = (pd.Timestamp, datetime.date)
    if isinstance(start, pd.Series) or isinstance(end, pd.Series):
        if isinstance(start, pd.Series) and isinstance(end, pd.Series):
            assert start.shape[0] == end.shape[0], 'Start and End has different lengths.'
            startin = np.vstack((start.dt.year, start.dt.month, start.dt.day)).T
            endin = np.vstack((end.dt.year, end.dt.month, end.dt.day)).T
        elif isinstance(start, pd.Series) and isinstance(end, singledatetypes):
            startin = np.vstack((start.dt.year, start.dt.month, start.dt.day)).T
            endin = np.tile([end.year, end.month, end.day], (start.shape[0], 1))
        elif isinstance(start, singledatetypes) and isinstance(end, pd.Series):
            startin = np.tile([start.year, start.month, start.day], (end.shape[0], 1))
            endin = np.vstack((end.dt.year, end.dt.month, end.dt.day)).T
        out = _datdif_arr(startin, endin)
    else:
        startin = (start.year, start.month, start.day)
        endin = (end.year, end.month, end.day)
        out = _datdif(startin, endin)
    return out


def yrdif(
        start: Union[pd.Series,pd.Timestamp,datetime.date],
        end: Union[pd.Series,pd.Timestamp,datetime.date],
        basis: str = 'ACT/ACT'
):
    """Mimic the yrdif function (ACT/ACT) in SAS to calculate number of years between two
    dates (assuming end-of-day).

    Parameters
    ----------
    start : pandas.Timestamp, datetime.date, or a pandas.Series of them
        The start date, i.e. the time 00:00.00 on the beginning of the date
    end : pandas.Timestamp, datetime.date, or a pandas.Series of them
        The end date, i.e. the time 00:00.00 on the beginning of the date
    basis : str, default: 'ACT/ACT'
        The day count convention to calculate the year fraction according to
        the SAS function. Currently only the 'ACT/ACT' convention is
        implemented.

    Returns
    -------
    float or a pandas series of float
        The number of years between the two days
    """
    singledatetypes = (pd.Timestamp, datetime.date)
    if isinstance(start, pd.Series) or isinstance(end, pd.Series):
        if isinstance(start, pd.Series) and isinstance(end, pd.Series):
            assert start.shape[0] == end.shape[0], 'Start and End has different lengths.'
            startin = np.vstack((start.dt.year, start.dt.month, start.dt.day)).T
            endin = np.vstack((end.dt.year, end.dt.month, end.dt.day)).T
        elif isinstance(start, pd.Series) and isinstance(end, singledatetypes):
            startin = np.vstack((start.dt.year, start.dt.month, start.dt.day)).T
            endin = np.tile([end.year, end.month, end.day], (start.shape[0], 1))
        elif isinstance(start, singledatetypes) and isinstance(end, pd.Series):
            startin = np.tile([start.year, start.month, start.day], (end.shape[0], 1))
            endin = np.vstack((end.dt.year, end.dt.month, end.dt.day)).T
        out = _yrdif_arr(startin, endin)
    else:
        startin = (start.year, start.month, start.day)
        endin = (end.year, end.month, end.day)
        out = _yrdif(startin, endin)
    return out


