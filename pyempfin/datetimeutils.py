import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from typing import Union

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
def date2ymd(d: Union[pd.Series,pd.Timestamp], type: str='int') -> pd.Series:
    """Convert a pandas series of pandas datetime into yyyymmdd format, either
    string or integer type.

    Parameters
    ----------
    d : pandas.Series
        A pandas series of pandas datatime
    type : str, default: 'int'
        Either 'int' or 'str' specifying the output type.

    Returns
    -------
    pandas.Series
        A pandas series of type str or int depending on the parameter `type`.
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


def yrdif(start: Union[pd.Timestamp,np.datetime64,pd.Series],
          end: Union[pd.Timestamp,np.datetime64,pd.Series],
          basis: str='ACT/ACT') -> Union[float,pd.Series]:
    """Mimic the yrdif function in SAS to calculate number of years between two
    dates.

    Parameters
    ----------
    start : pandas.Timestamp, numpy.datetime64, or a pandas.Series of them
        The start date, i.e. the time 00:00.00 on the beginning of the date
    end : pandas.Timestamp, numpy.datetime64, or a pandas.Series of them
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
    if isinstance(start, pd.Timestamp):
        start = np.datetime64(start)
    if isinstance(end, pd.Timestamp):
        end = np.datetime64(end)
    if isinstance(start, np.datetime64) and isinstance(end, np.datetime64):
        return _yrdif([start], [end], basis)[0]
    elif isinstance(start, pd.Series) and isinstance(end, pd.Series) and start.shape[0] == end.shape[0]:
        return pd.Series(
            Parallel(n_jobs=4)(delayed(_yrdif)(
                start[i], end[i], basis) for i in range(start.shape[0])),
            index=start.index)


def _yrdif(start: Union[pd.Timestamp,np.datetime64],
           end: Union[pd.Timestamp,np.datetime64],
           basis: str) -> float:
    """Lower level implementation of yrdif for a pair of dates.
    """
    if basis in ['ACT/ACT', 'Actual']:
        if isinstance(start, pd.Timestamp):
            start = np.datetime64(start)
        elif pd.isna(start):
            start = np.datetime64('NaT')
        if isinstance(end, pd.Timestamp):
            end = np.datetime64(end)
        elif pd.isna(end):
            end = np.datetime64('NaT')
        if not np.isnat(start) and not np.isnat(end) and start <= end:
            startyear = np.array([start]).tolist()[0].year
            endyear = np.array([end-1]).tolist()[0].year
            if startyear == endyear:
                res = (end - start) / np.timedelta64(1, 'D')
            else:
                res = 0
                # Days in the first year
                res += (np.datetime64(str(startyear+1)+'-01-01') - start) / \
                    np.timedelta64(1, 'D') / \
                    (366 if is_leap_year(startyear) else 365)
                # Days in the last year
                res += (end - np.datetime64(str(endyear-1)+'-12-31')) / \
                    np.timedelta64(1, 'D') / \
                    (366 if is_leap_year(endyear) else 365)
                # Days in the middle
                res += endyear - startyear - 1
            return res
        else:
            return np.nan
    else:
        raise ValueError('Invalid basis: ' + str(basis))


def is_leap_year(data: int) -> bool:
    """Return if a year is a leap year
    """
    return (np.mod(data, 400) == 0) | \
           ((np.mod(data, 4) == 0) & (np.mod(data, 100) != 0))
