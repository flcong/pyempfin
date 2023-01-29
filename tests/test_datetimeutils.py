from pyempfin.datetimeutils import *
import pandas as pd
import calendar
from pandas.testing import assert_series_equal

def test_yrdif():
    df1 = pd.DataFrame({
        'start': ['2010-08-27', '2010-08-27'],
        'end': ['2022-06-01', '2024-06-01']
    })
    df1['start'] = pd.to_datetime(df1['start'])
    df1['end'] = pd.to_datetime(df1['end'])
    df1['ttm'] = yrdif(df1['start'], df1['end'])

    def verify_func(start, end):
        res = 0
        if start.year != end.year:
            res += (pd.Timestamp(start.year, 12, 31).dayofyear - \
                start.dayofyear) / (366 if calendar.isleap(start.year) else 365)
            res += end.dayofyear / \
                (366 if calendar.isleap(end.year) else 365)
            res += end.year - start.year - 1
        else:
            res += (end.dayofyear - start.dayofyear) / \
               (366 if calendar.isleap(start.year) else 365)
        return res

    df1['ttm_corr'] = df1.apply(
        lambda x: verify_func(x['start'], x['end']),
        axis=1
    )

    assert_series_equal(df1['ttm'], df1['ttm_corr'], check_names=False)

def test_monthstart():
    df = pd.DataFrame({
        'test': [
            '1990-01-01', '1990-01-31', '2000-02-29', '2000-06-30', '2000-11-01',
            '2000-11-15'
        ],
        'correct': [
            '1990-01-01', '1990-01-01', '2000-02-01', '2000-06-01', '2000-11-01',
            '2000-11-01'
        ]
    })
    df['test'] = pd.to_datetime(df['test'])
    df['correct'] = pd.to_datetime(df['correct'])
    assert_series_equal(monthstart(df['test']), df['correct'], check_names=False)

def test_monthend():
    df = pd.DataFrame({
        'test': [
            '1990-01-01', '1990-01-31', '2000-02-29', '2000-06-30', '2000-11-01',
            '2000-11-15'
        ],
        'correct': [
            '1990-01-31', '1990-01-31', '2000-02-29', '2000-06-30', '2000-11-30',
            '2000-11-30'
        ]
    })
    df['test'] = pd.to_datetime(df['test'])
    df['correct'] = pd.to_datetime(df['correct'])
    assert_series_equal(monthend(df['test']), df['correct'], check_names=False)
