import pytest
import numpy as np
import pandas as pd
from pyempfin.dfutils import keepna, dropvar, rename

@pytest.fixture
def df():
    return pd.DataFrame(np.random.rand(10, 3), columns=['a', 'b', 'c'])


def test_rename(df):
    df1 = rename(df, a='A', b='B')
    assert df1.columns.tolist() == ['A', 'B', 'c']

def test_dropvar(df):
    df1 = dropvar(df, 'a b')
    assert df1.columns.tolist() == ['c']
    with pytest.raises(AssertionError):
        df2 = dropvar(df, 'ab')


def test_keepna(df):
    df = pd.DataFrame(np.random.rand(10, 3), columns=['a','b','c'])
    df.loc[3, 'a'] = np.nan
    df.loc[4, 'a'] = np.nan
    df.loc[4, 'b'] = np.nan
    df.loc[5, 'a'] = np.nan
    df.loc[5, 'b'] = np.nan
    df.loc[5, 'c'] = np.nan

    df1 = keepna(df, subset=['a'], how='any')
    assert df1.index.tolist() == [3, 4, 5]

    df2 = keepna(df, subset=['a', 'b'], how='any')
    assert df2.index.tolist() == [3, 4, 5]

    df3 = keepna(df, subset=['a', 'b'], how='all')
    assert df3.index.tolist() == [4, 5]

    df4 = keepna(df, subset=['a', 'b', 'c'], how='any')
    assert df4.index.tolist() == [3, 4, 5]

    df5 = keepna(df, subset=['a', 'b', 'c'], how='all')
    assert df5.index.tolist() == [5]