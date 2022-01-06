from pyempfin.datautils import *
from pandas.testing import assert_frame_equal

import numpy as np
import pandas as pd

def test_groupbyapply_simple():
    df = pd.DataFrame({
        'x': [0, 0, 0, 0, 1, 1, 1, 1, 1],
        'y': [2, 2, 3, 3, 2, 2, 2, 3, 3],
        'var1': [0, 1, 3, 4, 1, 2, 3, 4, 4],
    })

    @njit
    def testfunc(data):
        tmp = data[data[:,0] < data[0,1], 0]
        if tmp.shape[0] == 0:
            return np.array([-999])
        else:
            return np.array([tmp.max()])

    dfout = groupby_apply(
        data=df,
        by=['x', 'y'],
        func=testfunc,
        colargs=['var1', 'y'],
        otherargs=(),
        colout=['out']
    )

    dfcor = pd.DataFrame({
        'x': [0, 0, 1, 1],
        'y': [2, 3, 2, 3],
        'out': [1, -999, 1, -999]
    })

    assert_frame_equal(dfout, dfcor)




def test_groupbyapply():
    # Create the test data set
    nfirms = 10
    nbonds = 30
    ndates = 20
    df = pd.DataFrame(
        {'e': np.random.randn(nfirms*nbonds*ndates),
         'x': np.random.rand(nfirms*nbonds*ndates)},
        pd.MultiIndex.from_product(
            [np.arange(nfirms), np.arange(nbonds), np.arange(ndates)],
            names=['firm', 'bond', 'date']
        )
    ).reset_index(drop=False)
    df['y'] = 5 + df['x'] * 10 + df['e']
    np.random.seed(0)
    # Randomly drop observations
    dropmask = np.random.choice(df.index, size=1000, replace=False)
    df = df.drop(index=dropmask)
    # Randomly set x as NA
    xnamask = np.random.choice(df.index, size=2000, replace=False)
    df.loc[xnamask, 'x'] = np.nan
    # Randomly set y as NA
    ynamask = np.random.choice(df.index, size=2000, replace=False)
    df.loc[ynamask, 'y'] = np.nan
    # Function to apply


    @njit
    def ols_njit(data, scaley):
        """Single-variable OLS regression with constant.
        The first column is y and the second column is x."""
        # Remove NA
        nnanmask = (~np.isnan(data[:,0])) & (~np.isnan(data[:,1]))
        ytmp = data[nnanmask, 0] * scaley
        xtmp = data[nnanmask, 1]
        N = ytmp.shape[0]
        if N >= 10:
            X = np.column_stack((np.ones(N), xtmp))
            XX = X.T @ X
            beta = np.linalg.inv(XX) @ X.T @ ytmp
            return beta
        else:
            return np.nan * np.zeros(2)


    # Correct result
    res_njit = df.groupby(['firm', 'date']).apply(
        lambda x: ols_njit(x[['y', 'x']].to_numpy(), 10)
    )
    res_njit = pd.DataFrame(
        [x for x in res_njit],
        index=res_njit.index,
        columns=['const', 'x']
    ).reset_index(drop=False)
    # Test result
    res_test = groupby_apply(
        data=df, by=['firm', 'date'], func=ols_njit, colargs=['y', 'x'],
        otherargs=(10, ), colout=['const', 'x']
    )
    assert_frame_equal(res_njit, res_test)
