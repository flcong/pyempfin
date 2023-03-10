import pyempfin.datautils as pf
from pandas.testing import assert_frame_equal
from numpy.testing import assert_array_equal
import math
import numpy as np
import pandas as pd
import pytest
from numba import njit
import numba

@njit
def test():
    a = [np.ones((4,4)), np.zeros((4,4))]
    return [x[1:3,:] for x in a]
test()

class TestUtils:

    def test_des(self):
        df = pd.DataFrame({
            'x': [0, 1, np.nan, 3]
        })
        dftest = pf.des(df)
        assert dftest['%NA'].iloc[0] == '25.00%'

    def test_groupby_apply_nb(self):
        testarr = np.hstack((
            np.array([0]*15 + [1]*10)[:, np.newaxis],
            np.random.rand(25, 2)
        ))
        # Func that takes 1 argument and returns a number (wrapped in 2d ndarray)
        restest1 = pf._groupby_apply_nb(
            testarr[:,0], [testarr[:,1:2]], _mean, ()
        )
        restest1 = pd.DataFrame(
            [x[0] for x in restest1], columns=['grpid', 'mean']
        )
        rescorr1 = pd.DataFrame(testarr[:,:2], columns=['grpid', 'x']) \
            .groupby('grpid').apply(
            lambda x: _mean([x[['x']].to_numpy()])[0][0]
        )
        rescorr1 = pd.DataFrame(
            rescorr1.tolist(), index=rescorr1.index, columns=['mean']
        ).reset_index(drop=False)
        assert_frame_equal(restest1, rescorr1)
        # Func that takes 1 argument with additional arguments and
        # returns a row (wrapped in 2d ndarray)
        restest2 = pf._groupby_apply_nb(
            testarr[:,0], [testarr[:,1:]], _ols_njit, (10,)
        )
        restest2 = pd.DataFrame(
            [x[0] for x in restest2], columns=['grpid', 'coef0', 'coef1']
        )
        rescorr2 = pd.DataFrame(testarr, columns=['grpid', 'y', 'x'])\
            .groupby('grpid').apply(
            lambda x: _ols_njit([x[['y', 'x']].to_numpy()], 10)[0]
        )
        rescorr2 = pd.DataFrame(
            rescorr2.tolist(), index=rescorr2.index, columns=['coef0', 'coef1']
        ).reset_index(drop=False)
        assert_frame_equal(restest2, rescorr2)
        # Func that takes 2 argument (in a list) with additional arguments and
        # returns a row (wrapped in 2d ndarray)
        restest3 = pf._groupby_apply_nb(
            testarr[:,0], [testarr[:,1:2], testarr[:,2:3]], _ols2_njit, (10,)
        )
        restest3 = pd.DataFrame(
            [x[0] for x in restest3], columns=['grpid', 'coef0', 'coef1']
        )
        assert_frame_equal(restest3, rescorr2)
        # Func that returns higher dimensional ndarray


        @njit
        def _wrong_func(datalist):
            return np.random.rand(2, 2, 2)


        msg = 'The func must return a 2d ndarray'
        with pytest.raises(numba.core.errors.TypingError, match=msg):
            pf._groupby_apply_nb(
                testarr[:,0], [testarr[:,1:]], _wrong_func, ())




@njit
def _ols_njit(datalist, scaley):
    """Single-variable OLS regression with constant.
    The first column is y and the second column is x."""
    data = datalist[0]
    # Remove NA
    nnanmask = (~np.isnan(data[:, 0])) & (~np.isnan(data[:, 1]))
    ytmp = data[nnanmask, 0] * scaley
    xtmp = data[nnanmask, 1]
    N = ytmp.shape[0]
    if N >= 5:
        X = np.column_stack((np.ones(N), xtmp))
        XX = X.T @ X
        beta = np.linalg.inv(XX) @ X.T @ ytmp
        return np.array([list(beta)])
    else:
        return np.array([[np.nan, np.nan]])



@njit
def _ols2_njit(datalist, scaley):
    """Single-variable OLS regression with constant.
    The first column is y and the second column is x."""
    datay = datalist[0]
    datax = datalist[1]
    # Remove NA
    nnanmask = np.isfinite(datay[:,0]) & np.isfinite(datax[:,0])
    ytmp = datay[nnanmask,0] * scaley
    xtmp = datax[nnanmask,0]
    N = ytmp.shape[0]
    if N >= 5:
        X = np.column_stack((np.ones(N), xtmp))
        XX = X.T @ X
        beta = np.linalg.inv(XX) @ X.T @ ytmp
        return np.array([list(beta)])
    else:
        return np.array([[np.nan, np.nan]])


@njit
def _corr(datalist):
    data = datalist[0]
    nnanmask = np.isfinite(data[:,0]) & np.isfinite(data[:,1])
    if nnanmask.sum() > 0:
        m1 = data[nnanmask,0].mean()
        m2 = data[nnanmask,1].mean()
        std1 = data[nnanmask,0].std()
        std2 = data[nnanmask,1].std()
        if std1 > 0 and std2 > 0:
            return np.array([[
                (data[nnanmask,0]-m1).dot(data[nnanmask,1]-m2) / (
                        nnanmask.sum() * std1 * std2
                )
            ]])
        else:
            return np.array([[np.nan]])
    else:
        return np.array([[np.nan]])


@njit
def _corr2(datalist):
    data1 = datalist[0]
    data2 = datalist[1]
    nnanmask = np.isfinite(data1[:,0]) & np.isfinite(data2[:,0])
    if nnanmask.sum() > 0:
        m1 = data1[nnanmask,0].mean()
        m2 = data2[nnanmask,0].mean()
        std1 = data1[nnanmask,0].std()
        std2 = data2[nnanmask,0].std()
        if std1 > 0 and std2 > 0:
            return np.array([[
                (data1[nnanmask,0]-m1).dot(data2[nnanmask,0]-m2) / (
                        nnanmask.sum() * std1 * std2
                )
            ]])
        else:
            return np.array([[np.nan]])
    else:
        return np.array([[np.nan]])



@njit
def _mean(datalist):
    data = datalist[0]
    res = 0
    cnt = 0
    for i in range(data.size):
        if np.isfinite(data[i,0]):
            res += data[i,0]
            cnt += 1
    if cnt > 0:
        return np.array([[res / cnt]])
    else:
        return np.array([[np.nan]])

@njit
def _std(datalist):
    datam = _mean(datalist)[0][0]
    data = datalist[0]
    if np.isfinite(datam):
        res = 0
        cnt = 0
        for i in range(data.size):
            if np.isfinite(data[i,0]):
                res += (data[i,0] - datam) * (data[i,0] - datam)
                cnt += 1
        if cnt > 3:
            return np.array([[np.sqrt(res / cnt)]])
        else:
            return np.array([[np.nan]])
    else:
        return np.array([[np.nan]])

@pytest.fixture
def firm_bond_data():
    # Create the test data set
    nfirms = 2
    nbonds = 10
    ndates = 20
    droppct = 1/6
    napct = 1/3
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
    dropmask = np.random.choice(
        df.index, size=int(df.shape[0]*droppct), replace=False
    )
    df = df.drop(index=dropmask)
    # Randomly set x as NA
    xnamask = np.random.choice(
        df.index, size=int(df.shape[0]*napct), replace=False
    )
    df.loc[xnamask, 'x'] = np.nan
    # Randomly set y as NA
    ynamask = np.random.choice(
        df.index, size=int(df.shape[0]*napct), replace=False
    )
    df.loc[ynamask, 'y'] = np.nan
    return df


class TestAgg:

    def test_groupbyapply_simple(self):
        df = pd.DataFrame({
            'x': [0, 0, 0, 0, 1, 1, 1, 1, 1],
            'y': [2, 2, 3, 3, 2, 2, 2, 3, 3],
            'var1': [0, 1, 3, 4, 1, 2, 3, 4, 4],
        })

        @njit
        def testfunc(datalist):
            data = datalist[0]
            tmp = data[data[:,0] < data[0,1], 0]
            if tmp.shape[0] == 0:
                return np.array([[-999]])
            else:
                return np.array([[tmp.max()]])

        dfout = pf.groupby_apply(
            data=df,
            by=['x', 'y'],
            func=testfunc,
            colargs=[['var1', 'y']],
            otherargs=(),
            colout=['out']
        )

        dfcor = pd.DataFrame({
            'x': [0, 0, 1, 1],
            'y': [2, 3, 2, 3],
            'out': [1, -999, 1, -999]
        })

        assert_frame_equal(dfout, dfcor)

    def test_groupbyapply_simple_transform(self):
        N = 1000
        Ngx = 10
        Ngy = 20
        df = pd.DataFrame({
            'x': np.random.randint(0, Ngx, N),
            'y': np.random.randint(0, Ngy, N),
            'var1': np.random.rand(N),
        })
        df.sort_values(['x', 'y'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        @njit
        def testfunc(datalist):
            """Return the same size as input"""
            return datalist[0]

        dfout = pf.groupby_apply(
            data=df,
            by=['x'],
            func=testfunc,
            colargs=[['var1']],
            otherargs=(),
            colout=['var1']
        )

        assert_frame_equal(df[['x', 'var1']], dfout)

    def test_groupbyapply(self, firm_bond_data):
        df = firm_bond_data
        # Check sorted
        msg = 'data is not sorted on columns in "by"'
        with pytest.raises(ValueError, match=msg):
            dftest1 = pf.groupby_apply(
                data=df, by=['firm', 'date'], func=_mean, colargs=[['x']],
                otherargs=(), colout=['mean']
            )
        # ---------- Func returning a single number (wrapped in a 2d ndarray)
        # Correct result
        df.sort_values(['firm', 'date'], inplace=True)
        dfcorr1 = df.groupby(['firm', 'date']).apply(
            lambda x: _mean([x[['x']].to_numpy()])[0][0]
        ).to_frame('mean').reset_index(drop=False)
        # Test result
        dftest1 = pf.groupby_apply(
            data=df, by=['firm', 'date'], func=_mean, colargs=[['x']],
            otherargs=(), colout=['mean']
        )
        assert_frame_equal(dfcorr1, dftest1)
        # ---------- Func with 1 argument (in list) and returning two numbers
        df.sort_values(['firm', 'date'], inplace=True)
        dfcorr2 = df.groupby(['firm', 'date']).apply(
            lambda x: _ols_njit([x[['y', 'x']].to_numpy()], 10)
        )
        dfcorr2 = pd.DataFrame(
            [x[0] for x in dfcorr2],
            index=dfcorr2.index,
            columns=['const', 'x']
        ).reset_index(drop=False)
        dftest2 = pf.groupby_apply(
            data=df, by=['firm', 'date'], func=_ols_njit, colargs=[['y', 'x']],
            otherargs=(10, ), colout=['const', 'x']
        )
        assert_frame_equal(dfcorr2, dftest2)
        # ---------- Func with 2 arguments (in list) and return two numbers
        df.sort_values(['firm', 'date'], inplace=True)
        dftest3 = pf.groupby_apply(
            data=df, by=['firm', 'date'], func=_ols2_njit, colargs=[['y'], ['x']],
            otherargs=(10, ), colout=['const', 'x']
        )
        assert_frame_equal(dfcorr2, dftest3)
        # ---------- Func with arguments and returning a 2d ndarray

        @njit
        def _test_func(datalist, nobs):
            data = datalist[0]
            if data.shape[0] >= nobs:
                return data[:nobs,:]
            else:
                return data

        # Correct result
        df.sort_values(['firm', 'date'], inplace=True)
        dfcorr4 = df.groupby(['firm', 'date']).apply(
            lambda x: _test_func([x[['y', 'x']].to_numpy()], 10)
        ).explode()
        dfcorr4 = pd.DataFrame(
            [x for x in dfcorr4],
            index=dfcorr4.index,
            columns=['y_', 'x_']
        ).reset_index(drop=False)
        # Test result
        df.sort_values(['firm', 'date'], inplace=True)
        dftest3 = pf.groupby_apply(
            data=df, by=['firm', 'date'], func=_test_func, colargs=[['y', 'x']],
            otherargs=(10, ), colout=['y_', 'x_']
        )
        assert_frame_equal(dfcorr4, dftest3)

    def test_rangestat(self, firm_bond_data):
        df = firm_bond_data

        lb = -12.5
        ub = 2.5
        # ---------- Prepare dates for correct answer
        dfsmpl = df[['firm', 'bond', 'date']].copy()
        smpldays = df[['date']].drop_duplicates()
        smpldays['smpl_date'] = smpldays['date'].apply(
            lambda x: np.arange(x+math.ceil(lb), x+math.floor(ub)+1)
        )
        smpldays= smpldays.explode('smpl_date')
        dfcorr = dfsmpl.merge(
            smpldays, on=['date'], how='left'
        )
        dfcorr = dfcorr.merge(
            df[['firm', 'bond', 'date', 'x', 'y']].rename(
                columns={'date': 'smpl_date'}
            ),
            on=['firm', 'bond', 'smpl_date'],
            how='inner'
        ).sort_values(['firm', 'bond', 'date', 'smpl_date'])
        # ---------- Func returning a single number (wrapped in a 2d ndarray)
        dfcorr1 = dfcorr.groupby(['firm', 'bond', 'date']).apply(
            lambda x: _std([x[['x']].to_numpy()])[0][0]
        ).to_frame('std').reset_index(drop=False)
        dftest1 = pf.rangestat(
            data=df, by=['firm', 'bond'], timevar='date', interval=(lb, ub),
            func=_std, colargs=['x'], otherargs=(), colout=['std']
        )
        assert_frame_equal(dfcorr1, dftest1)
        # ---------- Func with arguments and returning two numbers
        dfcorr2 = dfcorr.groupby(['firm', 'bond', 'date']).apply(
            lambda x: _ols_njit([x[['y', 'x']].to_numpy()], 10)
        )
        dfcorr2 = pd.DataFrame(
            [x[0] for x in dfcorr2],
            index=dfcorr2.index,
            columns=['const', 'x']
        ).reset_index(drop=False)
        dftest2 = pf.rangestat(
            data=df, by=['firm', 'bond'], timevar='date', interval=(lb, ub),
            func=_ols_njit, colargs=['y', 'x'], otherargs=(10,), colout=['const', 'x']
        )
        assert_frame_equal(dfcorr2, dftest2)