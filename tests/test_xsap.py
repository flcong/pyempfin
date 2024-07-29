import pytest
import numpy as np
from pyempfin.xsap import winsor_njit, _newey_njit, estbeta, estbeta1m
from pyempfin.xsap import format_table, fmreg
from pyempfin.xsap import groupby_wavg, get_port_ret, tscssum
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import re

def assert_float_equal(x, y, tol=1e-15):
    return np.abs(x-y) < tol

def checkarr(arr: np.ndarray, orgarr: np.ndarray,
             lower: float, upper: float) -> bool:
    # Same missing values
    cond1 = (np.isfinite(arr) == np.isfinite(orgarr)).all()
    # Check maximum and minimum values
    cond2 = np.nanmax(arr) == upper
    cond3 = np.nanmin(arr) == lower
    # Check other numbers are not modified
    cond4 = (arr[(lower < arr) & (arr < upper)] ==
             orgarr[(lower < orgarr) & (orgarr < upper)]).all()
    return cond1 and cond2 and cond3 and cond4

@pytest.fixture
def xsstk():
    stk = pd.read_csv('xsstktestdata.csv')
    stk.columns = stk.columns.str.lower()
    stk['datemn'] = pd.to_datetime(stk['date'].astype(str)).dt.to_period('M')
    stk = stk.set_index(['permno', 'datemn']).sort_index()
    stk = stk[['exret']]
    return stk

@pytest.fixture
def ff():
    ff = pd.read_csv('ff.csv')
    ff['datemn'] = pd.to_datetime(ff['date'].astype(str)).dt.to_period('M')
    ff.drop(columns=['date'], inplace=True)
    ff.set_index('datemn', inplace=True)
    return ff

def test_winsor():
    # Test array
    arr = np.arange(0, 101)
    # Add missing
    arr = np.hstack((arr, np.array([np.nan]*10)))
    # Shuffle
    np.random.shuffle(arr)
    # Winsorize
    w1 = winsor_njit(arr, (0.052, 0.959), interpolation='inner')
    assert checkarr(w1, arr, lower=6.0, upper=95.0)
    w2 = winsor_njit(arr, (0.052, 0.959), interpolation='outer')
    assert checkarr(w2, arr, lower=5.0, upper=96.0)


def test_newey(ff):
    # Load data
    ffres = pd.read_stata('ff_mtkrf_newey.dta')
    data = ff['mktrf'].to_numpy()
    data = data - data.mean()
    # Test
    stderr2 = np.zeros(ffres.shape[0])
    for i in range(ffres.shape[0]):
        stderr2[i] = _newey_njit(data, int(ffres['lag'].iloc[i]))
    assert (np.abs(stderr2 - ffres['stderr'].to_numpy())<1e-5).all()

def test_estbeta1m_fmreg(ff, xsstk):
    # estbeta1m and Stata's rangestat are different. estbeta1m first convert
    # long-form panel data into wide form (rows are periods and columns are
    # stocks) filling missing returns as NaN. Then, estimate beta for each
    # column in rolling windows. As a result, for some stock and some period,
    # even if its return is missing, beta can still be calculated as long as
    # there are enough observations in the look-back period. However in such
    # case, rangestat will not estimate beta. As a result, when comparing the
    # results, we have to remove those cases where rangstat does not calculate.
    # The easiest way is to start from the panel data containing returns and
    # left join with beta estimates.
    # ---------- Test for beta estimation
    model = ['mktrf', 'smb', 'hml', 'umd']
    cols_test = model
    cols_targ = ['b_' + x for x in model]
    beta1 = estbeta1m(
        leftdata=xsstk['exret'],
        rightdata=ff,
        model=model,
        window=(-24, -1),
        minobs=6,
        hasconst=True
    )
    assert not isinstance(beta1.columns, pd.MultiIndex)
    beta2 = estbeta1m(
        leftdata=xsstk['exret'],
        rightdata=ff,
        model=model,
        window=(-20, -5),
        minobs=10,
        hasconst=True
    )
    assert not isinstance(beta2.columns, pd.MultiIndex)
    beta3 = estbeta1m(
        leftdata=xsstk['exret'],
        rightdata=ff,
        model=model,
        window=(-12, 0),
        minobs=13,
        hasconst=True
    )
    assert not isinstance(beta3.columns, pd.MultiIndex)
    betares1 = pd.read_stata('testbetares1.dta')
    betares1['datemn'] = betares1['datemn'].dt.to_period('M')
    betares1.set_index(['permno', 'datemn'], inplace=True)
    betares2 = pd.read_stata('testbetares2.dta')
    betares2['datemn'] = betares2['datemn'].dt.to_period('M')
    betares2.set_index(['permno', 'datemn'], inplace=True)
    betares3 = pd.read_stata('testbetares3.dta')
    betares3['datemn'] = betares3['datemn'].dt.to_period('M')
    betares3.set_index(['permno', 'datemn'], inplace=True)
    comp1 = xsstk[['exret']].join(beta1[cols_test]).join(betares1[cols_targ])
    comp2 = xsstk[['exret']].join(beta2[cols_test]).join(betares2[cols_targ])
    comp3 = xsstk[['exret']].join(beta3[cols_test]).join(betares3[cols_targ])
    # Check missing values
    assert comp1[cols_test+cols_targ].isna().sum().std() == 0
    assert comp2[cols_test+cols_targ].isna().sum().std() == 0
    assert comp3[cols_test+cols_targ].isna().sum().std() == 0
    # Drop NA
    comp1.dropna(subset=cols_test + cols_targ, inplace=True)
    comp2.dropna(subset=cols_test + cols_targ, inplace=True)
    comp3.dropna(subset=cols_test + cols_targ, inplace=True)
    # Check values
    assert (np.abs(
        comp1[cols_test].to_numpy() - comp1[cols_targ].to_numpy()
    ) < 1e-5).all()
    assert (np.abs(
        comp2[cols_test].to_numpy() - comp2[cols_targ].to_numpy()
    ) < 1e-5).all()
    assert (np.abs(
        comp3[cols_test].to_numpy() - comp3[cols_targ].to_numpy()
    ) < 1e-5).all()
    # ---------- Test for Fama-MacBeth regression
    models = [
        ['exret', 'mktrf'],
        ['exret', 'mktrf', 'smb', 'hml'],
        ['exret', 'mktrf', 'smb', 'hml', 'umd'],
    ]
    # Test result
    fmres1 = fmreg(
        leftdata=comp1[['exret']],
        rightdata=comp1[cols_test],
        models=models,
        maxlag=5,
        roworder=['mktrf', 'smb', 'hml', 'umd'],
        hasconst=True,
        scale=1,
        getlambda=False,
        estfmt=('.7f', '.7f'),
    )
    # Result from Stata's xtfmb
    fmres1c = pd.read_stata('testfmreg1.dta')
    fmres1c['model'] = fmres1c['model'].astype('int') - 1
    fmres1c['indepvar'] = fmres1c['indepvar'].replace({'_cons': 'Constant'})
    fmres1c['indepvar'] = fmres1c['indepvar'].str.replace('b_', '')
    fmres1c.set_index(['model', 'indepvar'], inplace=True)
    # Test
    for m in range(len(models)):
        indepvars = models[m][1:]
        for v in indepvars + ['Constant']:
            i = fmres1[''].to_list().index(v)
            print(f'{i}, {v}')
            # Test coefficient
            assert np.abs(
                fmres1c.loc[m].loc[v]['b'] -
                float(re.match('^([0-9\.\-]*)', fmres1[m].iloc[i]).group(1))
            ) < 1e-6
            # Test t-stat
            assert np.abs(
                fmres1c.loc[m].loc[v]['t'] -
                float(re.match(r'^\(([0-9\.\-]*)\)', fmres1[m].iloc[i+1]).group(1))
            ) < 1e-6
    # Test scale
    fmres1 = fmreg(
        leftdata=comp1[['exret']],
        rightdata=comp1[cols_test],
        models=models,
        maxlag=5,
        roworder=['mktrf', 'smb', 'hml', 'umd'],
        hasconst=True,
        scale=100,
        getlambda=False,
        estfmt=('.7f', '.7f'),
    )
    for m in range(len(models)):
        indepvars = models[m][1:]
        for v in indepvars + ['Constant']:
            i = fmres1[''].to_list().index(v)
            print(f'{i}, {v}')
            # Test coefficient
            assert np.abs(
                fmres1c.loc[m].loc[v]['b']*100 -
                float(re.match('^([0-9\.\-]*)', fmres1[m].iloc[i]).group(1))
            ) < 1e-6
            # Test t-stat
            assert np.abs(
                fmres1c.loc[m].loc[v]['t'] -
                float(re.match(r'^\(([0-9\.\-]*)\)', fmres1[m].iloc[i+1]).group(1))
            ) < 1e-6


def test_estbeta(ff, xsstk):
    # Index
    models = {
        'm1': ['exret', 'mktrf'],
        'm2': ['exret', 'mktrf', 'smb', 'hml'],
        'm3': ['exret', 'mktrf', 'smb', 'hml', 'umd']
    }
    beta1 = estbeta(
        leftdata=xsstk,
        rightdata=ff,
        models=models,
        window=(-20, -5),
        minobs=10,
        hasconst=True
    )
    assert beta1.columns.nlevels == 2
    beta1.columns = beta1.columns.to_flat_index()
    betares1 = pd.read_stata('testbetares4.dta')
    betares1['datemn'] = betares1['datemn'].dt.to_period('M')
    betares1.set_index(['permno', 'datemn'], inplace=True)
    l1 = beta1.columns.tolist()
    l2 = betares1.columns.tolist()
    c1 = xsstk[['exret']].join(beta1) \
        .join(betares1) \
        .dropna(subset=l1 + l2)
    assert (np.abs(c1[l1].to_numpy() - c1[l2].to_numpy()) < 1e-5).all()

    # Let stk and ff have a MultiIndex
    xsstk2 = xsstk.copy()
    xsstk2.columns = pd.MultiIndex.from_arrays([['exret'], ['exret']])
    ff2 = ff.copy()
    ff2.columns = pd.MultiIndex.from_product([['m1'], ff2.columns])
    ff2 = ff2.join(
        ff2.set_axis(ff2.columns.set_levels(['m2'], level=0), axis='columns'))
    models2 = {
        'm1': [('exret', 'exret'), ('m1', 'mktrf')],
        'm2': [('exret', 'exret'), ('m1', 'mktrf'), ('m1', 'smb'), ('m1', 'hml')],
        'm3': [('exret', 'exret'), ('m1', 'mktrf'), ('m1', 'smb'), ('m1', 'hml'),
               ('m1', 'umd')]
    }
    beta2 = estbeta(
        leftdata=xsstk2,
        rightdata=ff2,
        models=models2,
        window=(-20, -5),
        minobs=10,
        hasconst=True
    )
    l3 = ['_' + x for x in l2]
    c2 = xsstk[['exret']].join(beta2.set_axis(l3, axis='columns')) \
        .join(betares1) \
        .dropna(subset=l3 + l2)
    assert (np.abs(c2[l3].to_numpy() - c2[l2].to_numpy()) < 1e-5).all()



def test_format_table():
    df = pd.DataFrame({
        'intcol': [1000, 101, 2123432, 0],
        'floatcol': [0.1234934, 0.2342341234, 0.24238, 0.23427423],
    }, index=[f'Row {i}' for i in range(4)])
    sumstat = format_table(df, 4)
    df_corr = pd.DataFrame({
        'intcol': ['1,000', '101', '2,123,432', '0'],
        'floatcol': ['0.1235', '0.2342', '0.2424', '0.2343'],
    }, index=df.index)
    assert_frame_equal(sumstat, df_corr)


def test_groupby_wavg():
    N = 1000
    np.random.seed(1234)
    df = pd.DataFrame({
        'byvar1': np.random.randint(0, 5, (N,)),
        'byvar2': np.random.randint(0, 10, (N,)),
        'x': np.random.randn(N),
        'w': np.random.rand(N),
    }).sort_values(['byvar1', 'byvar2']).reset_index(drop=True)
    df.loc[np.random.randint(0, N, (int(.1*N),)), 'x'] = np.nan
    df.loc[np.random.randint(0, N, (int(.1*N),)), 'w'] = np.nan
    dftest = groupby_wavg(
        data=df, bys=['byvar1', 'byvar2'], var='x', weight='w'
    )
    dfcorr = df.dropna(subset=['x', 'w']).groupby(['byvar1', 'byvar2']).apply(
        lambda x: np.average(x['x'], weights=x['w'])
    )
    assert_series_equal(dftest, dfcorr, check_names=False)


def test_tscssum():
    N = 1000
    np.random.seed(1234)
    df = pd.DataFrame({
        'byvar1': np.random.randint(0, 5, (N,)),
        'byvar2': np.random.randint(0, 10, (N,)),
        'x': np.random.randn(N),
        'y': np.random.rand(N),
        'z': np.random.rand(N),
    }).sort_values(['byvar1', 'byvar2']).reset_index(drop=True)
    df.loc[np.random.randint(0, N, (int(.1*N),)), 'x'] = np.nan
    df.loc[np.random.randint(0, N, (int(.1*N),)), 'y'] = np.nan
    df.loc[np.random.randint(0, N, (int(.1*N),)), 'z'] = np.nan
    dftest = tscssum(
        df, by=['byvar1', 'byvar2'], subset=['x', 'z'], percentiles=(.01, .05, .25, .50, .75, .95, .99)
    )
    for v in ['x', 'z']:
        assert_float_equal(
            dftest.loc[v, 'N'],
            df.groupby(['byvar1','byvar2'])[v].count().sum()
        )
        assert_float_equal(
            dftest.loc[v, 'Mean'],
            df.groupby(['byvar1','byvar2'])[v].mean().mean()
        )
        assert_float_equal(
            dftest.loc[v, 'Std'],
            df.groupby(['byvar1','byvar2'])[v].std().mean()
        )
        assert_float_equal(
            dftest.loc[v, 'Min'],
            df.groupby(['byvar1','byvar2'])[v].min().mean()
        )
        assert_float_equal(
            dftest.loc[v, 'Max'],
            df.groupby(['byvar1','byvar2'])[v].max().mean()
        )
        for pct in [1, 5, 25, 50, 75, 95, 99]:
            assert_float_equal(
                dftest.loc[v, f'p{pct}'],
                df.groupby(['byvar1', 'byvar2'])[v].quantile(pct/100).mean(),
            )


# def test_get_port_ret():
#     N = 1000
#     np.random.seed(1234)
#     df = pd.DataFrame({
#         'period': np.random.randint(0, 5, (N,)),
#         'sortvar1': np.random.rand(N),
#         'sortvar2': np.random.rand(N),
#         'x': np.random.randn(N),
#         'w': np.random.rand(N),
#     }).sort_values(['sortvar1', 'sortvar2']).reset_index(drop=True)
#     df.loc[np.random.randint(0, N, (int(.1*N),)), 'sortvar1'] = np.nan
#     df.loc[np.random.randint(0, N, (int(.1*N),)), 'sortvar2'] = np.nan
#     df.loc[np.random.randint(0, N, (int(.1*N),)), 'x'] = np.nan
#     df.loc[np.random.randint(0, N, (int(.1*N),)), 'w'] = np.nan
#     dftest = get_port_ret(
#         data=df, nq=5, timevar='period', retvar='x',
#         rnkvars=['sortvar1', 'sortvar2'], rnkvarnames=['rnk1', 'rnk2'],
#         wvar='w', dep=True,
#     )
#



# if __name__ == '__main__':
#     arr = np.arange(0, 101)
#     _winsor_njit(arr, (0.05, 0.95), interpolation='inner')