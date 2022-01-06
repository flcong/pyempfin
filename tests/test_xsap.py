import pytest
import numpy as np
from pyempfin.xsap import _winsor_njit, _newey_njit, estbeta, estbeta1m
import pandas as pd

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
    w1 = _winsor_njit(arr, (0.052, 0.959), interpolation='inner')
    assert checkarr(w1, arr, lower=6.0, upper=95.0)
    w2 = _winsor_njit(arr, (0.052, 0.959), interpolation='outer')
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

def test_estbeta1m(ff, xsstk):
    model = ['mktrf', 'smb', 'hml', 'umd']
    l1 = model
    l2 = ['b_' + x for x in model]
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
    c1 = xsstk[['exret']].join(beta1) \
        .join(betares1) \
        .dropna(subset=l1 + l2)
    c2 = xsstk[['exret']].join(beta2) \
        .join(betares2) \
        .dropna(subset=l1 + l2)
    # Check missing values
    assert c1.isna().any(axis=1).sum() == 0
    assert c2.isna().any(axis=1).sum() == 0
    assert beta3[l1].shape == betares3[l2].shape
    # Check values
    assert (np.abs(c1[l1].to_numpy() - c1[l2].to_numpy()) < 1e-5).all()
    assert (np.abs(c2[l1].to_numpy() - c2[l2].to_numpy()) < 1e-5).all()
    assert (np.abs(beta3[l1].to_numpy() - betares3[l2].to_numpy()) < 1e-5).all()

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



# def test_fmreg(ff, xsstk):
#     # Estimate a series of models
#     models = [
#         ['exret', 'mktrf'],
#         ['exret', 'mktrf', 'smb', 'hml'],
#         ['exret', 'mktrf', 'smb', 'hml', 'umd']
#     ]

# if __name__ == '__main__':
#     arr = np.arange(0, 101)
#     _winsor_njit(arr, (0.05, 0.95), interpolation='inner')