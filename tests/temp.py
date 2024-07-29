import pytest
import numpy as np
from pyempfin.xsap import winsor_njit, _newey_njit, estbeta, estbeta1m
import pandas as pd
import os
from joblib import Parallel, delayed

os.chdir('test')
# Load stock data
xsstk = pd.read_csv('xsstktestdata.csv')
xsstk.columns = xsstk.columns.str.lower()
xsstk['datemn'] = pd.to_datetime(xsstk['date'].astype(str)).dt.to_period('M')
xsstk = xsstk.set_index(['permno', 'datemn']).sort_index()
xsstk = xsstk[['exret']]
# Load factor data
ff = pd.read_csv('ff.csv')
ff['datemn'] = pd.to_datetime(ff['date'].astype(str)).dt.to_period('M')
ff.drop(columns=['date'], inplace=True)
ff.set_index('datemn', inplace=True)
# Index
models = {
    'm1': ['exret', 'mktrf'],
    'm2': ['exret', 'mktrf', 'smb', 'hml'],
    'm3': ['exret', 'mktrf', 'smb', 'hml', 'umd']
}
beta = estbeta(
    leftdata=xsstk,
    rightdata=ff,
    models=models,
    window=(-20, -5),
    minobs=10,
    hasconst=True
)

data = beta
subset = None
percentiles = (.01, .05, .50, .95, .99)


# Let stk and ff have a MultiIndex
xsstk2 = xsstk.copy()
xsstk2.columns = pd.MultiIndex.from_arrays([['exret'], ['exret']])
ff2 = ff.copy()
ff2.columns = pd.MultiIndex.from_product([['m1'], ff2.columns])
ff2 = ff2.join(ff2.set_axis(ff2.columns.set_levels(['m2'], level=0), axis='columns'))
models2 = {
    # 'model1': [('exret', 'exret'), ('m1', 'mktrf')],
    'model2': [('exret', 'exret'), ('m1', 'mktrf'), ('m1', 'smb'), ('m2', 'hml')],
    'model3': [('exret', 'exret'), ('m1', 'mktrf'), ('m1', 'smb'), ('m2', 'hml'),
           ('m2', 'umd')]
}
beta2 = estbeta(
    leftdata=xsstk2,
    rightdata=ff2,
    models=models2,
    window=(-20, -5),
    minobs=10,
    hasconst=True
)

# leftdata = xsstk2
# rightdata = ff2
# models = models2
# window = (-20, -5)
# minobs = 10
# hasconst = True
#
# x = list(models.values())[0]
# res = estbeta1m(
#     leftdata=leftdata[x[0]],
#     rightdata=rightdata[x[1:]],
#     model=x[1:],
#     window=window,
#     minobs=minobs,
#     hasconst=hasconst
# )
# #
# model = x
# res = pd.DataFrame(
#         np.concatenate(
#             Parallel(n_jobs=6)(
#                 delayed(
#                     lambda x: _get_beta_njit(
#                         x, exog, window, minobs, hasconst)
#                 )(leftmat[v].to_numpy())
#                 for v in leftmat.columns), axis=1
#         ),
#         index=leftmat.index,
#         columns=pd.MultiIndex.from_product([leftmat.columns, model])
#     )
# res.stack(-2)
# res.columns.levels[0]