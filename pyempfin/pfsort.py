# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:43:52 2020

@author: flcon
"""

from statsmodels.regression.linear_model import OLS, OLSResults
from statsmodels.stats.sandwich_covariance import cov_hac
import scipy.stats
import pandas as pd 
import numpy as np 
from numba import njit, int32, float64
import warnings
# cut portfolios into groups
@njit
def qcut_jit(x, q):
    return np.floor(np.searchsorted(np.sort(x), x, side='right') * q / len(x) - 1e-12) + 1

# Univariate portfolio sorting
def uniportsort(modelsetup):
    """ 
    Input:
        rdata: Panel data containing timevar, indvar, and retvar, with timevar and indvar the MultiIndex for axis=0.
        sdata: Panel data containing timevar, indvar, and variables to be sorted. Timevar and indvar are the indexes for axis=0. [REMOVED] On axis=1, there can be MultiIndex (modelname is used to select the high-level index) or index (modelname=None) [/REMOVED]. The user should ensure that sdata contains the right model
        timevar: Name of the time variable
        [REMOVED] modelname: Name of the specific model for estimating beta. If there is only one model, set it to None.
        sortvar: Name of the sort variable
        retvar: Name of the return variable
        ngroups: Number of groups
        weight: Weight to calculate return. None for equal-weighting.
        shortlong: False for High-Low portfolio and True for Low-High portfolio 
        maxlag: maximum lag for Newey-West corretion
        fmodel: a dict of keys (model names) and values (list of factor variables) to specify factor models to correct alpha
        fdata: a time-series data frame containing factor data
        pct: whether the returns are in percentage points or not
    """
    rdata = modelsetup['rdata']
    sdata = modelsetup['sdata']
    timevar = modelsetup['timevar']
    # modelname = modelsetup['modelname']
    sortvar = modelsetup['sortvar']
    retvar = modelsetup['retvar']
    ngroups = modelsetup['ngroups']
    weight = modelsetup['weight']
    shortlong = modelsetup['shortlong']
    maxlag = modelsetup['maxlag']
    fmodel = modelsetup['fmodel']
    fdata = modelsetup['fdata']
    pct = modelsetup['pct']
    # Long-short or short-long
    if shortlong:
        lsvar = f'P1-P{ngroups}'
    else:
        lsvar = f'P{ngroups}-P1'

    # Merge sort variable with returns
    if weight is None:
        dfr = pd.concat(
            [rdata[retvar], sdata[sortvar].rename(sortvar)], axis=1
        ).dropna(axis=0, subset=[sortvar])
    else:
        dfr = pd.concat(
            [rdata[[retvar, weight]], sdata[sortvar].rename(sortvar)], axis=1
        ).dropna(axis=0, subset=[sortvar])
    # The user should ensure that the correct model is sent to the function
    # if weight is None:
    #     if modelname is None:
    #         dfr = pd.concat(
    #             [rdata[retvar], sdata[sortvar].rename(sortvar)],
    #             axis=1
    #         ).dropna(axis=0, subset=[sortvar])
    #     else:
    #         dfr = pd.concat(
    #             [rdata[retvar], sdata[(modelname, sortvar)].rename(sortvar)],
    #             axis=1
    #         ).dropna(axis=0, subset=[sortvar])
    # else:
    #     if modelname is None:
    #         dfr = pd.concat(
    #             [rdata[[retvar,weight]], sdata[sortvar].rename(sortvar)],
    #             axis=1
    #         ).dropna(axis=0, subset=[sortvar])
    #     else:
    #         dfr = pd.concat(
    #             [rdata[[retvar,weight]], sdata[(modelname, sortvar)].rename(sortvar)], axis=1
    #         ).dropna(axis=0, subset=[sortvar])
    # Get ranks
    dfr['rnk'] = dfr.groupby(timevar)[sortvar].transform(lambda x: qcut_jit(x.values, q=ngroups)).astype('int')
    # Calculate portfolio return
    if weight is None:
        dfp = dfr.reset_index().groupby([timevar, 'rnk']).apply(
            lambda x: np.mean(x[retvar])
        ).to_frame(name=retvar)
    else:
        dfp = dfr.reset_index().groupby([timevar, 'rnk']).apply(
            lambda x: np.average(x[retvar], weights=x[weight])
        ).to_frame(name=retvar)
    # Calculate average sorted variable
    dfs = dfr.reset_index().groupby([timevar, 'rnk']).apply(
        lambda x: np.mean(x[sortvar])
    ).to_frame(name=sortvar)
    del dfr
    # Transpose
    dfp = dfp.reset_index().pivot(index=timevar, columns='rnk', values=retvar).rename(
        columns=lambda x: 'P' + str(x)
    )
    dfs = dfs.reset_index().pivot(index=timevar, columns='rnk', values=sortvar).rename(
        columns=lambda x: 'P' + str(x)
    )
    # Average sort variable
    res_sortvar = dfs.mean().apply(lambda x: "{:5.4f}".format(x)).rename('fconst').to_frame().assign(ftstat="").stack().rename(r'$\beta_{\mathrm{'+sortvar+r'}}$')
    # Long-short portfolio return
    if shortlong:
        dfp[lsvar] = dfp['P1'] - dfp[f'P{ngroups}']
    else: 
        dfp[lsvar] = dfp[f'P{ngroups}'] - dfp['P1']
    # Calculate average raw return and t-stat for each portfolio
    dfavgp = pd.concat([_get_ave_ret(x[1], maxlag) for x in dfp.items()], axis=1, keys=dfp.columns)
    dfavgp = dfavgp.T
    # Format average return and t-stat 
    dfavgp['fconst'] = dfavgp[['const','t','pval']].apply(lambda x: _format_const(x, 'const', 'pval', pct), axis=1)
    dfavgp['ftstat'] = dfavgp[['t']].apply(lambda x: '({:4.2f})'.format(x['t']), axis=1)
    # Generate output column for average return
    res_avgret = dfavgp[['fconst','ftstat']].stack().rename('Average return')
    # Calculate alpha or not 
    if fmodel is None or fdata is None:
        res_out = pd.concat([res_sortvar, res_avgret], axis=1).fillna(value="").reindex(res_avgret.index).reset_index().drop(columns='level_1')
    else:
        # Calculate and format alpha
        res_alpha = pd.concat([_get_formated_alpha(dfp, fdata.reindex(dfp.index), fvars, maxlag, pct) for fname, fvars in fmodel.items()], axis=1, keys=fmodel.keys())
        res_out = pd.concat([res_sortvar, res_avgret, res_alpha], axis=1).fillna(value="").reindex(res_avgret.index).reset_index().drop(columns='level_1')
    # Output
    res_out['rnk'] = res_out['rnk'].where(np.arange((ngroups+1)*2) % 2 == 0, other="")
    res_out.rename(columns={'rnk':'Portfolio'}, inplace=True)
    return res_out

def biportsortall(modelsetup, cdata, ctrlvarlst, fvars):
    res_out = pd.concat(
        map(lambda x: biportsort1(modelsetup, cdata, x, fvars), ctrlvarlst), axis=1, keys=ctrlvarlst
    ).reset_index().drop(columns='level_1')
    res_out['rnk_sort'] = res_out['rnk_sort'].where(np.arange((modelsetup['ngroups']+1)*2) % 2 == 0, other="")
    res_out.rename(columns={'rnk_sort':'Portfolio'}, inplace=True)
    return res_out

def biportsort1(modelsetup, cdata, ctrlvar, fvars):
    rdata = modelsetup['rdata']
    sdata = modelsetup['sdata']
    timevar = modelsetup['timevar']
    # modelname = modelsetup['modelname']
    sortvar = modelsetup['sortvar']
    retvar = modelsetup['retvar']
    ngroups = modelsetup['ngroups']
    weight = modelsetup['weight']
    shortlong = modelsetup['shortlong']
    maxlag = modelsetup['maxlag']
    fdata = modelsetup['fdata']
    pct = modelsetup['pct']
    # Long-short or short-long
    if shortlong:
        lsvar = f'P1-P{ngroups}'
    else:
        lsvar = f'P{ngroups}-P1'
    # Merge sort variable with returns
    if weight is None:
        dfr = pd.concat(
            [rdata[retvar], cdata[ctrlvar], sdata[sortvar].rename(sortvar)], axis=1
        ).dropna(axis=0, subset=[ctrlvar, sortvar])
    else:
        dfr = pd.concat(
            [rdata[[retvar, weight]], cdata[ctrlvar], sdata[sortvar].rename(sortvar)], axis=1
        ).dropna(axis=0, subset=[ctrlvar, sortvar])
    # if weight is None:
    #     if modelname is None:
    #         dfr = pd.concat(
    #             [rdata[retvar], cdata[ctrlvar], sdata[sortvar].rename(sortvar)],
    #             axis=1
    #         ).dropna(axis=0, subset=[ctrlvar, sortvar])
    #     else:
    #         dfr = pd.concat(
    #             [rdata[retvar], cdata[ctrlvar], sdata[(modelname, sortvar)].rename(sortvar)],
    #             axis=1
    #         ).dropna(axis=0, subset=[ctrlvar, sortvar])
    # else:
    #     if modelname is None:
    #         dfr = pd.concat(
    #             [rdata[[retvar,weight]], cdata[ctrlvar], sdata[sortvar].rename(sortvar)],
    #             axis=1
    #         ).dropna(axis=0, subset=[ctrlvar, sortvar])
    #     else:
    #         dfr = pd.concat(
    #             [rdata[[retvar,weight]], cdata[ctrlvar], sdata[(modelname, sortvar)].rename(sortvar)], axis=1
    #         ).dropna(axis=0, subset=[ctrlvar, sortvar])
    # Get ctrl variable rank
    dfr['rnk_ctrl'] = dfr.groupby(timevar)[ctrlvar].transform(
        lambda x: qcut_jit(x.values, q=ngroups)
    ).astype('int')
    # Get sort variable rank controlling for ctrl variable
    # dfr['rnk_sort'] = dfr.groupby([timevar, 'rnk_ctrl'])[sortvar].transform(
    #     lambda x: pd.qcut(x, q=ngroups, labels=np.arange(1, ngroups+1))
    # ).astype('int')
    dfr['rnk_sort'] = dfr.groupby([timevar, 'rnk_ctrl'])[sortvar].transform(
        lambda x: qcut_jit(x.values, q=ngroups)
    ).astype('int')
    # Calculate portfolio return
    if weight is None:
        dfp = dfr.reset_index().groupby([timevar, 'rnk_ctrl', 'rnk_sort']).agg({retvar: 'mean'})
    else:
        dfr['__rw__'] = dfr[retvar].values * dfr[weight].values
        dfr = dfr.reset_index().groupby([timevar, 'rnk_ctrl', 'rnk_sort']).agg({weight: 'sum', '__rw__': 'sum'})
        dfp = pd.DataFrame(dfr['__rw__'].values / dfr[weight].values, index=dfr.index, columns=[retvar])    
    del dfr 
    #-------- Output 1: Average across ctrl ranks
    # Calculate average portfolio return across all ctrl ranks
    dfp = dfp.groupby(['date_mn', 'rnk_sort'])[[retvar]].agg('mean')
    # Transpose
    dfp = dfp.reset_index().pivot(index=timevar, columns='rnk_sort', values=retvar).rename(
        columns=lambda x: 'P' + str(x)
    )
    # Long-short portfolio return
    if shortlong:
        dfp[lsvar] = dfp['P1'] - dfp[f'P{ngroups}']
    else: 
        dfp[lsvar] = dfp[f'P{ngroups}'] - dfp['P1']
    # Calculate average return or alpha
    if fvars is None:
        # Calculate average raw return and t-stat for each portfolio
        dfavgp = pd.concat([_get_ave_ret(x[1], maxlag) for x in dfp.items()], axis=1, keys=dfp.columns)
        dfavgp = dfavgp.T
        # Format average return and t-stat 
        dfavgp['fconst'] = dfavgp[['const','t','pval']].apply(lambda x: _format_const(x, 'const', 'pval', pct), axis=1)
        dfavgp['ftstat'] = dfavgp[['t']].apply(lambda x: '({:4.2f})'.format(x['t']), axis=1)
        # Generate output column for average return
        res_out = dfavgp[['fconst','ftstat']].stack().rename(ctrlvar)
    else:
        res_out = _get_formated_alpha(dfp, fdata.reindex(dfp.index), fvars, maxlag).rename(ctrlvar)
    return res_out

# Estimate factor alpha given factor data (fdata), factor variables (fvars)
def _get_alpha(x, fdata, fvars, maxlag):
    exog = fdata[fvars].dropna(how='any').assign(const=1)
    endog = x.reindex(exog.index)
    mod = OLS(endog, exog).fit()
    res = pd.Series([mod.params['const'],
                     np.sqrt(cov_hac(mod, nlags=maxlag)[-1,-1]), 
                     mod.df_resid], index=['const','se','df'])
    res['t'] = res['const'] / res['se']
    res['pval'] = 2 * (1 - scipy.stats.t.cdf(abs(res['t']), res['df']))
    return res

# Calculate alpha and format columns for each factor model (prdata: portfolio return data)
def _get_formated_alpha(prdata, fdata, fvars, maxlag, pct):
    dfalpha = pd.concat([_get_alpha(x[1], fdata, fvars, maxlag) for x in prdata.items()], axis=1, keys=prdata.columns)
    dfalpha = dfalpha.T
    # Format alpha
    dfalpha['fconst'] = dfalpha[['const','t','pval']].apply(lambda x: _format_const(x, 'const', 'pval', pct), axis=1)
    dfalpha['ftstat'] = dfalpha[['t']].apply(lambda x: '({:4.2f})'.format(x['t']), axis=1)
    return dfalpha[['fconst','ftstat']].stack()


# Format point estimates with stars based on pval
def _format_const(estseries, paramvar, pvalvar, scale, format='9.2f'):
    """Format point estimates with stars based on p-value.

    Parameters
    ----------
    estseries : pandas.Series
        A series of parameter estimates and p-values, such that
        estseries[parameter] is the point estimate and estseries[pvalvar] is
        the corresponding p-value.
    paramvar : str
    pvalvar : str
    scale : int
        Multiplier to apply to the point estimate.
    format : str, default: '9.2f'
        Format of the point estimate

    Returns
    -------
    str
    """
    if estseries[pvalvar] > 0.1:
        return ('{:'+format+'}').format(estseries[paramvar] * scale)
    elif (0.05 < estseries[pvalvar]) and (estseries[pvalvar] <= 0.1):
        return ('{:'+format+'}*').format(estseries[paramvar] * scale)
    elif (0.01 < estseries[pvalvar]) and (estseries[pvalvar] <= 0.05):
        return ('{:'+format+'}**').format(estseries[paramvar] * scale)
    elif (0 <= estseries[pvalvar]) and (estseries[pvalvar] <= 0.01):
        return ('{:'+format+'}***').format(estseries[paramvar] * scale)


# Estimate average return and t-stat for each column (portfolio) with Newey-West standard error
def _get_ave_ret(x, maxlag):
    exog = np.ones(x.shape)
    mod = OLS(x, exog).fit()
    res = pd.Series(
        [mod.params.values[0], np.sqrt(cov_hac(mod, nlags=maxlag))[0][0], mod.df_resid],
        index=['const', 'se', 'df']
        )
    res['t'] = res['const'] / res['se']
    res['pval'] = 2 * (1 - tdist.cdf(abs(res['t']), res['df']))
    return res