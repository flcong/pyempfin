import pandas as pd
from typing import Union

def rename(data: pd.DataFrame, inplace: bool=False, **kwargs) -> Union[pd.DataFrame,None]:
    """Rename columns and return the new DataFrame

    Parameters
    ----------
    data : pandas.DataFrame
    **kwargs
        Format: `oldname=newname`.

    Returns
    -------
    pandas.DataFrame
    """

    if inplace:
        data.rename(columns=kwargs, inplace=True)
        return None
    else:
        return data.rename(columns=kwargs)

def dropvar(data: pd.DataFrame, subset: Union[list,str], inplace: bool=False
            ) -> Union[pd.DataFrame,None]:
    """Drop a list of variables and return the new dataframe

    Parameters
    ----------
    data : pandas.DataFrame
    subset : list or str
    inplace : bool, default: False

    Returns
    -------
    pandas.DataFrame
    """
    if isinstance(subset, str):
        subset = subset.replace('  ', ' ').split(' ')
        assert set(subset).issubset(set(data.columns)), 'Invalid subset'
    if inplace:
        data.drop(columns=subset, inplace=True)
        return None
    else:
        return data.drop(columns=subset)


def keepna(data: pd.DataFrame, subset: Union[list,None]=None, how='any') -> pd.DataFrame:
    """Keep rows where specific variables are missing

    Parameters
    ----------
    data : pandas.DataFrame
    subset : list
    how : str, default: 'all'
        The value can be 'all' or 'any'

    Returns
    -------
    pandas.DataFrame
    """
    if subset is None:
        subset = data.columns
    else:
        assert set(subset).issubset(set(data.columns))
    if how == 'any':
        return data.loc[lambda x: x[subset].isna().any(axis=1)]
    elif how == 'all':
        return data.loc[lambda x: x[subset].isna().all(axis=1)]
    else:
        raise ValueError('Invalid parameter how: ' + how)
