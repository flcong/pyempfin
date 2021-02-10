import pandas as pd


# Code ratings
mrtable = {
    'Aaa': 1,
    'Aa1': 2,
    'Aa2': 3,
    'Aa': 3,
    'Aa3': 4,
    'A1': 5,
    'A2': 6,
    'A': 6,
    'A3': 7,
    'Baa1': 8,
    'Baa2': 9,
    'Baa': 9,
    'Baa3': 10,
    'Ba1': 11,
    'Ba2': 12,
    'Ba': 12,
    'Ba3': 13,
    'B1': 14,
    'B2': 15,
    'B': 15,
    'B3': 16,
    'Caa1': 17,
    'Caa2': 18,
    'Caa': 18,
    'Caa3': 19,
    'Ca': 20,
    'C': 21,
    'D': 22
}
sprtable = {
    'AAA': 1,
    'AA+': 2,
    'AA': 3,
    'AA-': 4,
    'A+': 5,
    'A': 6,
    'A-': 7,
    'BBB+': 8,
    'BBB': 9,
    'BBB-': 10,
    'BB+': 11,
    'BB': 12,
    'BB-': 13,
    'B+': 14,
    'B': 15,
    'B-': 16,
    'CCC+': 17,
    'CCC': 18,
    'CCC-': 19,
    'CC': 20,
    'C': 21,
    'D': 22
}

def coderating(series: pd.Series, type: str) -> pd.Series:
    """Convert str credit rating to integer

    Parameters
    ----------
    series : pandas.Series
    type : str
        'mr' or 'spr'

    Returns
    -------
    pandas.Series
    """
    if type == 'mr':
        return series.apply(lambda x: mrtable.get(x, pd.NA)).astype('Int8')
    elif type == 'spr':
        return series.apply(lambda x: sprtable.get(x, pd.NA)).astype('Int8')
    else:
        raise ValueError('Invalid credit rating type!')
