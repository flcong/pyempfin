from pyempfin.datetimeutils import *
import pandas as pd

def test_yrdif():
    df1 = pd.DataFrame({
        'start': ['2010-08-27', '2010-08-27'],
        'end': ['2022-06-01', '2024-06-01']
    })
    df1['start'] = pd.to_datetime(df1['start'])
    df1['end'] = pd.to_datetime(df1['end'])
    df1['ttm'] = yrdif(
        df1['start'].to_numpy(),
        df1['end'].to_numpy()
    )

    df1