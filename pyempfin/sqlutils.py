import sqlite3
import pandas as pd
import numpy as np
from typing import Union

# SQL query
def dfsql(
        query: str,
        parse_dates: Union[list,None]=None,
        create_functions: Union[list,None]=None,
        create_aggregates: Union[list,None]=None,
        **kwargs
        ) -> pd.DataFrame:
    """Execute a SQL query

    Parameters
    ----------
    query : str
        The SQL query to be executed
    parse_dates : list of str or None
        The list of variables of datetime type
    create_functions : list or None
        Functions to be used in the SQL query
    create_aggregates : list or None
        Aggregate functions (classes) to be used in the SQL query
    **kwargs
        The key is the table name in the SQL query and the value is the
        corresponding pandas data frame.

    Returns
    -------
    pandas.DataFrame
        The output table from the SQL query
    """
    # Connect database
    conn = sqlite3.connect(':memory:')
    # Add tables to database
    for v in kwargs:
        kwargs[v].to_sql(v, conn, index=False)
    # Add functions to database
    if create_functions is not None:
        for f in create_functions:
            conn.create_function(*f)
    # Add aggregate to database
    if create_aggregates is not None:
        for f in create_aggregates:
            conn.create_aggregate(*f)
    # Enable tracing errors
    sqlite3.enable_callback_tracebacks(True)
    # Execute query
    out = pd.read_sql_query(query, conn, parse_dates=parse_dates)
    # Close database
    conn.close()
    # Convert datetime
    return out

class StdevFunc:
    """Standard deviation function to be used as an aggregate function in SQLite
    """

    def __init__(self):
        self.M = 0.0
        self.S = 0.0
        self.k = 0

    def step(self, value):
        try:
            # automatically convert text to float, like the rest of SQLite
            # if fails, skips this iteration, which also ignores nulls
            val = float(value)
            tM = self.M
            self.k += 1
            self.M += ((val - tM) / self.k)
            self.S += ((val - tM) * (val - self.M))
        except:
            pass

    def finalize(self):
        if self.k <= 1:  # avoid division by zero
            return np.nan
        else:
            return np.sqrt(self.S / (self.k-1))


class WeightedAverageFunc:
    """Weighted average function to be used as an aggregate function in SQLite
    """

    def __init__(self):
        self.numer = 0.0
        self.denom = 0.0

    def step(self, value, weight):
        try:
            # automatically convert text to float, like the rest of SQLite
            # if fails, skips this iteration, which also ignores nulls
            val = float(value)
            w = float(weight)
            self.denom += w
            self.numer += val * w
        except:
            pass

    def finalize(self):
        if self.denom == 0:  # avoid division by zero
            return np.nan
        else:
            return self.numer / self.denom
