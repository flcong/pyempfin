from tabulate import tabulate
import pandas as pd
import os
from datetime import datetime
from typing import Union

def dictout(dict: dict) -> str:
    """Print a dictionary in text format
    """
    dicttmp = dict.copy()
    for k in dicttmp:
        if not isinstance(dicttmp[k], list):
            dicttmp[k] = [dicttmp[k]]
    return tabulate(dicttmp, tablefmt='psql', headers='keys')

def tabout(data: pd.DataFrame, type: str='reg', showindex: bool=False,
           floatfmt: Union[str,None]=None, suffix: str='\n') -> str:
    """Print a table in text format
    """
    if showindex:
        if data.index.name is None:
            data.index.name = ''
        data = data.reset_index(drop=False)
    # Supplement other floatfmt
    if isinstance(floatfmt, tuple) \
            and len(floatfmt) < data.shape[1]+showindex:
        floatfmt += (floatfmt[-1],) * (data.shape[1]+showindex-len(floatfmt))
    elif floatfmt is None:
        # Default is to use '.0f' for integer and '.3f' for float
        floatfmt = ()
        for v in data:
            if data[v].dtype.kind in 'biu':
                floatfmt += ('.0f',)
            elif data[v].dtype.kind in 'f':
                floatfmt += ('.3f',)
            else:
                floatfmt += ('',)
    if type == 'reg':
        data.columns = data.columns.str.replace(r'level_\d*', '', regex=True)
        return tabulate(
            data, 
            headers=data.columns,
            showindex=False,
            tablefmt='psql',
            floatfmt=floatfmt,
            colalign=('left',) + ('center',) * (data.shape[1]-1),
            ) + suffix
    else:
        raise ValueError('Invalid type: ' + type)

class logger:
    def __init__(self, file: str, overwrite: str='replace'):
        """Create a logger

        Parameters
        ----------
        file : str
            Name (and path) of the log file
        overwrite : str
            Possible values are `'replace'' and `'append''
        """
        # Open file and get the handler
        if overwrite == 'replace':
            self.filehandle = open(file, 'w')
        elif overwrite == 'append':
            self.filehandle = open(file, 'a')
        # Print file header
        self.logheader()
        
    def logheader(self):
        self.log('='*100)
        self.log(' File path: ' + self.filehandle.name)
        self.log('Created at: ' + datetime.now().strftime(r'%Y-%m-%d %H:%M:%S'))
        self.log('='*100)
        return None

    def log(self, text: str, suffix: str='\n'):
        """Print text to the log file, a newline is automatically added

        Parameters
        ----------
        text : str
        suffix : str
        """
        self.filehandle.write(text + suffix)
        self.filehandle.flush()
        os.fsync(self.filehandle)

    def close(self):
        """Close log file
        """
        self.filehandle.close()
    
    def __del__(self):
        self.filehandle.close()