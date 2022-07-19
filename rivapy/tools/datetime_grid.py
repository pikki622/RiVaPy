import datetime as dt
import abc 
import pandas as pd
import numpy as np


class DateTimeGrid:
    def __init__(self, start:dt.datetime, end:dt.datetime, freq: str='1H', daycounter=None, tz=None):
        self.start = start
        self.end = end
        self.freq = freq
        self.tz = tz

        if self.start is not None:
            self.dates = pd.date_range(self.start, self.end, freq=self.freq, tz=self.tz, closed='left').to_pydatetime()
            self.timegrid = np.array([(d-self.start).total_seconds()/pd.Timedelta('1Y').total_seconds() for d in self.dates])
            self.shape = self.timegrid.shape
        else:
            self.dates = None
            self.timegrid = None
            self.shape = None
        

    def get_daily_subgrid(self):
        df = pd.DataFrame({'dates': self.dates, 'tg': self.timegrid})
        df['dates_'] =df['dates'].dt.date
        df = df.groupby(by=['dates_']).min()
        df = df.reset_index()
        result = DateTimeGrid(None, None, freq='1D')
        result.dates=np.array([d.to_pydatetime() for d in df['dates']])
        result.timegrid = df['tg'].values
        result.shape = result.timegrid.shape
        return result

    def get_grid_indices(self, dates):
        df = pd.DataFrame({'dates': self.dates, 'tg': self.timegrid})
        df = df.reset_index()
        df = df.set_index('dates')
        df_tg = pd.DataFrame({'dates_': dates})
        df.join(df_tg)

class __TimeGridFunction(abc.ABC):
    @abc.abstractmethod
    def _compute(self, d: dt.datetime)->float:
        pass

    def compute(self, tg: DateTimeGrid, x=None)->np.ndarray:
        if x is None:
            x = np.empty(tg.shape)
        for i in range(tg.shape[0]):
            x[i] = self._compute(tg.dates[i])
        return x
    
class _Add(__TimeGridFunction):
    def __init__(self, f1,f2):
        self._f1 = f1
        self._f2 = f2
    
    def _compute(self, d: dt.datetime)->float:
        return self._f1._compute(d)+self._f2._compute(d)

class _Mul(__TimeGridFunction):
    def __init__(self, f1,f2):
        self._f1 = f1
        self._f2 = f2
    
    def _compute(self, d: dt.datetime)->float:
        return self._f1._compute(d)*self._f2._compute(d)

class _TimeGridFunction(__TimeGridFunction):
    """Abstract base class for all functions that are defined on datetimes

    Args:
        _TimeGridFunction (_type_): _description_

    Returns:
        _type_: _description_
    """
    @abc.abstractmethod
    def _compute(self, d: dt.datetime)->float:
        pass
    
    def __add__(self, other):
        return _Add(self, other)

    def __mul__(self, other):
        return _Mul(self, other)

class MonthlyConstantFunction(_TimeGridFunction):
    def __init__(self, values:list):
        """Function that is constant across a month.

        Args:
            values (list): values[i] contains the value for the (i+1)-th month
        """
        self.values = values

    def _compute(self, d: dt.datetime)->float:
        return self.values[d.month-1]

class HourlyConstantFunction(_TimeGridFunction):
    def __init__(self, values:list):
        """Function that is constant on hours.

        Args:
            values (list): values[i] contains the value for the i-th hour
        """
        self.values = values

    def _compute(self, d: dt.datetime)->float:
        return self.values[d.hour]
