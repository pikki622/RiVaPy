
import abc
import datetime as dt
import numpy as np
#from rivapy.tools.datetime_grid import DateTimeGrid

# class DateTimeFunction(abc.ABC):
#     @abc.abstractmethod
#     def compute(self, ref_date: dt.datetime, dt_grid: DateTimeGrid)->np.ndarray:
#         pass

class BaseDatedCurve(abc.ABC):
    @abc.abstractmethod
    def compute(self, ref_date: dt.datetime)->np.ndarray:#, dt_grid: DateTimeGrid)->np.ndarray:
        pass