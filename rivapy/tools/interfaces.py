
import abc
import datetime as dt
import numpy as np
from rivapy.tools.datetime_grid import DateTimeGrid

class DateTimeFunction(abc.ABC):
    @abc.abstractmethod
    def compute(self, ref_date: dt.datetime, dt_grid: DateTimeGrid)->np.ndarray:
        pass

class FactoryObject(abc.ABC):

    def to_dict(self):
        result = self._to_dict()
        result['cls'] = type(self).__name__
        return result

    @abc.abstractmethod
    def _to_dict(self)->dict:
        pass

    @classmethod
    def from_dict(cls, data: dict)->object:
        #data_ = {k:v for k,v in data}
        return cls(**{k:v for k,v in data.items() if k != 'cls'})

class BaseDatedCurve(abc.ABC):
    @abc.abstractmethod
    def compute(self, ref_date: dt.datetime)->np.ndarray:#, dt_grid: DateTimeGrid)->np.ndarray:
        pass