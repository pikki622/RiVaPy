
import abc
from typing import List, Tuple
import datetime as dt
import numpy as np
import json
import hashlib
from rivapy.tools.datetime_grid import DateTimeGrid

class DateTimeFunction(abc.ABC):
    @abc.abstractmethod
    def compute(self, ref_date: dt.datetime, dt_grid: DateTimeGrid)->np.ndarray:
        pass

class _JSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        return {
            key: dt.fromisoformat(value)
            if key in {'timestamp', 'whatever'}
            else value
            for key, value in obj.items()
        }

class _JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (dt.date, dt.datetime)):#, pd.Timestamp)):
            return obj.isoformat()
        return json.JSONEncoder.default(obj)
        
class FactoryObject(abc.ABC):

    def to_dict(self):
        result = self._to_dict()
        result['cls'] = type(self).__name__
        return result

    def to_json(self):
        return json.dumps(self.to_dict(), cls=_JSONEncoder).encode()

    @classmethod
    def from_json(cls, json_str: str):
        tmp = json.loads(json_str, cls=_JSONDecoder)
        return cls.from_dict(tmp)

    @staticmethod
    def hash_for_dict(data: dict):
        return hashlib.sha1(json.dumps(data, cls=_JSONEncoder).encode()).hexdigest()
    

    def hash(self):
        return FactoryObject.hash_for_dict(self.to_dict())
        
    @abc.abstractmethod
    def _to_dict(self)->dict:
        pass

    @classmethod
    def from_dict(cls, data: dict)->object:
        return cls(**{k:v for k,v in data.items() if k != 'cls'})

class BaseDatedCurve(abc.ABC):
    @abc.abstractmethod
    def value(self, ref_date: dt.datetime, d: dt.datetime)->np.ndarray:#, dt_grid: DateTimeGrid)->np.ndarray:
        pass


class HasExpectedCashflows(abc.ABC):
    @abc.abstractmethod
    def expected_cashflows(self)->List[Tuple[dt.datetime, float]]:
        pass
