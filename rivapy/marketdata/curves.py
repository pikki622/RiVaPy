from typing import List, Union, Tuple
from enum import Enum
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt

from pyvacon.finance.marketdata import DiscountCurve as _DiscountCurve
import pyvacon as _pyvacon

class DiscountCurve:

    class InterpolationMethod(Enum):
        """Enum of different interpolation methods that can be used in the discount curve
        """
        HAGAN_DF = 1
        LINEAR = 2


    def __init__(self, 
                id: str,
                refdate: Union[datetime, date], 
                dates: List[Union[datetime, date]], 
                df: List[float],
                interpolation: InterpolationMethod = InterpolationMethod.HAGAN_DF):
        if len(dates) < 1:
            raise Exception('Please specify at least one date and discount factor')
        if len(dates) != len(df):
            raise Exception('List of dates and discount factors must have equal length.')
        self.values = sorted(zip(dates,df), key=lambda tup: tup[0]) # zip dates and discount factors and sort by dates
        if isinstance(refdate, datetime):
            self.refdate = refdate
        else:
            self.refdate = datetime(refdate,0,0,0)
        if isinstance(interpolation, DiscountCurve.InterpolationMethod):
            self.interpolation = interpolation.name
        else:
            self.interpolation = interpolation
        self.id = id
        #check if dates are monotonically increasing and if first date is greather then refdate
        if self.values[0][0] < refdate:
            raise Exception('First date must be equal or greater then reference date.')
        if self.values[0][0] > refdate:
            self.values = [(self.refdate, 1.0)] + self.values
        if self.values[0][1] != 1.0:
            raise Exception('Discount factor for today must equal 1.0.')
        for i in range(1,len(self.values)):
            if self.values[i-1]>= self.values[i]:
                raise Exception('Dates must be given in monotonically increasing order.')
        self._pyvacon_obj = None

    def get_dates(self)->Tuple[datetime]:
        """Return list of dates of curve

        Returns:
            Tuple[datetime]: List of dates
        """
        x,y = zip(*self.values)
        return x

    def get_df(self)->Tuple[float]:
        """Return list of discount factors

        Returns:
            Tuple[float]: List of discount factors
        """
        x,y = zip(*self.values)
        return y

    def value(self, refdate: Union[date, datetime], d: Union[date, datetime])->float:
        """Return discount factor for a given date

        Args:
            refdate (Union[date, datetime]): The reference date. If the reference date is in the future (compared to the curves reference date), the forward discount factor will be returned.
            d (Union[date, datetime]): The date for which the discount factor will be returned

        Returns:
            float: discount factor
        """
        if not isinstance(refdate, datetime):
            refdate = datetime(refdate,0,0,0)
        if not isinstance(d, datetime):
            d = datetime(d,0,0,0)

        if self._pyvacon_obj is None:
            self._pyvacon_obj = _DiscountCurve(self.id, self.refdate, 
                                            [x for x in self.get_dates()], [x for x in self.get_df()], 
                                            _pyvacon.finance.definition.DayCounter.Type.Act365Fixed, 
                                            _pyvacon.numerics.interpolation.InterpolationType.LINEAR,
                                            _pyvacon.numerics.extrapolation.ExtrapolationType.NONE)
        return self._pyvacon_obj.value(refdate, d)

    def plot(self):
        dates = self.get_dates()
        dates_new = [dates[0]]
        days = 10
        for i in range(1,len(dates)):
            while dates_new[-1] + timedelta(days=days) < dates[i]:
                dates_new.append(dates_new[-1]+ timedelta(days=days))
        dates_new.append(dates[-1])
        values = [self.value(self.refdate, d) for d in dates_new]
        plt.plot(dates_new, values, label=self.id)
            