from typing import List, Union, Tuple
from enum import Enum
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import math

from pyvacon.finance.marketdata import EquityForwardCurve as _EquityForwardCurve

from rivapy.enums import DayCounterType, InterpolationType, ExtrapolationType

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
                interpolation: InterpolationType = InterpolationType.HAGAN_DF,
                extrapolation: ExtrapolationType = ExtrapolationType.NONE,
                daycounter: DayCounterType = DayCounterType.Act365Fixed):
        """Discountcurve

        Args:
            id (str): Identifier of the discount curve.
            refdate (Union[datetime, date]): Reference date of the discount curve.
            dates (List[Union[datetime, date]]): List of dates belonging to the list of discount factors. All dates must be distinct and equal or after the refdate, otherwise an exception will be thrown.
            df (List[float]): List of discount factors. Length of list of discount factors must equal to length of list of dates, otherwise an exception will be thrown.
            interpolation (enums.InterpolationType, optional): Defaults to InterpolationType.HAGAN_DF.
            extrapolation (enums.ExtrapolationType, optional): Defaults to ExtrapolationType.NONE which does not allow to compute a discount factor for a date past all given dates given to this constructor.
            daycounter (enums.DayCounterType, optional): Daycounter used within the interpolation formula to compute a discount factor between two dates from the dates-list above. Defaults to DayCounterType.Act365Fixed.

        Raises:
            Exception: [description]
            Exception: [description]
            Exception: [description]
            Exception: [description]
            Exception: [description]
        """
        if len(dates) < 1:
            raise Exception('Please specify at least one date and discount factor')
        if len(dates) != len(df):
            raise Exception('List of dates and discount factors must have equal length.')
        self.values = sorted(zip(dates,df), key=lambda tup: tup[0]) # zip dates and discount factors and sort by dates
        if isinstance(refdate, datetime):
            self.refdate = refdate
        else:
            self.refdate = datetime(refdate,0,0,0)
        if not isinstance(interpolation, InterpolationType):
            raise TypeError('Interpolation is not of type enums.InterpolationType')
        self.interpolation = interpolation
        if not isinstance(extrapolation, ExtrapolationType):
            raise TypeError('Extrapolation is not of type enums.ExtrapolationType')
        self.extrapolation = extrapolation
        if not isinstance(daycounter, DayCounterType):
            raise TypeError('Daycounter is not of type enums.DaycounterType')
        self.daycounter = daycounter
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
        if refdate < self.refdate:
            raise Exception('The given reference date is before the curves reference date.')
        return self._get_pyvacon_obj.value(refdate, d)

    def _get_pyvacon_obj(self):
        if self._pyvacon_obj is None:
            self._pyvacon_obj = _DiscountCurve(self.id, self.refdate, 
                                            [x for x in self.get_dates()], [x for x in self.get_df()], 
                                            self.daycounter, 
                                            self.interpolation,
                                            self.extrapolation)
        return self._pyvacon_obj

    def plot(self, days:int = 10, discount_factors: bool = False, **kwargs):
        """Plots the discount curve using matplotlibs plot function.
        The timegrid includes the dates of the discount curve. Here either the discount factors or the zero rates (continuously compounded, ACT365 yearfraction) are plotted.

        Args:
            days (int, optional): The number of days between two plotted rates/discount factors. Defaults to 10.
            discount_factors (bool, optional): If True, discount factors will be plotted, otherwise the rates. Defaults to False.
            **kwargs: optional arguments that will be directly passed to the matplotlib plto function
        """
        dates = self.get_dates()
        dates_new = [dates[0]]
        for i in range(1,len(dates)):
            while dates_new[-1] + timedelta(days=days) < dates[i]:
                dates_new.append(dates_new[-1]+ timedelta(days=days))
        dates_new.append(dates[-1])
        values = [self.value(self.refdate, d) for d in dates_new]

        if not discount_factors:
            for i in range(1,len(values)):
                dt = float((dates_new[i]-self.refdate).days)/365.0
                values[i] = -math.log(values[i])/dt
        values[0] = values[1]
        plt.plot(dates_new, values, label=self.id, **kwargs)
            

class EquityForwardCurve:
    def __init__(self, 
                    spot: float, 
                    funding_curve: DiscountCurve, 
                    borrow_curve: DiscountCurve, 
                    div_table):
        """Equity Forward Curve

        Args:
            
            spot (float): Current spot
            discount_curve (DiscountCurve): [description]
            funding_curve (DiscountCurve): [description]
            borrow_curve (DiscountCurve): [description]
            div_table ([type]): [description]
        """
        self.spot = spot
        
        self.bc = borrow_curve
        self.fc = funding_curve
        self.div = div_table
        self._pyvacon_obj = None
        self.refdate = self.fc.refdate
        if self.bc is not None:
            if self.refdate < self.bc.refdate:
                self.refdate = self.bc.refdate

        if self.div is not None:
            if hasattr(self.div, 'refdate'):
                if self.refdate < self.div.refdate:
                    self.refdate = self.div.refdate

    def _get_pyvacon_obj(self):
        if self._pyvacon_obj is None:
            args = {}
            if hasattr(self.fc, '_get_pyvacon_obj'):
                fc = self.fc._get_pyvacon_obj()
            else:
                fc = self.fc
            
            if hasattr(self.bc, '_get_pyvacon_obj'):
                bc = self.bc._get_pyvacon_obj()
            else:
                bc = self.bc
            self._pyvacon_obj = _EquityForwardCurve(self.refdate, self.spot, fc, bc, self.div) 
        return self._pyvacon_obj
           
    def value(self, refdate, expiry):
        return self._get_pyvacon_obj().value(refdate, expiry)

    def plot(self, days:int = 10, days_end: int = 10*365, **kwargs):
        """Plots the forward curve using matplotlibs plot function.
        
        Args:
            days (int, optional): The number of days between two plotted rates/discount factors. Defaults to 10.
            days_end (int. optional): Number of days when plotting will end. Defaults to 10*365 (10yr)
            **kwargs: optional arguments that will be directly passed to the matplotlib plto function
        """
        dates = [self.refdate + timedelta(days=i) for i in range(0, days_end, days)]
        values = [self.value(self.refdate, d) for d in dates]
        plt.plot(dates, values, **kwargs)
        plt.xlabel('expiry')
        plt.ylabel('forward value')