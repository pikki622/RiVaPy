
from typing import List, Union, Tuple
from rivapy.marketdata.curves import *

import pyvacon.finance.marketdata as _mkt_data
import pyvacon.finance.utils as _utils

InflationIndexForwardCurve = _mkt_data.InflationIndexForwardCurve
SurvivalCurve = _mkt_data.SurvivalCurve
DatedCurve = _mkt_data.DatedCurve
# DividendTable = _mkt_data.DividendTable

class DividendTable:
    def __init__(self, id: str,
                refdate: datetime, 
                ex_dates: List[datetime],
                pay_dates: List[datetime],
                div_yield: List[float],
                div_cash: List[float],
                tax_factors: List[float]):
        """[summary]

        Args:
            id (str): [description]
            refdate (datetime): [description]
            ex_dates (List[datetime]): [description]
            pay_dates (List[datetime]): [description]
            div_yield (List[float]): [description]
            div_cash (List[float]): [description]
            tax_factors (List[float]): [description]

        Yields:
            [type]: [description]
        """
        self.id = id
        self.refdate = refdate
        self.ex_dates = ex_dates
        self.pay_dates = pay_dates
        self.div_yield = div_yield
        self.div_cash = div_cash
        self.tax_factors = tax_factors
        self._pyvacon_obj = None

    def _get_pyvacon_obj(self):
        if self._pyvacon_obj is None:
            self._pyvacon_obj = _mkt_data.DividendTable(self.id, self.refdate, self.ex_dates, self.div_yield, self.div_cash, self.tax_factors, self.pay_dates)
        return self._pyvacon_obj

class VolatilitySurfaceFlat:
    def __init__(self, id: str, refdate: datetime, forward_curve, vol: float, daycounter: DayCounterType=DayCounterType.Act365Fixed):
        self.id = id
        self.refdate = refdate
        self.fwd_curve = forward_curve
        self.vol = vol
        self.daycounter = daycounter
        self._pyvacon_obj = None

    def _get_pyvacon_obj(self):
        if self._pyvacon_obj is None:
            p = _mkt_data.VolatilityParametrizationFlat(self.vol)
            self._pyvacon_obj = _mkt_data.VolatilitySurface(self.id, self.refdate, self.fwd_curve._get_pyvacon_obj(), self.daycounter.name, p)
        return self._pyvacon_obj
    
class VolatilitySurface:
    def __init__(self, id: str, refdate: datetime, forward_curve, daycounter, vol_param):
        self.id = id
        self.refdate = refdate
        self.forward_curve = forward_curve
        self.daycounter = daycounter
        self.vol_param = vol_param
        self._pyvacon_obj = None
        
           
    def _get_pyvacon_obj(self):
        if self._pyvacon_obj is None:
            self._pyvacon_obj = _mkt_data.VolatilitySurface(self.id, self.refdate, \
                self.forward_curve._get_pyvacon_obj(),self.daycounter.name, self.vol_param)
        return self._pyvacon_obj
    
    def calcImpliedVol(self, refdate: datetime, expiry: datetime, strike: float)->float:
        # # convert strike into x_strike 
        # forward_curve_obj = self.forward_curve._get_pyvacon_obj() 
        # x_strike = _utils.computeXStrike(strike, forward_curve_obj.value(refdate, expiry), forward_curve_obj.discountedFutureCashDivs(refdate, expiry))
        # print(x_strike)
        # if x_strike < 0:
        #     raise Exception(f'The given strike value seems implausible compared to the discounted future cash dividends\
        #         ({forward_curve_obj.discountedFutureCashDivs(refdate, expiry)}).')
        # vol = self._get_pyvacon_obj()
        # return vol.calcImpliedVol(refdate, expiry, x_strike)
        return 6.0

        
            