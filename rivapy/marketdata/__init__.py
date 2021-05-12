
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
    
class VolatilityParametrizationFlat:
    def __init__(self,vol: float):
        """[summary]

        Args:
            vol (float): [description]
        """
        self.vol = vol
        self._pyvacon_obj = None
        
    def _get_pyvacon_obj(self):
        if self._pyvacon_obj is None:
            self._pyvacon_obj = _mkt_data.VolatilityParametrizationFlat(self.vol)  
        return self._pyvacon_obj
    
class VolatilityParametrizationTerm:
    def __init__(self, expiries: List[float], fwd_atm_vols: List[float]):
        """[summary]

        Args:
            expiries (List[float]): [description]
            fwd_atm_vols (List[float]): [description]
        """
        self.expiries = expiries
        self.fwd_atm_vols = fwd_atm_vols
        self._pyvacon_obj = None
        
    def _get_pyvacon_obj(self):
        if self._pyvacon_obj is None:
            self._pyvacon_obj = _mkt_data.VolatilityParametrizationTerm(self.expiries, self.fwd_atm_vols)  
        return self._pyvacon_obj
    
class VolatilityParametrizationSSVI:
    def __init__(self, expiries: List[float], fwd_atm_vols: List[float], rho: float, eta: float, gamma: float):
        """[summary]

        Args:
            expiries (List[float]): [description]
            fwd_atm_vols (List[float]): [description]
            rho (float): [description]
            eta (float): [description]
            gamma (float): [description]
        """
        self.expiries = expiries
        self.fwd_atm_vols = fwd_atm_vols
        self.rho = rho
        self.eta = eta
        self.gamma = gamma
        self._pyvacon_obj = None

        
    def _get_pyvacon_obj(self):
        if self._pyvacon_obj is None:
            self._pyvacon_obj = _mkt_data.VolatilityParametrizationSSVI(self.expiries, self.fwd_atm_vols, self.rho, self.eta, self.gamma)  
        return self._pyvacon_obj
    
class VolatilitySurface:
    def __init__(self, id: str, refdate: datetime, forward_curve, daycounter, vol_param):
        """[summary]

        Args:
            id (str): [description]
            refdate (datetime): [description]
            forward_curve ([type]): [description]
            daycounter ([type]): [description]
            vol_param ([type]): [description]
        """
        self.id = id
        self.refdate = refdate
        self.forward_curve = forward_curve
        self.daycounter = daycounter
        self.vol_param = vol_param
        self._pyvacon_obj = None        
           
    def _get_pyvacon_obj(self):
        if self._pyvacon_obj is None:
            self._pyvacon_obj = _mkt_data.VolatilitySurface(self.id, self.refdate, \
                self.forward_curve._get_pyvacon_obj(),self.daycounter.name, self.vol_param._get_pyvacon_obj())
        return self._pyvacon_obj
    
    def calcImpliedVol(self, refdate: datetime, expiry: datetime, strike: float)->float:
        # convert strike into x_strike 
        forward_curve_obj = self.forward_curve._get_pyvacon_obj() 
        x_strike = _utils.computeXStrike(strike, forward_curve_obj.value(refdate, expiry), forward_curve_obj.discountedFutureCashDivs(refdate, expiry))
        if x_strike < 0:
            raise Exception(f'The given strike value seems implausible compared to the discounted future cash dividends\
                ({forward_curve_obj.discountedFutureCashDivs(refdate, expiry)}).')
        vol = self._get_pyvacon_obj()
        return vol.calcImpliedVol(refdate, expiry, x_strike)

        
            