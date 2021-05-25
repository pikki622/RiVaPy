
from pyvacon.pyvacon_swig import GlobalSettings
from rivapy import enums
from typing import List, Union, Tuple
from rivapy.marketdata.curves import *

import pyvacon.finance.marketdata as _mkt_data
import pyvacon.finance.utils as _utils
import pyvacon.finance.pricing as _pricing

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
        """Flat volatility parametrization

        Args:
            vol (float): Constant volatility.
        """
        self.vol = vol
        self._pyvacon_obj = None
        
    def _get_pyvacon_obj(self):
        if self._pyvacon_obj is None:
            self._pyvacon_obj = _mkt_data.VolatilityParametrizationFlat(self.vol)  
        return self._pyvacon_obj
    
class VolatilityParametrizationTerm:
    def __init__(self, expiries: List[float], fwd_atm_vols: List[float]):
        """Term volatility parametrization

        Args:
            expiries (List[float]): List of expiration dates.
            fwd_atm_vols (List[float]): List of at-the-money volatilities.
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
        """SSVI volatility parametrization
        https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2033323

        Args:
            expiries (List[float]): List of expiration dates.
            fwd_atm_vols (List[float]): List of at-the-money volatilities.
            rho (float): Responsible for the skewness of the volatility surface.
            eta (float): Responsible for the curvature.
            gamma (float): Responsible for the "rate of decay".

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
        """Volatility surface

        Args:
            id (str): Identifier (name) of the volatility surface.
            refdate (datetime): Valuation date.
            forward_curve (rivapy.market_data.EquityForwardCurve): Forward curve.
            daycounter (enums.DayCounterType): [description]
            vol_param ([VolatilityParametrizationFlat,VolatilityParametrizationTerm,VolatilityParametrizationSSVI]): Volatility parametrization.
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
    
    def calc_implied_vol(self, refdate: datetime, expiry: datetime, strike: float)->float:
        """Calculate implied volatility

        Args:
            refdate (datetime): Valuation date.
            expiry (datetime): Expiration date.
            strike (float): Strike price.

        Raises:
            Exception: [description]

        Returns:
            float: Implied volatility.
        """
        # convert strike into x_strike 
        forward_curve_obj = self.forward_curve._get_pyvacon_obj() 
        x_strike = _utils.computeXStrike(strike, forward_curve_obj.value(refdate, expiry), forward_curve_obj.discountedFutureCashDivs(refdate, expiry))
        if x_strike < 0:
            raise Exception(f'The given strike value seems implausible compared to the discounted future cash dividends\
                ({forward_curve_obj.discountedFutureCashDivs(refdate, expiry)}).')
        vol = self._get_pyvacon_obj()
        return vol.calcImpliedVol(refdate, expiry, x_strike)
    
    
    def shift_fwd_curve(self, id: str, shifted_forward_curve, stickyness_assumption=None):
        """Creates a new volatility surface using a shifted forward curve.

        Args:
            id (str): Identifier (name) of the shifted volatility surface.
            shifted_forward_curve ([type]): Shifted equity forward curve.


        Returns:
            VolatilitySurface: Shifted volatility surface.
        """
        return VolatilitySurfaceShifted(id, self.refdate, shifted_forward_curve, self.daycounter, self.vol_param, self.forward_curve, stickyness_assumption)
            
    def shift_fwd_spot(self, id: str,  shifted_spot: float):
        """Creates a new volatility surface using a shifted spot (ceteris paribus).

        Args:
            id (str): Identifier (name) of the shifted volatility surface.
            shifted_spot (float): Shifted spot.

        Returns:
           VolatilitySurface: Shifted volatility surface.
        """
        shifted_forward_curve = EquityForwardCurve(shifted_spot, self.forward_curve.fc , self.forward_curve.bc, self.forward_curve.div)
        # old_old return VolatilitySurface(self.id, self.refdate, shifted_forward_curve, self.daycounter, self.vol_param)
        return VolatilitySurface(id, self.refdate, shifted_forward_curve, self.daycounter, self.vol_param, _vol_shift="fwd" )
    # old _mkt_data.VolatilitySurface.createVolatilitySurfaceShiftedFwd(self._get_pyvacon_obj(),shifted_forward_curve._get_pyvacon_obj())
    
    @staticmethod        
    def set_stickyness(vol_stickyness: enums.VolatilityStickyness):
        if vol_stickyness is enums.VolatilityStickyness.StickyXStrike:
            _pricing.GlobalSettings.setVolatilitySurfaceFwdStickyness(_pricing.VolatilitySurfaceFwdStickyness.Type.StickyXStrike)
        elif vol_stickyness is enums.VolatilityStickyness.StickyStrike:
            _pricing.GlobalSettings.setVolatilitySurfaceFwdStickyness(vol_stickyness)
        elif vol_stickyness is enums.VolatilityStickyness.StickyFwdMoneyness:
            _pricing.GlobalSettings.setVolatilitySurfaceFwdStickyness(_pricing.VolatilitySurfaceFwdStickyness.Type.StickyFwdMoneyness)
        elif vol_stickyness is enums.VolatilityStickyness.NONE:
            _pricing.GlobalSettings.setVolatilitySurfaceFwdStickyness(_pricing.VolatilitySurfaceFwdStickyness.Type.NONE)
        else:
            raise Exception ('Error')

class VolatilitySurfaceShifted():

    def __init__(self, id: str, refdate: datetime, shifted_forward_curve, daycounter, vol_param, old_forward_curve, stickyness_assumption):
        self.id = id
        self.refdate = refdate
        self.shifted_forward_curve = shifted_forward_curve
        self.daycounter = daycounter
        self.vol_param = vol_param
        self.old_forward_curve = old_forward_curve
        self.stickyness_assumption = stickyness_assumption
        self._pyvacon_obj = None 
        
    def _get_pyvacon_obj(self):
        if self._pyvacon_obj is None:
            self._pyvacon_obj = _mkt_data.VolatilitySurface(self.id, self.refdate, \
                self.shifted_forward_curve._get_pyvacon_obj(),self.daycounter.name, self.vol_param._get_pyvacon_obj())
        return self._pyvacon_obj
    
         
    def calc_implied_vol(self, refdate: datetime, expiry: datetime, strike: float)->float:
        shifted_forward_curve_obj = self.shifted_forward_curve._get_pyvacon_obj()
        old_forward_curve_obj =  self.old_forward_curve._get_pyvacon_obj()
        
        if self.stickyness_assumption == enums.VolatilityStickyness.StickyStrike:
            x_strike = _utils.computeXStrike(strike, shifted_forward_curve_obj.value(refdate, expiry), shifted_forward_curve_obj.discountedFutureCashDivs(refdate, expiry))
        elif self.stickyness_assumption == enums.VolatilityStickyness.StickyXStrike:
            x_strike = _utils.computeXStrike(strike, old_forward_curve_obj.value(refdate, expiry), old_forward_curve_obj.discountedFutureCashDivs(refdate, expiry))
        elif self.stickyness_assumption == enums.VolatilityStickyness.StickyFwdMoneyness:
            moneyness = strike/old_forward_curve_obj.value(refdate, expiry)
            calc_strike =  shifted_forward_curve_obj.value(refdate, expiry) * moneyness
            x_strike = _utils.computeXStrike(calc_strike, shifted_forward_curve_obj.value(refdate, expiry), shifted_forward_curve_obj.discountedFutureCashDivs(refdate, expiry))
        if x_strike < 0:
            raise Exception('The given strike value seems implausible compared to the discounted future cash dividends')
        vol = self._get_pyvacon_obj()
        return vol.calcImpliedVol(refdate, expiry, x_strike)