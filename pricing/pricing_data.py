# from datetime import datetime
import pyvacon.analytics as _analytics
from pyvacon.marketdata.analytics_classes import \
    BaseDatedCurve, \
    DiscountCurve, \
    SurvivalCurve
from typing import Union
from datetime import date, datetime
from RiVaPy.instruments.specifications import Bond
from RiVaPy.tools._converter import \
    _add_converter, \
    converter as _converter
from RiVaPy.tools.datetools import datetime_to_date


class CDSPricingData:
    def __init__(self, spec, val_date, discount_curve, survival_curve, recovery_curve=None):
        self.spec = spec
        self.val_date = val_date
        self.discount_curve = discount_curve
        self.survival_curve = survival_curve
        self.recovery_curve = recovery_curve
        self._pricer_type = 'ISDA'
        
    def price(self):
        pass


# BondPricingData = _add_converter(_analytics.BondPricingData)
BondPricingParameter = _add_converter(_analytics.BondPricingParameter)
PricingRequest = _add_converter(_analytics.PricingRequest)
getPricingData = _converter(_analytics.getPricingData)


class BondPrice:
    def __init__(self, bond: Bond, valuation_date: Union[date, datetime], discount_curve: DiscountCurve,
                 fixing_curve: DiscountCurve, survival_curve: SurvivalCurve, recovery_curve: BaseDatedCurve,
                 parameters: BondPricingParameter, past_fixing: float):
        self.__bond = bond
        self.__valuation_date = datetime_to_date(valuation_date)
        self.__discount_curve = discount_curve
        self.__fixing_curve = fixing_curve
        self.__survival_curve = survival_curve
        self.__recovery_curve = recovery_curve
        self.__parameters = parameters
        self.__past_fixing = past_fixing

    @property
    def bond(self):
        return self.__bond

    @property
    def valuation_date(self):
        return self.__valuation_date

    @property
    def discount_curve(self):
        return self.__discount_curve

    @property
    def fixing_curve(self):
        return self.__fixing_curve

    @property
    def survival_curve(self):
        return self.__survival_curve

    @property
    def recovery_curve(self):
        return self.__recovery_curve

    @property
    def parameters(self):
        return self.__parameters

    @property
    def past_fixing(self):
        return self.__past_fixing
