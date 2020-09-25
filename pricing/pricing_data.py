# from datetime import datetime
import pyvacon.analytics as _analytics
from RiVaPy.marketdata import \
    BaseDatedCurve, \
    DiscountCurve, \
    SurvivalCurve
from typing import Union as _Union
from datetime import date, datetime
from RiVaPy.instruments.specifications import Bond
from RiVaPy.tools._converter import _add_converter
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
# getPricingData = _converter(_analytics.getPricingData)


class BasePricingData:
    def __init__(self, pricer: str, pricing_request: PricingRequest):
        self.pricer = pricer
        self.pricing_request = pricing_request
        # TODO: analyse if simulationData is needed (here)

    @property
    def pricer(self):
        return self.__pricer

    @pricer.setter
    def pricer(self, pricer):
        self.__pricer = pricer

    @property
    def pricing_request(self):
        return self.pricing_request

    @pricing_request.setter
    def pricing_request(self, pricing_request):
        self.__pricing_request = pricing_request


class BondPricingData(BasePricingData):
    def __init__(self, bond: Bond, valuation_date: _Union[date, datetime], discount_curve: DiscountCurve,
                 fixing_curve: DiscountCurve, parameters: BondPricingParameter, pricing_request: PricingRequest,
                 pricer: str = 'BondPricer', past_fixing: float = None, survival_curve: SurvivalCurve = None,
                 recovery_curve: BaseDatedCurve = None):
        super().__init__(pricer, pricing_request)
        self.__bond = bond  # spec
        self.valuation_date = valuation_date  # valDate
        self.discount_curve = discount_curve  # discountCurve
        self.fixing_curve = fixing_curve  # fixingCurve
        self.parameters = parameters  # param
        self.past_fixing = past_fixing  # pastFixing
        self.survival_curve = survival_curve  # sc
        self.recovery_curve = recovery_curve  # recoveryCurve

    @property
    def bond(self):
        return self.__bond

    @property
    def valuation_date(self):
        return self.__valuation_date

    @valuation_date.setter
    def valuation_date(self, valuation_date):
        self.__valuation_date = datetime_to_date(valuation_date)

    @property
    def discount_curve(self):
        return self.__discount_curve

    @discount_curve.setter
    def discount_curve(self, discount_curve):
        self.__discount_curve = discount_curve

    @property
    def fixing_curve(self):
        return self.__fixing_curve

    @fixing_curve.setter
    def fixing_curve(self, fixing_curve):
        self.__fixing_curve = fixing_curve

    @property
    def parameters(self):
        return self.__parameters

    @parameters.setter
    def parameters(self, parameters):
        self.__parameters = parameters

    @property
    def past_fixing(self):
        return self.__past_fixing

    @past_fixing.setter
    def past_fixing(self, past_fixing):
        self.__past_fixing = past_fixing

    @property
    def survival_curve(self):
        return self.__survival_curve

    @survival_curve.setter
    def survival_curve(self, survival_curve):
        self.__survival_curve = survival_curve

    @property
    def recovery_curve(self):
        return self.__recovery_curve

    @recovery_curve.setter
    def recovery_curve(self, recovery_curve):
        self.__recovery_curve = recovery_curve

    @property
    def obj_id(self):
        return self.__bond.obj_id
