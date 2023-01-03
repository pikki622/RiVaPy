from datetime import datetime
from rivapy.tools.interfaces import BaseDatedCurve, HasExpectedCashflows
from rivapy.marketdata import DiscountCurveParametrized, ConstantRate
from rivapy.pricing.pricing_request import PricingRequest
from scipy.optimize import brentq

class SimpleCashflowPricer:
    @staticmethod
    def pv_cashflows(val_date: datetime, specification: HasExpectedCashflows, discount_curve: BaseDatedCurve)->float:
        cashflows =  specification.expected_cashflows()
        pv_cashflows = 0.0
        for c in cashflows:
            if c[0]>= val_date:
                pv_cashflows += discount_curve.value(val_date, c[0])*c[1]
        print(pv_cashflows)
        return pv_cashflows
        
    @staticmethod
    def compute_yield(target_dirty_price: float, val_date: datetime, 
                    specification: HasExpectedCashflows)->float:
        def target_function(r: float)->float:
            dc = DiscountCurveParametrized('', val_date, ConstantRate(r))
            return SimpleCashflowPricer.pv_cashflows(val_date, specification, dc)-target_dirty_price
        return brentq(target_function, -0.2, 0.2, full_output = False)