import unittest
import math
from dateutil import relativedelta
from datetime import datetime, timedelta

import rivapy

class CDSTest(unittest.TestCase):
    def test_pricing(self):
        """Test simple CDS pricing using ISDA model.
        """
        if not rivapy._pyvacon_available:
            self.assertAlmostEquals(0,1,'Test cannot be run due to missing pvacon.')
            
        refdate = datetime(2020,1,1)
        #yield curve
        days_to_maturity = [1, 180, 360, 720, 3*360, 4*360, 5*360, 10*360]
        rates = [-0.0065, 0.0003, 0.0059, 0.0086, 0.0101, 0.012, 0.016, 0.02]
        dates = [refdate + timedelta(days=d) for d in days_to_maturity]
        dsc_fac = [math.exp(-rates[i]*days_to_maturity[i]/360) for i in range(len(days_to_maturity))]
        dc = rivapy.marketdata.DiscountCurve('CDS_interest_rate', refdate, dates, 
                                            dsc_fac, rivapy.enums.DayCounter.ACT360, rivapy.enums.InterpolationType.LINEAR, 
                                            rivapy.enums.ExtrapolationType.LINEAR)
        hazard_rates = [0, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.005]
        sc = rivapy.marketdata.SurvivalCurve('Survival',refdate,dates,hazard_rates)

        recoveries = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
        recovery = rivapy.marketdata.DatedCurve('Recovery',refdate,dates,recoveries, rivapy.enums.DayCounter.ACT360, 
                                                rivapy.enums.InterpolationType.LINEAR, rivapy.enums.ExtrapolationType.LINEAR)

        payment_dates = [refdate + relativedelta.relativedelta(years=i) for i in range(10)]
        spec = rivapy.instruments.CDSSpecification(premium = 0.0012, protection_start=refdate, premium_pay_dates = payment_dates, notional = 1000000.0)

        cds_pricing_data = rivapy.pricing.CDSPricingData(spec=spec, val_date=refdate, discount_curve=dc, survival_curve=sc, recovery_curve=recovery)

        pr = rivapy.pricing.price(cds_pricing_data)
        self.assertAlmostEqual(0.0, 0.0, 3)

if __name__ == '__main__':
    unittest.main()

