import unittest
import math
from dateutil import relativedelta
from datetime import datetime, timedelta

import RiVaPy

class CDSTest(unittest.TestCase):
    def pricing_test(self):
        """Test simple CDS pricing using ISDA model.
        """
        refdate = datetime(2020,1,1)
        #yield curve
        days_to_maturity = [1, 180, 360, 720, 3*360, 4*360, 5*360, 10*360]
        rates = [-0.0065, 0.0003, 0.0059, 0.0086, 0.0101, 0.012, 0.016, 0.02]
        dates = [refdate + timedelta(days=d) for d in days_to_maturity]
        dsc_fac = [math.exp(-rates[i]*days_to_maturity[i]/360) for i in range(len(days_to_maturity))]
        dc = RiVaPy.marketdata.DiscountCurve('CDS_interest_rate', refdate, dates, 
                                            dsc_fac, RiVaPy.enums.DayCounter.ACT360, RiVaPy.enums.InterpolationType.LINEAR, 
                                            RiVaPy.enums.ExtrapolationType.LINEAR)
        hazard_rates = [0, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.05]
        sc = RiVaPy.marketdata.SurvivalCurve('Survival',refdate,dates,hazard_rates)

        recoveries = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
        recovery = RiVaPy.marketdata.DatedCurve('Recovery',refdate,dates,recoveries, RiVaPy.enums.DayCounter.ACT360, 
                                                RiVaPy.enums.InterpolationType.LINEAR, RiVaPy.enums.ExtrapolationType.LINEAR)

        payment_dates = [refdate + relativedelta.relativedelta(years=i) for i in range(10)]
        spec = RiVaPy.instruments.CDSSpecification(premium = 0.1,premium_pay_dates = payment_dates, notional = 1.0)

        cds_pricing_data = RiVaPy.pricing.CDSPricingData(spec=spec, val_date=refdate, discount_curve=dc, survival_curve=sc, recovery_curve=recovery)

        pr = RiVaPy.pricing.price(cds_pricing_data)
        self.assertAlmostEqual(pr.pv_protection, 0.0, 3)

if __name__ == '__main__':
    unittest.main()

