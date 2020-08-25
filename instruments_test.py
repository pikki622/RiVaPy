import unittest
import math
from dateutil import relativedelta
from datetime import datetime, timedelta

import crumble

class CDSTest(unittest.TestCase):

    def test_pricing_default_prob_one(self):
        """Test simple CDS pricing using ISDA model and default prob very close to one.
        """
        refdate = datetime(2020,1,1)
        #yield curve
        days_to_maturity = [1, 180, 360, 720, 3*360, 4*360, 5*360, 10*360]
        dates = [refdate + timedelta(days=d) for d in days_to_maturity]
        rates = [0.0] * len(dates)
        dsc_fac = [math.exp(-rates[i]*days_to_maturity[i]/360) for i in range(len(days_to_maturity))]
        dc = crumble.marketdata.DiscountCurve('CDS_interest_rate', refdate, dates, 
                                            dsc_fac, crumble.enums.DayCounter.ACT360, crumble.enums.InterpolationType.LINEAR, 
                                            crumble.enums.ExtrapolationType.LINEAR)
        hazard_rates = [1.0]*len(dates)
        sc = crumble.marketdata.SurvivalCurve('Survival',refdate,dates,hazard_rates)

        recoveries = [0.0]*len(dates)
        recovery = crumble.marketdata.DatedCurve('Recovery',refdate,dates, recoveries, crumble.enums.DayCounter.ACT360, 
                                                crumble.enums.InterpolationType.LINEAR, crumble.enums.ExtrapolationType.LINEAR)

        payment_dates = [refdate + relativedelta.relativedelta(years=i) for i in range(10)]
        spec = crumble.instruments.CDSSpecification(premium = 0.1, premium_pay_dates = payment_dates, 
                                                    protection_start = refdate, notional = 1.0)

        cds_pricing_data = crumble.pricing.CDSPricingData(spec=spec, val_date=refdate, 
                                        discount_curve=dc, 
                                        survival_curve=sc, 
                                        recovery_curve=recovery)

        pr = crumble.pricing.price(cds_pricing_data)
        self.assertAlmostEqual(pr.pv_protection, 1.0, places=3) #zero recovery, therefore 0% value of protection

        cds_pricing_data.recovery_curve = crumble.marketdata.DatedCurve('Recovery',refdate,dates,[1.0]*len(dates), crumble.enums.DayCounter.ACT360, 
                                                crumble.enums.InterpolationType.LINEAR, crumble.enums.ExtrapolationType.LINEAR)

        pr = crumble.pricing.price(cds_pricing_data)
        self.assertAlmostEqual(pr.pv_protection, 0.0, places=3) # 100% recovery, therefore zero value of protection
        
        cds_pricing_data.recovery_curve = crumble.marketdata.DatedCurve('Recovery',refdate,dates,[0.4]*len(dates), crumble.enums.DayCounter.ACT360, 
                                                crumble.enums.InterpolationType.LINEAR, crumble.enums.ExtrapolationType.LINEAR)

        pr = crumble.pricing.price(cds_pricing_data)
        self.assertAlmostEqual(pr.pv_protection, 0.6, places=3) # 40% recovery, therefore 60% value of protection

        # now set recoveries in curve but also recovery of zero into 
        cds_pricing_data.spec.recovery = 0.0
        pr = crumble.pricing.price(cds_pricing_data)
        self.assertAlmostEqual(pr.pv_protection, 1.0, places=3)

if __name__ == '__main__':
    unittest.main()

