import unittest
import numpy as np
import datetime as dt

import rivapy
from rivapy.marketdata import VolatilityGridParametrization, VolatilitySurface, DiscountCurve, EquityForwardCurve, enums
from rivapy import enums

class VolatilityGridParamTest(unittest.TestCase):
    def test_exceptions(self):
        expiries = np.linspace(1.0/365.0, 4.0, 10)
        strikes = np.linspace(0.4, 1.6, 100)
        # test for exception if number of strikes does not match number of cols
        vols =np.empty((expiries.shape[0], 1))
        try:
            vol_grid_param = VolatilityGridParametrization(expiries, strikes, vols)
            self.assertFalse(False)
        except:
            self.assertFalse(True)

    def test_calc_implied_vol(self):
        """Simple tests for class VolatilityGridParametrization
        """
        expiries = np.linspace(1.0/365.0, 4.0, 10)
        strikes = np.linspace(0.4, 1.6, 100)
        vols = 0.3*np.ones((expiries.shape[0], strikes.shape[0]))
        vol_grid_param = VolatilityGridParametrization(expiries, strikes, vols)
        self.assertAlmostEqual(0.3, vol_grid_param.calc_implied_vol(1.0, 1.3), delta=1e-7)
        # now add to volatility surface
        refdate = dt.datetime(2021,1,1)
        dummy = DiscountCurve('',refdate=refdate, dates=[refdate, refdate+dt.timedelta(days=10*365)], df=[1.0,1.0])
        fwd = EquityForwardCurve(100.0, funding_curve=dummy, borrow_curve=dummy,div_table=None)
        vol_surface = VolatilitySurface('', refdate, fwd, enums.DayCounterType.Act365Fixed, vol_grid_param)
        vol = vol_surface.calc_implied_vol(refdate+dt.timedelta(days=365), 100.0)
        self.assertAlmostEqual(vol, vol_grid_param.calc_implied_vol(1.0, 1.3), delta=1e-7)

if __name__ == '__main__':
    unittest.main()

