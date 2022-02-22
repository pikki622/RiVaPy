import unittest
import numpy as np

import rivapy.models as models
import rivapy.marketdata as mktdata
import rivapy.pricing.analytics as analytics

class LocalVolModelTest(unittest.TestCase):
	def test_local_vol(self):
		"""Simple test where call price from Local Vol MC simulation is compared against BS price
		"""
		ssvi = mktdata.VolatilityParametrizationSSVI(expiries=[1.0/365, 30/365, 0.5, 1.0], fwd_atm_vols=[0.25, 0.3, 0.28, 0.25], rho=-0.9, eta=0.5, gamma=0.5)
		x_strikes = np.linspace(0.5,1.5)
		time_grid = np.linspace(0.0,1.0)
		lv = models.LocalVol(ssvi, x_strikes, time_grid)
		n_sims = 100_000
		S = np.ones((n_sims,1))
		np.random.seed(42)
		for i in range(1,time_grid.shape[0]):
			rnd = np.random.normal(size=(n_sims, 1))
			lv.apply_mc_step(S,time_grid[i-1], time_grid[i],rnd, inplace=True)
		strike = 1.0
		call_price = np.mean(np.maximum(S-strike, 0))
		call_price_ref = analytics.compute_european_price_Buehler(strike = strike, maturity=1.0, volatility=ssvi.calc_implied_vol(1.0,strike))
		self.assertAlmostEqual(call_price, call_price_ref, delta=0.1)

class HestonModelTest(unittest.TestCase):
	
	def test_callprice_formula(self):
		"""Test analytic call price formula by comparing with MC simulated values
		"""
		heston = models.HestonModel(long_run_variance=0.3**2, 
                             mean_reversion_speed=0.5 , 
                             vol_of_vol=0.2, 
                             initial_variance=0.1**2, 
                             correlation = -0.9)

		n_sims = 40_000
		np.random.seed(42)
		strikes = np.array([0.9,1.0,1.1])
		timegrid = np.linspace(0.0,1.0,365)
		
		simulated_values = np.empty((n_sims, 2))
		simulated_values[:,0] = 1.0
		simulated_values[:,1] = heston._initial_variance

		
		cp_anayltic = heston.call_price(1.0, heston._initial_variance, K = strikes, tau = 1.0)

		for i in range(1, timegrid.shape[0]):
			rnd = np.random.normal(size=(n_sims,2))
			heston.apply_mc_step(simulated_values, timegrid[i-1], timegrid[i], rnd, inplace=True)

		for i in range(strikes.shape[0]):
			cp_mc = np.mean(np.maximum(simulated_values[:,0]-strikes[i], 0.0))
			self.assertAlmostEqual(cp_anayltic[i], cp_mc, delta=1e-3)
		


if __name__ == '__main__':
    unittest.main()