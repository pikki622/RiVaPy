import unittest
import numpy as np

import rivapy.models as models

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