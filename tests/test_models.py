import unittest
import numpy as np
import datetime as dt

import rivapy.models as models
import rivapy.marketdata as mktdata
import rivapy.pricing.analytics as analytics
import rivapy.enums as enums

class LocalVolModelTest(unittest.TestCase):
	
	def test_local_vol_mc_with_ssvi(self):
		"""Simple test where call price from Local Vol MC simulation is compared against BS price
		"""
		ssvi = mktdata.VolatilityParametrizationSSVI(expiries=[1.0/365, 30/365, 0.5, 1.0], fwd_atm_vols=[0.25, 0.3, 0.28, 0.25], rho=-0.9, eta=0.5, gamma=0.5)
		x_strikes = np.linspace(0.5,1.5,100)
		time_grid = np.linspace(0.0,1.0,100)
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
		self.assertAlmostEqual(call_price, call_price_ref, places=2)
	
	def test_local_vol_with_flat_input_vol(self):
		"""Simple test where the flat input volatility is compared with the local volatility calculated with call prices from black scholes
		"""
		x_strikes = np.linspace(0.5,1.5,100)
		time_grid = np.linspace(0.0,1.0,365)
		input_vol = 0.3
		call_prices = np.array([[analytics.compute_european_price_Buehler(strike=k, maturity=t, volatility=input_vol) for k in x_strikes] for t in time_grid])
		lv_model = models.LocalVol(vol_param=None, x_strikes=x_strikes, time_grid=time_grid, call_param=call_prices)
		var = lv_model.compute_local_var(vol_param=None, call_param = call_prices, x_strikes = x_strikes, time_grid = time_grid)
		vol = np.sqrt(np.abs(var))

		for i,t in enumerate(time_grid):
			if t < 1/365: 
				continue
			elif t < 2/365: 
				factor = 1.0
			else: 
				factor = 2.0
			range_low = np.exp(-factor * input_vol * np.sqrt(t))
			range_up = np.exp(factor * input_vol * np.sqrt(t))
			val_in_range = vol[i,(x_strikes>range_low)&(x_strikes<range_up)]

			self.assertAlmostEqual(input_vol, np.mean(val_in_range), places=2)
	
	def test_compare_local_var_implied_and_call(self):
		"""Simple test where the local volatility of a volatility surface is compared with the local volatility of the corresponding call price surface
		"""
		x_strikes = np.linspace(0.5,1.5,100)
		time_grid = np.linspace(0.0,1.0,100)
		
		ssvi = mktdata.VolatilityParametrizationSSVI(expiries=[1.0/365, 30/365, 0.5, 1.0], fwd_atm_vols=[0.25, 0.3, 0.28, 0.25], rho=-0.9, eta=0.5, gamma=0.5)
		var_vol = np.sqrt(models.LocalVol.compute_local_var(ssvi, x_strikes, time_grid))
		
		input_vol = np.array([[ssvi.calc_implied_vol(ttm=t, strike=k) for k in x_strikes] for t in time_grid])		
		call_prices = np.array([[analytics.compute_european_price_Buehler(strike=k, maturity=t, volatility=input_vol[i,j]) for j,k in enumerate(x_strikes)] for  i,t in enumerate(time_grid)])

		lv_model = models.LocalVol(vol_param=None, x_strikes=x_strikes, time_grid=time_grid, call_param=call_prices)
		var_call = np.sqrt(lv_model.compute_local_var(vol_param=None, call_param = call_prices, x_strikes = x_strikes, time_grid = time_grid))
				
		for i,t in enumerate(time_grid):
			if t < 1/365: 
				continue
			range_low = np.exp(-input_vol[i,:] * np.sqrt(t))
			range_up = np.exp(input_vol[i,:] * np.sqrt(t))
			var_vol_range = var_vol[i,(x_strikes>range_low)&(x_strikes<range_up)]
			var_call_range = var_call[i,(x_strikes>range_low)&(x_strikes<range_up)]
			error = np.mean(var_vol_range) - np.mean(var_call_range)
			self.assertLess(error, 2E-2)

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
		
class HestonLocalVolModelTest(unittest.TestCase):
	@staticmethod
	def calc_imlied_vol_grid(expiries, strikes, call_prices):
		vols = np.zeros((expiries.shape[0],strikes.shape[0]))
		for i in range(expiries.shape[0]):
			for j in range(strikes.shape[0]):
				try:
					vols[i,j] = analytics.compute_implied_vol_Buehler(strikes[j], maturity=expiries[i], 
																	price=call_prices[i, j], min_vol=0.01)
				except:
					pass
			
		for i in range(expiries.shape[0]):
			extrapolation = False
			for j in range(int(strikes.shape[0]/2),strikes.shape[0]):
				if vols[i,j] <1e-6:
					vols[i,j] = vols[i,j-1]
					extrapolation = True
			for j in range(23,-1,-1):
				if vols[i,j] <1e-6:
					vols[i,j] = vols[i,j+1]
					extrapolation = True
			if extrapolation:
				print('Extrapolation necessary for expiry ' + str(i))
		return vols

	def test_simple(self):
		"""Simple test: The given implied volatility surface equals the heston surface->stoch local variance must equal 1
		"""
		heston = models.HestonModel(long_run_variance=0.3**2, 
                            mean_reversion_speed=0.5, 
                            vol_of_vol=0.01, 
                            initial_variance=0.3**2, 
                            correlation = -0.95)
		x_strikes = np.linspace(0.5,1.5, num=240)
		time_grid = np.linspace(1.0/365.0,1.0, num=120) 
		call_prices = heston.call_price(1.0, heston._initial_variance, x_strikes, time_grid)
		vols_heston = HestonLocalVolModelTest.calc_imlied_vol_grid(time_grid, x_strikes, call_prices)   
		heston_grid_param = mktdata.VolatilityGridParametrization(time_grid, x_strikes, vols_heston)
		#dc = mktdata.curves.DiscountCurve('dc', dt.datetime(2021,1,1), [dt.datetime(2021,1,1), dt.datetime(2021,2,1)],[1.0, 1.0])
		# heston_surface = mktdata.VolatilitySurface('heston_surface', dt.datetime(2021,1,1), 
        #                   mktdata.EquityForwardCurve(1.0,dc,dc,div_table=None), 
        #                  enums.DayCounterType.Act365Fixed, heston_grid_param)
		heston_lv = models.HestonLocalVol(heston)
		heston_lv.calibrate_MC(heston_grid_param, x_strikes, time_grid, n_sims=10000)
		self.assertAlmostEqual(heston_lv._stoch_local_variance, 1.0, 3)

if __name__ == '__main__':
    unittest.main()