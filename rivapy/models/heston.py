from typing import Union
import numpy as np
import scipy 


class HestonModel:
	def __init__(self, long_run_variance, mean_reversion_speed, vol_of_vol, 
				initial_variance, correlation):
		self._long_run_variance = long_run_variance
		self._mean_reversion_speed = mean_reversion_speed
		self._vol_of_vol = vol_of_vol
		self._initial_variance = initial_variance
		self._correlation = correlation

	def feller_condition(self):
		"""Return True if the model parameter fulfill the Feller condition
		..:

		Returns:
			bool: True->Feller condition is fullfilled
		"""
		return 2*self._mean_reversion_speed*self._long_run_variance>self._vol_of_vol > 0

	def get_initial_value(self)->np.ndarray:
		"""Return the initial value (x0, v0)

		Returns:
			np.ndarray: Initial value.
		"""
		return np.array([1.0, self._initial_variance])

	def _characteristic_func(self, xi, s0, v0, tau):
		"""Characteristic function needed internally to compute call prices with analytic formula.
		"""
		ixi = 1j * xi
		d = np.sqrt((self._mean_reversion_speed - ixi * self._correlation * self._vol_of_vol)**2
					   + self._vol_of_vol**2 * (ixi + xi**2))
		g = (self._mean_reversion_speed - ixi * self._correlation * self._vol_of_vol - d) / (self._mean_reversion_speed - ixi * self._correlation * self._vol_of_vol + d)
		ee = np.exp(-d * tau)
		C = self._mean_reversion_speed * self._long_run_variance / self._vol_of_vol**2 * (
			(self._mean_reversion_speed - ixi * self._correlation * self._vol_of_vol - d) * tau - 2. * np.log((1 - g * ee) / (1 - g))
		)
		D = (self._mean_reversion_speed - ixi * self._correlation * self._vol_of_vol - d) / self._vol_of_vol**2 * (
			(1 - ee) / (1 - g * ee)
		)
		return np.exp(C + D*v0 + ixi * np.log(s0))
	
	def call_price(self, s0: float, v0: float, K: Union[np.ndarray, float], tau: Union[np.ndarray, float])->Union[np.ndarray, float]:
		"""Computes a call price for the Heston model via integration over characteristic function.

		Args:
			s0 (float): current spot
			v0 (float): current variance
			K (float): strike
			tau (float): time to maturity
		"""
		if isinstance(tau, np.ndarray):
			result = np.empty((tau.shape[0], K.shape[0], ))
			for i in range(tau.shape[0]):
				#for j in range(K.shape[0]):
					#result[i,j] = self.call_price(s0,v0,K[j], tau[i])
				result[i,:] = self.call_price(s0,v0,K, tau[i])
			return result

		def integ_func(xi, s0, v0, K, tau, num):
			ixi = 1j * xi
			if num == 1:
				return (self._characteristic_func(xi - 1j, s0, v0, tau) / (ixi * self._characteristic_func(-1j, s0, v0, tau)) * np.exp(-ixi * np.log(K))).real
			else:
				return (self._characteristic_func(xi, s0, v0, tau) / (ixi) * np.exp(-ixi * np.log(K))).real

		if tau < 1e-3:
			res = (s0-K > 0) * (s0-K)
		else:
			"Simplified form, with only one integration. "
			h = lambda xi: s0 * integ_func(xi, s0, v0, K, tau, 1) - K * integ_func(xi, s0, v0, K, tau, 2)
			res = 0.5 * (s0 - K) + 1/scipy.pi * scipy.integrate.quad_vec(h, 0, 500.)[0]  #vorher 500
		return res
	

	def apply_mc_step(self, x: np.ndarray, t0: float, t1: float, rnd: np.ndarray, inplace: bool = True, slv: np.ndarray= None):
		"""Apply a MC-Euler step for the Heston Model for n different paths.

		Args:
			x (np.ndarray): 2-d array containing the start values for the spot and variance. The first column contains the spot, the second the variance values.
			t0 ([type]): [description]
			t1 ([type]): [description]
			rnd ([type]): [description]
			slv (np.ndarray): Stochastic local variance (for each path) to be multiplied with the heston variance. This is used by the StochasticVolatilityModel.
		"""
		if not inplace:
			x_ = x.copy()
		else:
			x_ = x
		rnd_corr_S = np.sqrt(1.0-self._correlation**2)*rnd[:,0] + self._correlation*rnd[:,1]
		rnd_V = rnd[:,1]
		S = x_[:,0]
		v = x_[:,1]
		dt = t1-t0
		sqrt_dt = np.sqrt(dt)
		if slv is None:
			slv=1.0
		S *= np.exp(- 0.5*v*slv*dt + np.sqrt(v*slv)*rnd_corr_S*sqrt_dt)
		v += self._mean_reversion_speed*(self._long_run_variance-v)*dt + self._vol_of_vol*np.sqrt(v)*rnd_V*sqrt_dt
		x_[:,1] = np.maximum(v,0)
		return x_
