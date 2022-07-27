from typing import Union, Callable
import numpy as np
class OrnsteinUhlenbeck:
    def _eval_grid(f, timegrid):
        try:
            return f(timegrid)
        except:
            result = np.full(timegrid.shape, f)
            return result

    def __init__(self, speed_of_mean_reversion: Union[float, Callable], 
                    volatility: Union[float, Callable]):
        self.speed_of_mean_reversion = speed_of_mean_reversion
        self.volatility = volatility
        self._timegrid = None

    def _set_timegrid(self, timegrid):
        self._timegrid = np.copy(timegrid)
        self._delta_t = self._timegrid[1:]-self._timegrid[:-1]
        self._sqrt_delta_t = np.sqrt(self._delta_t)

        self._speed_of_mean_reversion = OrnsteinUhlenbeck._eval_grid(self.speed_of_mean_reversion, timegrid)
        self._volatility = OrnsteinUhlenbeck._eval_grid(self.volatility, timegrid)
        
    def simulate(self, timegrid, start_value, rnd):
        self._set_timegrid(timegrid)
        result = np.empty((self._timegrid.shape[0], rnd.shape[0]))
        result[0,:] = start_value 
        for i in range(self._timegrid.shape[0]-1):
            result[i+1,:] = (1.0  - self._speed_of_mean_reversion[i]*self._delta_t[i])*result[i,:] + self._volatility[i]*self._sqrt_delta_t[i]*rnd[:,i]    
        return result


    def apply_mc_step(self, x: np.ndarray, 
                        t0: float, t1: float, 
                        rnd: np.ndarray, 
                        inplace: bool = True, 
                        slv: np.ndarray= None):
        if not inplace:
            x_ = x.copy()
        else:
            x_ = x
        dt = t1-t0
        sqrt_dt = np.sqrt(dt)
        try:
            mu = self.speed_of_mean_reversion(t0)
        except:
            mu = self.speed_of_mean_reversion
        try:
            sigma = self.volatility(t0)
        except:
            sigma = self.volatility
        x_ = (1.0  - mu*dt)*x + sigma*sqrt_dt*rnd   
        return x_
        
    def conditional_probability_density(self, X_delta_t, delta_t, X0, 
                                        volatility=None, 
                                        speed_of_mean_reversion=None, 
                                        mean_reversion_level = 0):
        if volatility is None:
            volatility = self.volatility
        if speed_of_mean_reversion is None:
            speed_of_mean_reversion = self.speed_of_mean_reversion
        volatility_2_ = volatility**2*(1.0-np.exp(-2.0*speed_of_mean_reversion*delta_t))/(2.0*speed_of_mean_reversion)
        result = 1.0/(2.0*np.pi*volatility_2_)*np.exp(
                        -(X_delta_t-X0*np.exp(-speed_of_mean_reversion*delta_t)
                        -mean_reversion_level*(1.0-np.exp(-speed_of_mean_reversion*delta_t)))
                    /(2.0*volatility_2_))
        return result

    