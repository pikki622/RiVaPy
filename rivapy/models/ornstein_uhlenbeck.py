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
                    volatility: Union[float, Callable], 
                    mean_reversion_level: Union[float, Callable] = 0):
        self.speed_of_mean_reversion = speed_of_mean_reversion
        self.mean_reversion_level = mean_reversion_level,
        self.volatility = volatility
        self._timegrid = None

    def _set_timegrid(self, timegrid):
        self._timegrid = np.copy(timegrid)
        self._delta_t = self._timegrid[1:]-self._timegrid[:-1]
        self._sqrt_delta_t = np.sqrt(self._delta_t)

        self._speed_of_mean_reversion = OrnsteinUhlenbeck._eval_grid(self.speed_of_mean_reversion, timegrid)
        self._volatility = OrnsteinUhlenbeck._eval_grid(self.volatility, timegrid)
        self._mean_reversion_level = OrnsteinUhlenbeck._eval_grid(self.mean_reversion_level, timegrid)
        
    def simulate(self, timegrid, start_value, rnd):
        self._set_timegrid(timegrid)
        result = np.empty((self._timegrid.shape[0], rnd.shape[1]))
        result[0,:] = start_value

        for i in range(self._timegrid.shape[0]-1):
            result[i+1,:] = (result[i, :] * np.exp(-self._speed_of_mean_reversion[i]*self._delta_t[i])
                        + self._mean_reversion_level[i]* (1 - np.exp(-self._speed_of_mean_reversion[i]*self._delta_t[i])) 
                        + self._volatility[i]* np.sqrt((1 - np.exp(-2*self._speed_of_mean_reversion[i]*self._delta_t[i])) / (2*self._speed_of_mean_reversion[i])) * rnd[i,:]
                        )
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
                                        mean_reversion_level = None):
        if volatility is None:
            volatility = self.volatility
        if speed_of_mean_reversion is None:
            speed_of_mean_reversion = self.speed_of_mean_reversion
        if mean_reversion_level is None:
            mean_reversion_level = self.mean_reversion_level
        volatility_2_ = volatility**2*(1.0-np.exp(-2.0*speed_of_mean_reversion*delta_t))/(2.0*speed_of_mean_reversion)
        result = 1.0/(2.0*np.pi*volatility_2_)*np.exp(
                        -(X_delta_t-X0*np.exp(-speed_of_mean_reversion*delta_t)
                        -mean_reversion_level*(1.0-np.exp(-speed_of_mean_reversion*delta_t)))
                    /(2.0*volatility_2_))
        return result

    def calibrate(self, data: np.ndarray, dt: float, method: str='maximum_likelihood'):
        """Calibrate the Ornstein Uhlenbeck model with constant parameters to the given data.

        Args:
            data (np.ndarray): Array of values the model is fitted to (uniform timegrid is assumed).
            dt (float): Time step size between two datapoints from data (uniform timegrid is assumed.
            method (str, optional): Determines if maximum likelihood ('maximum_likelihood') or minimum least square ('minimum_least_square') is used for calibration. Defaults to 'maximum_likelihood'.
        """
        Sx  = (data[:-1]).sum()
        Sy  = (data[1:]).sum()
        Sxx = (data[:-1]**2).sum()
        Syy = (data[1:]**2).sum()
        Sxy = (data[:-1] * data[1:]).sum()
        n = data[:-1].shape[0]
        if method == 'maximum_likelihood':        
            mu = (Sy*Sxx - Sx*Sxy) / (n*(Sxx-Sxy) - (Sx**2-Sx*Sy))
            if ((Sxy-mu*(Sx+Sy-n*mu))/(Sxx-2*mu*Sx+n*mu**2)) <= 0:
                raise Exception('Calibration failed.')
            speed_mr = -1/dt * np.log((Sxy-mu*(Sx+Sy-n*mu))/(Sxx-2*mu*Sx+n*mu**2)) 
            alpha = np.exp(-speed_mr*dt)
            sigma_2 = 1/n * (Syy-2*alpha*Sxy+alpha**2*Sxx-2*mu*(1-alpha)*(Sy-alpha*Sx)+n*mu**2*(1-alpha)**2)
            sigma = np.sqrt(sigma_2 * (2*speed_mr) / (1-alpha**2))          
        elif method == 'minimum_least_square':
            a = (n*Sxy - Sx*Sy) / (n*Sxx - Sx**2)
            b = (Sy - a*Sx) / n
            sd = np.sqrt((n*Syy-Sy**2 - a*(n*Sxy-Sx*Sy)) / (n*(n-2)))
            speed_mr = -np.log(a) / dt
            mu = b / (1-a)
            sigma = sd * np.sqrt((-2*np.log(a)) / (dt*(1-a**2)))
        else:
            raise ValueError('Fitting method not defined ('+method+')')        
        self.speed_of_mean_reversion = speed_mr
        self.volatility = sigma
        self.mean_reversion_level = mu
#