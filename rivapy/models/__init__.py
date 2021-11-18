from typing import Union
import bisect
import numpy as np
from numpy.core.fromnumeric import var
import scipy
import scipy.interpolate as interpolation
import rivapy.numerics.kernel_regression as kernel_regression

def _interpolate_2D(time_grid, strikes, f, x, t):
    t_index = bisect.bisect_left(time_grid, t)
    if t_index == 0:# or t0_index == self._time_grid.shape[0]:
        result = f[0]
    elif t_index == time_grid:
        result = f[-1]
    else:
        dt = time_grid[t_index] - time_grid[t_index-1]
        w1 = (t-time_grid[t_index-1])/dt
        w2 = (time_grid[t_index] - t)/dt
        result = w1*f[t_index] + w2*f[t_index-1]
    return np.interp(x[:,0], strikes, result)
    
class LocalVol:

    def __init__(self, vol_param, x_strikes: np.array, time_grid: np.array):
        self._local_variance = LocalVol.compute_local_var(vol_param, x_strikes, time_grid)
        self._x_strikes = x_strikes
        self._time_grid = time_grid
        self._variance = interpolation.interp2d(time_grid, x_strikes, self._local_variance.T)
                        #interpolation.RectBivariateSpline(time_grid, x_strikes,  
                         #               self._local_variance, bbox=[None, None, None, None], kx=1, ky=1, s=0)
        
    @staticmethod
    def compute_local_var(vol_param, x_strikes: np.array, time_grid: np.array):
        # setup grids 
        eps = 1e-8
        log_x_strikes = np.log(x_strikes)
        iv = np.empty(shape=(time_grid.shape[0], x_strikes.shape[0])) #implied variance grid
        for i in range(time_grid.shape[0]):
            for j in range(x_strikes.shape[0]):
                iv[i,j] = vol_param.calc_implied_vol(time_grid[i], x_strikes[j])
        iv *= iv
        tiv = np.maximum((time_grid*iv.T).T, eps)
        h = log_x_strikes[1:] - log_x_strikes[:-1]
        hm = h[:-1]
        hp = h[1:]
        fd1a = -1.0 / (2 * hm)
        fd1c = 1.0 / (2 * hp)
        fd1b = -(fd1c + fd1a)
        fd2a = 2.0 / (hm*(hm + hp))
        fd2c = 2.0 / (hp*(hm + hp))
        fd2b = -(fd2a + fd2c)

        min_lv = 0.01
        max_lv = 1.5
        inv_dt = 1.0/(time_grid[1:]-time_grid[:-1])

        dyw = fd1a*tiv[:,:-2] + fd1b*tiv[:,1:-1] + fd1c*tiv[:,2:]
        dyyw = fd2a*tiv[:,:-2] + fd2b*tiv[:,1:-1] + fd2c*tiv[:,2:]
        dtw = np.maximum((inv_dt*(tiv[1:,:]-tiv[:-1,:]).T).T,eps)

        p = log_x_strikes[1:-1] / tiv[:,1:-1]
        q = np.maximum(1 - p*dyw + 0.25*(-0.25 - 1.0 / tiv[:,1:-1] + p*p)*dyw*dyw + 0.5*dyyw, eps)
        local_var = np.empty(shape=(time_grid.shape[0], x_strikes.shape[0])) 
        local_var[1:-1,1:-1] = np.minimum(np.maximum(min_lv*min_lv, dtw[:-1,1:-1] / q[1:-1,:]), max_lv*max_lv)
        local_var[:,-1] = local_var[:,-2]
        local_var[:,0] = local_var[:, 1]
        local_var[0,:] = local_var[1,:]
        local_var[-1,:] = local_var[-2,:]
        return local_var


    def apply_mc_step(self, x: np.ndarray, t0: float, t1: float, rnd: np.ndarray, inplace: bool = True):
        """Apply a MC-Euler step for the LV Model for n different paths.

        Args:
            x (np.ndarray): 2-d array containing the start values for the spot and variance. The first column contains the spot, the second the variance values.
            t0 ([type]): [description]
            t1 ([type]): [description]
            rnd ([type]): [description]
        """
        if not inplace:
            x_ = x.copy()
        else:
            x_ = x
        S = x_[:,0]
        lv = self._variance(t0, S).reshape((-1,)) #_interpolate_2D(self._time_grid, self._x_strikes, self._local_variance)
        dt = t1-t0
        sqrt_dt = np.sqrt(dt)
        S *= np.exp(- 0.5*lv*dt + np.sqrt(lv)*rnd[:,0]*sqrt_dt)
        return x_
 
    
class HestonModel:
    def __init__(self, long_run_variance, mean_reversion_speed, vol_of_vol, 
                initial_variance, correlation):
        self._long_run_variance = long_run_variance
        self._mean_reversion_speed = mean_reversion_speed
        self._vol_of_vol = vol_of_vol
        self._initial_variance = initial_variance
        self._correlation = correlation

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
        if isinstance(K, np.ndarray):
            result = np.empty((tau.shape[0], K.shape[0], ))
            for i in range(tau.shape[0]):
                for j in range(K.shape[0]):
                    result[i,j] = self.call_price(s0,v0,K[j], tau[i])
            return result

        def integ_func(xi, s0, v0, K, tau, num):
            ixi = 1j * xi
            if num == 1:
                return (self._characteristic_func(xi - 1j, s0, v0, tau) / (ixi * self._characteristic_func(-1j, s0, v0, tau)) * np.exp(-ixi * np.log(K))).real
            else:
                return (self._characteristic_func(xi, s0, v0, tau) / (ixi) * np.exp(-ixi * np.log(K))).real

        "Simplified form, with only one integration. "
        h = lambda xi: s0 * integ_func(xi, s0, v0, K, tau, 1) - K * integ_func(xi, s0, v0, K, tau, 2)
        res = 0.5 * (s0 - K) + 1/scipy.pi * scipy.integrate.quad_vec(h, 0, 500.)[0]  #vorher 500
        if tau == 0:
            res = (s0-K > 0) * (s0-K)
        return res
    

    def apply_mc_step(self, x: np.ndarray, t0: float, t1: float, rnd: np.ndarray, inplace: bool = True):
        """Apply a MC-Euler step for the Heston Model for n different paths.

        Args:
            x (np.ndarray): 2-d array containing the start values for the spot and variance. The first column contains the spot, the second the variance values.
            t0 ([type]): [description]
            t1 ([type]): [description]
            rnd ([type]): [description]
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
        S *= np.exp(- 0.5*v*dt + np.sqrt(v)*rnd_corr_S*sqrt_dt)
        v += self._mean_reversion_speed*(self._long_run_variance-v)*dt + self._vol_of_vol*np.sqrt(v)*rnd_V*sqrt_dt
        x_[:,1] = np.maximum(v,0)
        return x_
        
class HestonLocalVol:
    def __init__(self, heston):
        self._heston = heston
        #self._local_vol = LocalVol(vol_param, x_strikes, time_grid, drift_stock)
        self._stoch_local_variance = None #np.ones(shape=(time_grid.shape[0], x_strikes.shape[0]))
        self._x_strikes = None
        self._time_grid = None

    def calibrate_MC(self,
                    vol_param, 
                    x_strikes: np.array,
                    time_grid: np.array, 
                    n_sims, 
                    local_var: np.ndarray=None):
        self._stoch_local_variance = HestonLocalVol._calibrate_MC(self._heston, vol_param,
                                x_strikes, time_grid,  n_sims, local_var)
        self._x_strikes = x_strikes
        self._time_grid = time_grid

    def apply_mc_step(self, x: np.ndarray, t0: float, t1: float, rnd: np.ndarray, inplace: bool = True):
        """Apply a MC-Euler step for the Heston Local Vol Model for n different paths.

        Args:
            x (np.ndarray): 2-d array containing the start values for the spot and variance. The first column contains the spot, the second the variance values.
            t0 ([type]): [description]
            t1 ([type]): [description]
            rnd ([type]): [description]
        """
        if not inplace:
            x_ = x.copy()
        else:
            x_ = x
        t0_index = bisect.bisect_left(self._time_grid, t0)
        if t0_index == 0:# or t0_index == self._time_grid.shape[0]:
            slv = self._stoch_local_variance[0]
        elif t0_index == self._time_grid.shape[0]:
            slv = self._stoch_local_variance[-1]
        else:
            dt = self._time_grid[t0_index] - self._time_grid[t0_index-1]
            w1 = (t0-self._time_grid[t0_index-1])/dt
            w2 = (self._time_grid[t0_index] - t0)/dt
            slv = w1*self._stoch_local_variance[t0_index] + w2*self._stoch_local_variance[t0_index-1]
        slv = np.interp(x[:,0], self._x_strikes, slv)
        rnd_S = rnd[:,0]
        rnd_V = rnd[:,1]
        rnd_corr_S = np.sqrt(1.0-self._heston._correlation**2)*rnd_S + self._heston._correlation*rnd_V
        S = x_[:,0]
        v = x_[:,1]
        dt = t1-t0
        sqrt_dt = np.sqrt(dt)
        S *= np.exp(-0.5*v*slv*dt + np.sqrt(v*slv)*rnd_corr_S*sqrt_dt)
        v += self._heston._mean_reversion_speed*(self._heston._long_run_variance-v)*dt + self._heston._vol_of_vol*np.sqrt(v)*rnd_V*sqrt_dt
        x_[:,1] = np.maximum(v,0)
        return x_


    @staticmethod
    def _calibrate_MC(heston: HestonModel,  
                    vol_param, 
                    x_strikes: np.array,
                    time_grid: np.array, 
                    n_sims, 
                    local_var: np.ndarray=None,
                    x0 = 1.0):
        
        def apply_mc_step( x, t0, t1, rnd, stoch_local_var):
            slv = np.interp(x[:,0], x_strikes, stoch_local_var)
            rnd_S = rnd[:,0]
            rnd_V = rnd[:,1]
            rnd_corr_S = np.sqrt(1.0-heston._correlation**2)*rnd_S + heston._correlation*rnd_V
            S = x[:,0]
            v = x[:,1]
            dt = t1-t0
            sqrt_dt = np.sqrt(dt)
            S *= np.exp((0.5*v*slv)*dt + np.sqrt(v*slv)*rnd_corr_S*sqrt_dt)
            v += heston._mean_reversion_speed*(heston._long_run_variance-v)*dt + heston._vol_of_vol*np.sqrt(v)*rnd_V*sqrt_dt
            x[:,1] = np.maximum(v,0)

        if local_var is None:
            local_var = LocalVol.compute_local_var(vol_param, x_strikes, time_grid)
        else:
            if local_var.shape[0] != time_grid.shape[0] or local_var.shape[1] != x_strikes.shape[0]:
                raise Exception('Local variance has not the right dimension.')
        stoch_local_variance = np.empty(local_var.shape)
        stoch_local_variance[0] = local_var[0]/heston._initial_variance
        #now apply explicit euler to get new values for v and S and then apply kernel regression to estimate new local variance
        x = np.empty((n_sims,2))
        x[:,0] = x0
        x[:,1] = heston._initial_variance
        for i in range(1,time_grid.shape[0]):
            rnd = np.random.normal(size=(n_sims,2))
            apply_mc_step(x, time_grid[i-1], time_grid[i], rnd, stoch_local_variance[i-1])
            gamma = ((4.0*np.std(x[:,0])**5)/x.shape[0])**(-1.0/5.0)
            kr = kernel_regression.KernelRegression(gamma = gamma).fit(x[:,0:1],x[:,1])
            stoch_local_variance[i] = local_var[i]/kr.predict(x_strikes.reshape((-1,1)))
        return stoch_local_variance

if __name__=='__main__':
    import rivapy.marketdata as mktdata
    #ssvi = mktdata.VolatilityParametrizationSSVI(expiries=[1.0/365, 30/365, 0.5, 1.0], fwd_atm_vols=[0.25, 0.3, 0.28, 0.25], rho=-0.9, eta=0.5, gamma=0.5)
    #x_strikes = np.linspace(0.9, 1.1, 50)
    #time_grid = np.linspace(0.0, 1.0, 80)
    #LocalVol.compute_local_var(ssvi, x_strikes, time_grid)

    import pickle
    with open('C:/Users/Anwender/development/RIVACON/RiVaPy/notebooks/models/depp.pck', 'rb') as f:
        heston_grid_param = pickle.load(f)
    heston_grid_param._pyvacon_obj = None
    # heston = HestonModel(long_run_variance=0.3**2, 
    #                         mean_reversion_speed=0.5 , 
    #                         vol_of_vol=0.2, 
    #                         initial_variance=0.1**2, 
    #                         correlation = -0.9)
    # x_strikes = np.linspace(0.7, 1.3, 50)
    # time_grid = np.linspace(0.0, 1.0, 80)
    # heston_lv = HestonLocalVol(heston)
    # heston_lv.calibrate_MC(heston_grid_param,  x_strikes=x_strikes, time_grid=time_grid, n_sims=1000)

    n_sims=100_000
    strikes = np.linspace(0.7, 1.3, 50)
    timegrid = np.linspace(0.0,3.0,3*365)
    expiries = timegrid[10::30]

    local_vol = LocalVol(heston_grid_param, np.linspace(0.7,1.3, 120), timegrid)
    simulated_values = np.empty((n_sims, 1))
    simulated_values[:,0] = 1.0

    paths = np.empty((timegrid.shape[0], n_sims, 1))
    paths[0] = simulated_values

    for i in range(1,timegrid.shape[0]):
        rnd = np.random.normal(size=(n_sims,1))
        paths[i] = local_vol.apply_mc_step(simulated_values, timegrid[i-1], timegrid[i], rnd, inplace=True)
        depp = local_vol.apply_mc_step(simulated_values, timegrid[i-1], timegrid[i], rnd, inplace=True)
        if isinstance(depp,tuple):
            break
    call_prices_local_vol = np.empty((expiries.shape[0], strikes.shape[0]))
    p = paths[10::30]

    for i in range(expiries.shape[0]):
        for k in range(strikes.shape[0]):
            call_prices_local_vol[i,k] = np.mean(np.maximum(p[i][:,0]-strikes[k],0.0))