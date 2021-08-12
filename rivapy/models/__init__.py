import numpy as np
from numpy.core.fromnumeric import var
import scipy.interpolate as interpolation
import rivapy.numerics.kernel_regression as kernel_regression
class LocalVol:

    def __init__(self, vol_param, x_strikes: np.array, time_grid: np.array, drift: np.array):
        self._local_variance = LocalVol.compute_local_var(vol_param, x_strikes, time_grid)
        self._x_strikes = x_strikes
        self._time_grid = time_grid
        self._variance = interpolation.interp2d(x_strikes, time_grid, self._local_variance)
        self._drift = interpolation.interp1d(time_grid, drift)

    def _get_SDE_coeff(self, t, x):
        variance = self._variance(x, t)
        drift = self._drift(t) - 0.5*variance
        vol = np.sqrt(variance)
        return drift, vol

    @staticmethod
    def compute_local_var(vol_param, x_strikes: np.array, time_grid: np.array):
        # setup grids 
        log_x_strikes = np.log(x_strikes)
        iv = np.empty(shape=(time_grid.shape[0], x_strikes.shape[0])) #implied variance grid
        for i in range(time_grid.shape[0]):
            for j in range(x_strikes.shape[0]):
                iv[i,j] = vol_param.calc_implied_vol(time_grid[i], x_strikes[j])
        iv *= iv
        tiv = (time_grid*iv.T).T
        h = log_x_strikes[1:] - log_x_strikes[:-1]
        hm = h[:-1]
        hp = h[1:]
        fd1a = -1.0 / (2 * hm)
        fd1c = 1.0 / (2 * hp)
        fd1b = -(fd1c + fd1a)
        fd2a = 2.0 / (hm*(hm + hp))
        fd2c = 2.0 / (hp*(hm + hp))
        fd2b = -(fd2a + fd2c)

        eps = 1e-8
        min_lv = 0.001
        max_lv = 2.5
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


    def apply_mc_step(self, x, t0, t1, rnd):
        drift, vol = self._get_SDE_coeff(t0, x)
        x*=np.exp(drift*(t1-t0) + vol*np.sqrt(t1-t0)*rnd)

class HestonModel:
    def __init__(self, long_run_variance, mean_reversion_speed, vol_of_vol, 
                initial_variance, correlation, drift_stock):
        self._long_run_variance = long_run_variance
        self._mean_reversion_speed = mean_reversion_speed
        self._vol_of_vol = vol_of_vol
        self._initial_variance = initial_variance
        self._correlation = correlation
        self._drift_stock = drift_stock

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
        rnd_S = rnd[:,0]
        rnd_V = rnd[:,1]
        rnd_corr_S = np.sqrt(1.0-self._correlation**2)*rnd_S + self._correlation*rnd_V
        S = x_[:,0]
        v = x_[:,1]
        dt = t1-t0
        sqrt_dt = np.sqrt(dt)
        log_s = (self._drift_stock(t0) - 0.5*v)*(t1-t0) + np.sqrt(v)*rnd_corr_S*np.sqrt(t1-t0)
        S *= np.exp(log_s)
        v += self._mean_reversion_speed*(self._long_run_variance-v)*dt + self._vol_of_vol*np.sqrt(v)*rnd_V*sqrt_dt
        v = np.max(v,0)
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
                    local_var: np.ndarray=None,
                    x0 = 1.0):
        self._stoch_local_variance = HestonLocalVol._calibrate_MC(self._heston, vol_param,
                                x_strikes, time_grid,  n_sims, local_var)
        self._x_strikes = x_strikes
        self._time_grid = time_grid


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
            x = (0.5*v*slv)*dt + np.sqrt(v*slv)*rnd_corr_S*sqrt_dt
            S *= np.exp(x)
            v += heston._mean_reversion_speed*(heston._long_run_variance-v)*dt + heston._vol_of_vol*np.sqrt(v)*rnd_V*sqrt_dt
            v = np.max(v,0)

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
            kr = kernel_regression.KernelRegression().fit(x[:,0:1],x[:,1])
            stoch_local_variance[i] = kr.predict(x_strikes.reshape((-1,1)))
        return stoch_local_variance

    @staticmethod    
    def apply_mc_step(ln_x0, x0, t0, t1, rnd, model):
        drift, vol = model.get_SDE_coeff(t0, x0)
        #print(vol)
        #ln_x0 += drift*(t1-t0) + vol*np.sqrt(t1-t0)*rnd
        return ln_x0 + drift*(t1-t0) + vol*np.sqrt(t1-t0)*rnd

if __name__=='__main__':
    import rivapy.marketdata as mktdata
    #ssvi = mktdata.VolatilityParametrizationSSVI(expiries=[1.0/365, 30/365, 0.5, 1.0], fwd_atm_vols=[0.25, 0.3, 0.28, 0.25], rho=-0.9, eta=0.5, gamma=0.5)
    #x_strikes = np.linspace(0.9, 1.1, 50)
    #time_grid = np.linspace(0.0, 1.0, 80)
    #LocalVol.compute_local_var(ssvi, x_strikes, time_grid)

    ssvi = mktdata.VolatilityParametrizationSSVI(expiries=[1.0/365, 30/365, 0.5, 1.0], fwd_atm_vols=[0.25, 0.3, 0.28, 0.25], rho=-0.9, eta=0.5, gamma=0.5)
    x_strikes = np.linspace(0.7, 1.3, 50)
    time_grid = np.linspace(0.0, 1.0, 80)
    heston = HestonModel(long_run_variance=0.2**2, 
                            mean_reversion_speed=2.0, 
                            vol_of_vol=0.1, 
                            initial_variance=0.1**2, 
                            correlation = -0.9, 
                            drift_stock=lambda x: 0)
    heston_lv = HestonLocalVol.calibrate_MC(heston, ssvi,  x_strikes=x_strikes, 
                                                time_grid=time_grid, drift_stock=np.zeros(time_grid.shape),
                                                n_sims=1000) 

    import matplotlib.pyplot as plt
    time_index = 0
    plt.plot(x_strikes, heston_lv._stoch_local_variance[time_index])
    plt.plot(x_strikes, heston_lv._local_vol._local_variance[time_index])
    plt.show()
