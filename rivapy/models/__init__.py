import numpy as np
from numpy.core.fromnumeric import var
import scipy.interpolate as interpolation
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
        self._drift_stock =drift_stock

    def apply_mc_step(self, x, t0, t1, rnd):
        rnd_S = rnd[:,0]
        rnd_V = rnd[:,1]
        rnd_corr_S = np.sqrt(1.0-self._correlation**2)*rnd_S + self._correlation*rnd_V
        S = x[:,0]
        v = x[:,1]
        dt = t1-t0
        sqrt_dt = np.sqrt(dt)
        x = (self._drift_stock(t0) - 0.5*v)*(t1-t0) + np.sqrt(v)*rnd_corr_S*np.sqrt(t1-t0)
        S *= np.exp(x)
        #print(self._long_run_variance-v, dt)
        v += self._mean_reversion_speed*(self._long_run_variance-v)*dt + self._vol_of_vol*np.sqrt(v)*rnd_V*sqrt_dt
        v = np.max(v,0)
class HestonLocalVol:
    def __init__(self, heston,  vol_param, x_strikes: np.array, time_grid: np.array, drift_stock: np.array):
        self._heston = heston
        self._local_vol = LocalVol(vol_param, x_strikes, time_grid, drift_stock)
        self._stoch_local_variance = np.empty(shape=(time_grid.shape[0], x_strikes.shape[0]))

    
    @staticmethod
    def _kernel_regression(x, y, bandwidth):
        def eval(xx):
            k = np.exp(-(xx-x)**2/bandwidth)
            return np.sum(y*k)/np.sum(k)
                
    @staticmethod
    def calibrate_MC(heston: HestonModel,  vol_param, x_strikes: np.array,
                    time_grid: np.array, drift_stock: np.array, 
                    n_sims, 
                    local_var: np.ndarray=None,
                    x0 = 1.0):
        
        def apply_mc_step( x, t0, t1, rnd, stoch_local_var):
            slv = interpolation.interp1d(time_grid, stoch_local_var)(x[0])
            rnd_S = rnd[:,0]
            rnd_V = rnd[:,1]
            rnd_corr_S = np.sqrt(1.0-heston._correlation**2)*rnd_S + heston._correlation*rnd_V
            S = x[:,0]
            v = x[:,1]
            dt = t1-t0
            sqrt_dt = np.sqrt(dt)
            x = (drift_stock(t0) - 0.5*v*slv)*(t1-t0) + np.sqrt(v*slv)*rnd_corr_S*np.sqrt(t1-t0)
            S *= np.exp(x)
            #print(self._long_run_variance-v, dt)
            v += heston._mean_reversion_speed*(heston._long_run_variance-v)*dt + heston._vol_of_vol*np.sqrt(v)*rnd_V*sqrt_dt
            v = np.max(v,0)

        if local_var is None:
            local_var = LocalVol.compute_local_var(vol_param, x_strikes, time_grid)
        else:
            if local_var.shape[0] != time_grid.shape[0] or local_var.shape[1] != x_strikes.shape[0]:
                raise Exception('Local variance has not the right dimension.')
        stoch_local_variance = np.empty(local_var.shape)
        stoch_local_variance[0] = local_var/heston._initial_variance
        #now apply explicit euler to get new values for v and S and then apply kernel regression to estimate new local variance
        x = np.empty((n_sims,2))
        x[:,0] = x0
        x[:,1] = heston._initial_variance
        for i in range(1,time_grid.shape[0]):
            rnd = np.random.normal(size=(n_sims,2))
            apply_mc_step(x, time_grid[i-1], time_grid[i], rnd, stoch_local_variance[i-1])
            regression = HestonLocalVol._kernel_regression(x[:,0],x[:,1], bandwidth=1.0)
            for j in range(x_strikes.shape[0]):
                stoch_local_variance[i,j] = regression(x_strikes[j])
                
    @staticmethod    
    def apply_mc_step(ln_x0, x0, t0, t1, rnd, model):
        drift, vol = model.get_SDE_coeff(t0, x0)
        #print(vol)
        #ln_x0 += drift*(t1-t0) + vol*np.sqrt(t1-t0)*rnd
        return ln_x0 + drift*(t1-t0) + vol*np.sqrt(t1-t0)*rnd

if __name__=='__main__':
    import rivapy.marketdata as mktdata
    ssvi = mktdata.VolatilityParametrizationSSVI(expiries=[1.0/365, 30/365, 0.5, 1.0], fwd_atm_vols=[0.25, 0.3, 0.28, 0.25], rho=-0.9, eta=0.5, gamma=0.5)
    x_strikes = np.linspace(0.9, 1.1, 50)
    time_grid = np.linspace(0.0, 1.0, 80)
    LocalVol.compute_local_var(ssvi, x_strikes, time_grid)
