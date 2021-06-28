import numpy as np
from numpy.core.fromnumeric import var
import scipy.interpolate as interpolation
class LocalVol:

    def __init__(self, vol_param, x_strikes: np.array, time_grid: np.array, drift: np.array):
        self._local_variance = LocalVol.compute_local_var(vol_param, x_strikes, time_grid)
        self._variance = interpolation.interp2d(x_strikes, time_grid, self._local_variance)
        self._drift = interpolation.interp1d(time_grid, drift)

    def get_SDE_coeff(self, t, x):
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
        inv_dt = 1.0/time_grid[1:]-time_grid[:-1]

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

if __name__=='__main__':
    import rivapy.marketdata as mktdata
    ssvi = mktdata.VolatilityParametrizationSSVI(expiries=[1.0/365, 30/365, 0.5, 1.0], fwd_atm_vols=[0.25, 0.3, 0.28, 0.25], rho=-0.9, eta=0.5, gamma=0.5)
    x_strikes = np.linspace(0.9, 1.1, 50)
    time_grid = np.linspace(0.0, 1.0, 80)
    LocalVol.compute_local_var(ssvi, x_strikes, time_grid)
