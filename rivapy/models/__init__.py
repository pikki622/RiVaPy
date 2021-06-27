import numpy as np

class LocalVol:
    @staticmethod
    def compute_local_var(vol_param, x_strikes: np.array, time_grid: np.array):
        # setup grids 
        log_x_strikes = np.log(x_strikes)
        iv = np.empty(shape=(time_grid.shape[0], x_strikes.shape[0])) #implied variance grid
        for i in range(time_grid.shape[0]):
            for j in range(x_strikes.shape[0]):
                iv[i,j] = vol_param.calc_implied_vol(time_grid[i], x_strikes[j])
        iv *= iv
        tiv = iv*time_grid
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

        p = log_x_strikes[1:-1] / tiv[1:-1]
        q = np.maximum(1 - p*dyw + 0.25*(-0.25 - 1.0 / tiv[:,1:-1] + p*p)*dyw*dyw + 0.5*dyyw, eps)
        local_var = np.empty(shape=(time_grid.shape[0], x_strikes.shape[0])) 
        local_var[1:-1,1:-1] = np.minimum(np.maximum(min_lv*min_lv, dtw[:-1] / q[1:-1,:]), max_lv*max_lv)
        local_var[:,-1] = local_var[:,:-2]
        local_var[:,0] = local_var[:, :1]
        local_var[0,:] = local_var[1,:]
        local_var[-1,:] = local_var[-2,:]
        return local_var
