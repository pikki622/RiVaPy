
import numpy as np
from numpy.core.fromnumeric import var

from rivapy.models.local_vol import LocalVol
from rivapy.models.heston import HestonModel
from rivapy.models.stoch_local_vol import StochasticLocalVol
from rivapy.models.scott_chesney import ScottChesneyModel
from rivapy.models.ornstein_uhlenbeck import OrnsteinUhlenbeck
from rivapy.models.residual_demand_model import ResidualDemandModel,  WindPowerModel, WindPowerForecastModel, MultiRegionWindForecastModel, SolarPowerModel, SupplyFunction, LoadModel


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