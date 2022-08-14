import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from typing import Union, Callable
from  rivapy.tools.datetime_grid import DateTimeGrid

def _logit(x):
    return np.log(x/(1-x))

def _inv_logit(x):
    return 1.0/(1+np.exp(-x))


class SolarProfile:
    def __init__(self, profile:Callable):
        self._profile = profile

    def get_profile(self, timegrid: DateTimeGrid):
        result = np.empty(timegrid.timegrid.shape[0])
        for i in range(result.shape[0]):
            result[i] = self._profile(timegrid.dates[i])
        return result

class SolarPowerModel:
    def _eval_grid(f, timegrid):
        try:
            return f(timegrid)
        except:
            result = np.full(timegrid.shape, f)
            return result

    def __init__(self, daily_maximum_process,
                    profile,
                    mean_level,
                    name:str = 'Solar_Germany'):
        self.name = name
        self._daily_maximum_process = daily_maximum_process
        self._profile = profile
        self.mean_level = mean_level

  
    def simulate(self, timegrid: DateTimeGrid, start_value: float, rnd):
        # daily timegrid for daily maximum simulation
        tg_ = timegrid.get_daily_subgrid()
        ml = SolarPowerModel._eval_grid(self.mean_level, tg_)
        start_value_ = _logit(start_value) - ml[0]
        daily_maximum = self._daily_maximum_process.simulate(tg_.timegrid, start_value_, rnd)
        profile = self._profile.get_profile(timegrid)
        result = np.empty((timegrid.shape[0], rnd.shape[0]))
        day = 0
        d = tg_.dates[0].date()
        for i in range(timegrid.timegrid.shape[0]):
            if d != timegrid.dates[i].date():
                day += 1
                d = timegrid.dates[i].date()
            result[i,:] = _inv_logit(daily_maximum[day,:] + ml[day])* profile[i] 
        return result

class WindPowerModel:
    def _eval_grid(f, timegrid):
        try:
            return f(timegrid)
        except:
            result = np.full(timegrid.shape, f)
            return result

    def __init__(self, deviation_process: object, 
                    seasonal_function: object,
                    name:str = 'Wind_Germany'):
        """Wind Power Model to model the efficiency of wind power production.

        Args:
            speed_of_mean_reversion (Union[float, Callable]): _description_
            volatility (Union[float, Callable]): _description_
            mean_level (Union[float, Callable]): _description_
        """
        self.deviation_process = deviation_process
        self.seasonal_function = seasonal_function
        # self.speed_of_mean_reversion = speed_of_mean_reversion
        # self.volatility = volatility
        # self.mean_level = mean_level
        self.name = name
        self._timegrid = None

    
    def simulate(self, timegrid, start_value, rnd):
        mean = self.seasonal_function.compute(timegrid)[:, np.newaxis]
        start_value_ = _logit(start_value) - mean[0,0]
        deviation = self.deviation_process.simulate(timegrid.timegrid, start_value_, rnd)
        return _inv_logit(mean + deviation)

class SupplyFunction:
    def __init__(self, floor:tuple, cap:tuple, peak:tuple, offpeak:tuple, peak_hours: set):
        self.floor = floor
        self.cap = cap
        self.peak = peak
        self.offpeak = offpeak
        self.peak_hours = peak_hours

    def compute(self, q, d:dt.datetime):
        def cutoff(x):
            return np.minimum(self.cap[1], np.maximum(self.floor[1], x))
        if q<=self.floor[0]:
            return self.floor[1]
        elif q>=self.cap[0]:
            return self.cap[1]
        if d.hour not in self.peak_hours:
            return cutoff(self.offpeak[0]+self.offpeak[1]/(q-self.floor[0])+self.offpeak[2]*q)
        return cutoff(self.peak[0]+self.peak[1]/(self.cap[0]-q))

    def plot(self, d:dt.datetime, res_demand_low = None, res_demand_high = None):
        if res_demand_low is None:
            res_demand_low = self.floor[0]
        if res_demand_high is None:
            res_demand_high = self.cap[0]
        q = np.linspace(res_demand_low, res_demand_high, 50)
        f = [self.compute(x, d) for x in q]
        plt.plot(q,f,'-', label=str(d))
        plt.xlabel('residual demand')
        plt.ylabel('price')

class LoadModel:
    def __init__(self,deviation_process: object, load_profile: object):
        """Model the power load. 

        Args:
            deviation_process (object): _description_
            load_profile (object): _description_
        """
        self.load_profile = load_profile
        self.deviation_process = deviation_process

    def simulate(self, timegrid: DateTimeGrid, start_value: float, rnd:np.ndarray):
        result = np.empty((timegrid.shape[0], rnd.shape[0]))
        result[0,:] = start_value
        deviation = self.deviation_process.simulate(timegrid.timegrid, start_value, rnd)
        return self.load_profile.get_profile(timegrid)[:, np.newaxis] + deviation
   
class ResidualDemandModel:
    def __init__(self, wind_model: object, capacity_wind: float, 
                    solar_model: object, capacity_solar: float,  
                    load_model: object, supply_curve: object):
        """Residual demand model to model power prices.

        This model is based on the paper by :footcite:t:`Wagner2012` and models power (spot) prices :math:`p_t` depending (by a deterministic function :math:`f`) on the residual demand
        :math:`R_t`,

        .. math::
            p_t = f(R_t) = f(L_t - IC^w\cdot E_t^w - IC_t^s\cdot E_t^s)

        where

            - :math:`L_t` is the demand (load) at time :math:`t`,
            - :math:`IC^w` denotes the installed capacity of wind (in contrast to the paper this is not time dependent),
            - :math:`E_t^w` is the wind efficiency at time :math:`t`,
            - :math:`IC^s` denotes the installed capacity of solar (in contrast to the paper this is not time dependent),
            - :math:`E_t^s` is the solar efficiency at time :math:`t`.



        Args:
            wind_model (object): Model for wind efficiency (needs to implement a method simulate in order to work with this model). 
                        See :func:`rivapy.models.WindPowerModel` as an example for a wind model.
            capacity_wind (object): The capacity of wind power. This is multiplied with the simulated efficiency to obtain the simulated absolute amount of wind.
            solar_model (object): Model for solar efficiency (needs to implement a method simulate in order to work with this model). 
                        See :func:`rivapy.models.SolarPowerModel` as an example for a solar model.
            capacity_solar (object): The capacity of solar power. This is multiplied with the simulated efficiency to obtain the simulated absolute amount of solar.
            load_model (object): Model for load. See :func:`rivapy.models.LoadModel` as an example for a load model.
            supply_curve (object): The total demand, see :func:`rivapy.models.SupplyFunction` for an example.
        """
        self.wind_model = wind_model
        self.capacity_wind = capacity_wind
        self.solar_model = solar_model
        self.capacity_solar = capacity_solar
        self.load_model = load_model
        self.supply_curve = supply_curve

    def simulate(self, timegrid: DateTimeGrid, 
                    start_value_wind: float, 
                    start_value_solar: float, 
                    start_value_load: float,
                    n_sims: int,
                    rnd_wind: np.ndarray=None,
                    rnd_solar: np.ndarray=None,
                    rnd_load: float=None,
                    rnd_state = None):
        """Simulate the residual demand model on a given datetimegrid.

        Args:
            timegrid (DateTimeGrid): _description_
            start_value_wind (float): _description_
            start_value_solar (float): _description_
            start_value_load (float): _description_
            n_sims (int): _description_
            rnd_wind (np.ndarray, optional): _description_. Defaults to None.
            rnd_solar (np.ndarray, optional): _description_. Defaults to None.
            rnd_load (float, optional): _description_. Defaults to None.
            rnd_state (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        np.random.seed(rnd_state)
        if rnd_wind is None:
            rnd_wind = np.random.normal(size=(n_sims, timegrid.shape[0]-1))
        if rnd_solar is None:
            rnd_solar = np.random.normal(size=(n_sims, timegrid.get_daily_subgrid().shape[0]-1))
        if rnd_load is None:
            rnd_load = np.random.normal(size=(n_sims, timegrid.shape[0]-1))
        lm = self.load_model.simulate(timegrid, start_value_load, rnd_load)
        sm = self.capacity_solar*self.solar_model.simulate(timegrid, start_value_solar, rnd_solar)
        wm = self.capacity_wind*self.wind_model.simulate(timegrid, start_value_wind, rnd_wind)
        residual_demand = lm - sm - wm
        power_price = np.zeros(shape=( timegrid.shape[0], n_sims))
        for i in range(timegrid.shape[0]):
            for j in range(n_sims):
                power_price[i,j] =  self.supply_curve.compute(residual_demand[i,j],timegrid.dates[i] )
        result = {}
        result['load'] = lm
        result['solar'] = sm
        result['wind'] = wm
        result['price'] = power_price
        return result
