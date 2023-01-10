import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import warnings
from typing import Union, Callable, List, Tuple, Dict
from scipy.special import comb
from rivapy.tools.datetime_grid import DateTimeGrid, InterpolatedFunction, PeriodicFunction
from rivapy.models.ornstein_uhlenbeck import OrnsteinUhlenbeck
from rivapy.tools.interfaces import DateTimeFunction, FactoryObject

def _logit(x):
    return np.log(x/(1-x))

def _inv_logit(x):
    return 1.0/(1+np.exp(-x))

class CosinusSeasonality:
    def __init__(self, x: np.ndarray = np.array([0,1,0,1,0])):
        self.x = x
        
    def __call__(self, x):
        return self.x[0]*np.cos(2*np.pi*x+self.x[1]) + self.x[2]*np.cos(4*np.pi*x+self.x[3]) + self.x[4]
        

class SolarProfile:
    def __init__(self, profile:Callable):
        self._profile = profile

    def get_profile(self, timegrid: DateTimeGrid):
        result = np.empty(timegrid.timegrid.shape[0])
        for i in range(result.shape[0]):
            result[i] = self._profile(timegrid.dates[i])
        return result

class MonthlySolarProfile(SolarProfile):
    def __init__(self, monthly_profiles: np.ndarray):
        self._monthly_profiles = monthly_profiles
        super().__init__(self.__monthly_solar_profile)
        if monthly_profiles is not None:
            if monthly_profiles.shape != (12,24):
                raise ValueError('Monthly profiles must have shape (12,24)')

    def __monthly_solar_profile(self, d):
        return self._monthly_profiles[d.month-1, d.hour]
        

class SolarPowerModel:
    def _eval_grid(f, timegrid):
        return f(timegrid)
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
        ml = self.mean_level.compute(timegrid)[:, np.newaxis]
        start_value_ = _logit(start_value) - ml[0,0]
        daily_maximum = self._daily_maximum_process.simulate(tg_.timegrid, start_value_, rnd)
        profile = self._profile.get_profile(timegrid)
        result = np.empty((timegrid.shape[0], rnd.shape[1]))
        day = 0
        d = tg_.dates[0].date()
        for i in range(timegrid.timegrid.shape[0]):
            if d != timegrid.dates[i].date():
                day += 1
                d = timegrid.dates[i].date()
            result[i,:] = _inv_logit(daily_maximum[day,:] + ml[i,0])* profile[i] 
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

    def simulate(self, timegrid: DateTimeGrid, start_value, rnd):
        mean = self.seasonal_function.compute(timegrid)[:, np.newaxis]
        start_value_ = _logit(start_value) - mean[0,0]
        deviation = self.deviation_process.simulate(timegrid.timegrid, start_value_, rnd)
        return _inv_logit(mean + deviation)
    
    @staticmethod
    def calibrate(deviation_model, capacities:pd.DataFrame, data: pd.DataFrame, seasonality_function, min_efficiency=0.001, max_efficiency=0.99, **kwargs):
        if capacities is not None:
            if 'efficiency' in  data.columns:
                warnings.warn('Capacities are defined but the data already contains a column efficiency with productions transformed by capacity!')
            capacities_interp = InterpolatedFunction(capacities.index, capacities['capacity'])
            data['efficiency'] = data['production']/capacities_interp.compute(data.index)
        data['logit_efficiency'] = _logit(np.minimum(np.maximum(data['efficiency'], min_efficiency), max_efficiency)) 
        f = CosinusSeasonality(x=np.array([1.0, 1, 0.9, 1.1, 0.5, -1.0]))
        pf_target = PeriodicFunction(f, frequency='Y', granularity=pd.infer_freq(data.index), ignore_leap_day=True)
        pf_target.calibrate(data.index, data['logit_efficiency'].values)
        data['des_logit_efficiency'] = data['logit_efficiency']-pf_target.compute(DateTimeGrid(data.index))
        deviation_model.calibrate(data['des_logit_efficiency'].values,dt=1.0/(24.0*365.0),**kwargs)
        return WindPowerModel(deviation_model, pf_target)

class SmoothstepSupplyCurve(FactoryObject): 
    def __init__(self, s, N):
        self.s = s
        self.N = N
        
    def _to_dict(self):
        return {'s': self.s, 'N': self.N}

    @staticmethod
    def smoothstep(x, x_min=0, x_max=1, N=1):
        x = np.clip((x - x_min) / (x_max - x_min), 0, 1)
        result = 0
        for n in range(0, N + 1):
            result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n
        result *= x ** (N + 1)
        return result

    def __call__(self, residual):
        #wind_production = wind_production#np.maximum(np.minimum(wind_production, 0.99), 0.01)
        #residual = (1.0-wind_production)
        residual = np.power(residual, self.s)
        return SmoothstepSupplyCurve.smoothstep(residual, N=self.N)

class WindPowerForecastModel(FactoryObject):
    class SimulationResult:
        def __init__(self, paths:np.ndarray, wind_forecast_model, timegrid):
            self._paths = paths
            self._timegrid = timegrid
            self._wind_forecast_model = wind_forecast_model

        def n_forwards(self)->float:
            return self._wind_forecast_model.n_forwards()

        def expiry(self, expiry: int)->float:
            return self._wind_forecast_model.expiries[expiry]

        def get_fwd(self, index_t, index_T):
            return self._wind_forecast_model.get_forward(self._paths[index_t,:],self._timegrid[index_t], index_T)
            
        def get_path(self):
            return self._paths

    def __init__(self, speed_of_mean_reversion: float, 
                    volatility: float,
                    expiries: List[float],
                    forecasts: List[float], # must be given as logit of percentage of max_capacity
                    region: str = 'Wind_Germany',
                    ):
        """Simple Model to simulate wind forecasts.

        The base model used within this model is the Ornstein-Uhlenbeck process and uses internally the class :class:`rivapy.models.OrnsteinUhlenbeck` with a mean level of zero to simulate the forecast.
        Here, the forecast is computed as the expected value of the Ornstein-Uhlenbeck process conditioned on current simulated value.

        Args:
            speed_of_mean_reversion (float): The speed of mean reversion of the underlying Ornstein-Uhlenbeck process (:class:`rivapy.models.OrnsteinUhlenbeck`).
            volatility (float): The volatility of the underlying Ornstein-Uhlenbeck process (:class:`rivapy.models.OrnsteinUhlenbeck`).
            expiries (List[float]): A list of the expiries of the futures that will be simulated.
            forecasts (List[float]): The forecasted value of each of the futures that will be simulated.
            name (str): The name of teh respective wind region.
        """
        self.ou = OrnsteinUhlenbeck(speed_of_mean_reversion, volatility, mean_reversion_level=0.0)
        self.expiries = expiries
        self.forecasts = forecasts
        self.region = region
        self._timegrid = None
        
        self._ou_additive_forward_corrections = np.empty((len(expiries)))
        for i in range(len(expiries)):
            #mean_ou = _inv_logit(self.ou.compute_expected_value(0.0, expiries[i]))
            correction = _logit(forecasts[i])-self.ou.compute_expected_value(0.0, expiries[i])
            self._ou_additive_forward_corrections[i] = correction

    def region(self)->str:
        return self.region

    def _to_dict(self)->dict:
        return {'speed_of_mean_reversion': self.ou.speed_of_mean_reversion, 
            'volatility': self.ou.volatility,
                'expiries': self.expiries, 'forecasts': self.forecasts, 
                'region': self.region}
    def n_forwards(self):
        return len(self.expiries)
    
    def get_forward(self, paths, t, num_expiry):
        expected_ou = self.ou.compute_expected_value(paths, self.expiries[num_expiry]-t)#+correction
        return _inv_logit(expected_ou + self._ou_additive_forward_corrections[num_expiry])
            
    def rnd_shape(self, n_sims: int, n_timesteps: int)->tuple:
        return (n_timesteps-1, n_sims)
    
    def simulate(self, timegrid, rnd: np.ndarray, startvalue=0.0):
        paths = self.ou.simulate(timegrid, startvalue, rnd)
        return WindPowerForecastModel.SimulationResult(paths, self, timegrid)

class MultiRegionWindForecastModel:
    class Region:
        def __init__(self, model: WindPowerForecastModel,  capacity: float, rnd_weights: List[float]):
            self.model = model
            self.capacity  = capacity
            self.rnd_weights = rnd_weights

        def name(self):
            return self.model.region

        def n_random(self):
            return len(self.rnd_weights)

    def __init__(self, region_forecast_models: List[Region]):
        if len(region_forecast_models)==0:
            raise Exception('Empty list of models is not allowed')
        n_rnd_ref_model = region_forecast_models[0].n_random()
        n_fwd_ref_model = region_forecast_models[0].model.n_forwards()
        for i in range(1, len(region_forecast_models)):
            if region_forecast_models[i].n_random() != n_rnd_ref_model:
                raise Exception('All regions must have the same number of random variables.')
            if region_forecast_models[i].model.n_forwards() != n_fwd_ref_model:
                raise Exception('All regions must have the same number of fwds.')
        self._region_forecast_models = region_forecast_models

    def n_forwards(self):
        return self._region_forecast_models[0].model.n_forwards()

    def rnd_shape(self, n_sims: int, n_timesteps: int)->tuple:
        return (self._region_forecast_models[0].rnd_weights, n_timesteps-1, n_sims)

    def total_capacity(self):
        result = 0.0
        for r in self._region_forecast_models:
            result += r.capacity
        return result

    def region_relative_capacity(self, region: str):
        for r in self._region_forecast_models:
            if r.name() == region:
                return r.capacity/self.total_capacity()
        raise Exception('Model does not contain a region with name ' + region)

    def region_names(self)->List[str]:
        return [r.name() for r in self._region_forecast_models]

    def simulate(self, timegrid, rnd: np.ndarray, startvalue=0.0):
        results = {}
        for region in self._region_forecast_models:
            rnd_ = region.rnd_weights[0]*rnd[0,:,:]
            for i in range(1, region.n_random()):
                rnd_ += region.rnd_weights[i]*rnd[i,:,:]
            results[region.name()] = region.model.simulate(timegrid, rnd_, startvalue)
        return results

    def rnd_shape(self, n_sims: int, n_timesteps: int)->tuple:
        return (len(self._region_forecast_models), n_timesteps-1, n_sims)

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

class ResidualDemandForwardModel(FactoryObject):
        
    def __init__(self, wind_power_forecast,
                        highest_price_ou_model, 
                        supply_curve: Callable[[float], float],
                        max_price: float,
                        forecast_hours: List[int]=None):
        self.wind_power_forecast = wind_power_forecast
        self.highest_price_ou_model = highest_price_ou_model
        self.supply_curve = supply_curve
        self.forecast_hours = forecast_hours
        self.max_price = max_price
        #self.region_to_capacity = region_to_capacity
        
    def _to_dict(self)->dict:
        return {'wind_power_forecast': self.wind_power_forecast.to_dict(),
                'supply_curve': self.supply_curve.to_dict(),
                'highest_price_ou_model': self.highest_price_ou_model.to_dict(),
                'forecast_hours': self.forecast_hours,
                'max_price': self.max_price,
                #'region_to_capacity': self.region_to_capacity
                }

    def rnd_shape(self, n_sims: int, n_timesteps: int)->tuple:
        rnd_shape = self.wind_power_forecast.rnd_shape(n_sims, n_timesteps)
        if len(rnd_shape) == 3:
            return (rnd_shape[0]+1, rnd_shape[1], rnd_shape[2])
        return (2, rnd_shape[0], rnd_shape[1])

    def get_technology(self)->str:
        """Return name of the technology modeled.

        Returns:
            str: Name of instrument.
        """
        return self.wind_power_forecast.name

    def _simulate_multi_region(self, timegrid: Union[np.ndarray, DateTimeGrid], 
                rnd: np.ndarray, 
                forecast_timepoints: List[int]=None):
        if forecast_timepoints is None and self.forecast_hours is None:
            raise Exception('Either a list of timepoints or a list of publishing hours for forecast must be specified.')
        if forecast_timepoints is not None and self.forecast_hours is not None:
            raise Exception('You cannot specify forecast_timepoints since forecast_hours have already been specified.')
        if forecast_timepoints is None:
            if not isinstance(timegrid, DateTimeGrid):
                raise Exception('If forecast_timepoints is None, timegrid must be of type DateTimeGrid so that the points can be determined.')
            forecast_timepoints = [i for i in range(len(timegrid.dates)) if timegrid.dates[i].hour in self.forecast_hours]
            timegrid = timegrid.timegrid
        highest_prices = self.highest_price_ou_model.simulate(timegrid, 1.0, rnd[0,:])*self.max_price
        wind = self.wind_power_forecast.simulate(timegrid, rnd[1:,:])
        efficiency_forecast_total = np.zeros((timegrid.shape[0], rnd.shape[2], self.wind_power_forecast.n_forwards()))
        result_efficiencies = {}
        for region, forecast in wind.items():
            efficiency_forecast = np.empty((timegrid.shape[0], rnd.shape[2], self.wind_power_forecast.n_forwards()))
            multiplier = self.wind_power_forecast.region_relative_capacity(region)
            for i in range(timegrid.shape[0]):
                for j in range(self.wind_power_forecast.n_forwards()):
                    if i in forecast_timepoints or i == 0:
                        efficiency_forecast[i,:,j] = forecast.get_fwd(i,j)
                        efficiency_forecast_total[i,:,j] += multiplier*efficiency_forecast[i,:,j]
                    else:
                        efficiency_forecast[i,:,j] = efficiency_forecast[i-1,:,j]
                        efficiency_forecast_total[i,:,j] = efficiency_forecast_total[i-1,:,j]
            result_efficiencies[region] = efficiency_forecast     
        power_fwd = np.empty((timegrid.shape[0], rnd.shape[2], self.wind_power_forecast.n_forwards()))
        for i in range(timegrid.shape[0]):
            for j in range(self.wind_power_forecast.n_forwards()):
                power_fwd[i,:,j] =  self.supply_curve(1.0-efficiency_forecast_total[i,:,j] )*highest_prices[i,:]
        return power_fwd, result_efficiencies

    def simulate(self, timegrid: Union[np.ndarray, DateTimeGrid], 
                rnd: np.ndarray, 
                forecast_timepoints: List[int]=None):
        multi_region = True
        #self.wind_power_forecast.region_names()
        try:
            r_names = self.wind_power_forecast.region_names()    
        except:
            multi_region = False
        if multi_region:
            return self._simulate_multi_region(timegrid, rnd, forecast_timepoints)

        if forecast_timepoints is None and self.forecast_hours is None:
            raise Exception('Either a list of timepoints or a list of publishing hours for forecast must be specified.')
        if forecast_timepoints is not None and self.forecast_hours is not None:
            raise Exception('You cannot specify forecast_timepoints since forecast_hours have already been specified.')
        if forecast_timepoints is None:
            if not isinstance(timegrid, DateTimeGrid):
                raise Exception('If forecast_timepoints is None, timegrid must be of type DateTimeGrid so that the points can be determined.')
            forecast_timepoints = [i for i in range(len(timegrid.dates)) if timegrid.dates[i].hour in self.forecast_hours]
            timegrid = timegrid.timegrid
        highest_prices = self.highest_price_ou_model.simulate(timegrid, 1.0, rnd[0,:])*self.max_price
        wind = self.wind_power_forecast.simulate(timegrid, rnd[1:,:])._paths
        power_fwd = np.empty((timegrid.shape[0], rnd.shape[2], self.wind_power_forecast.n_forwards()))
        current_forecast = np.empty((timegrid.shape[0], rnd.shape[2], self.wind_power_forecast.n_forwards()))
        for i in range(timegrid.shape[0]):
            for j in range(self.wind_power_forecast.n_forwards()):
                if i in forecast_timepoints or i == 0:
                        current_forecast[i,:,j] =self.wind_power_forecast.get_forward(wind[i,:], timegrid[i],j)
                else:
                    current_forecast[i,:,j] = current_forecast[i-1,:,j]
            for j in range(self.wind_power_forecast.n_forwards()):
                power_fwd[i,:,j] =  self.supply_curve( 1.0-current_forecast[i,:,j] )*highest_prices[i,:]
        return {'POWER_FWD': power_fwd, self.wind_power_forecast.region: current_forecast}
        
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


# class SimpleRenewableModel:
#     def __init__(self, wind_model_onshore: object, 
#                     capacity_wind_onhore: float, 
#                     target_wind_onhore: DateTimeFunction, 
#                     wind_model_offshore: object, 
#                     capacity_wind_offore: float, 
#                     target_wind_offhore: float, 
#                     solar_model: object, 
#                     capacity_solar: float,
#                     target_solar: object,
#                     power_model: object ):
#         """Simple model to simulate power prices together with renewables

#         This model is inspried by the paper by :footcite:t:`BieglerKoenig2022` and models power (spot) prices together with renewables
            

#         Args:
#             capacity_wind (interfaces.DateTimeFunction): The capacity of wind power. This is multiplied with the simulated efficiency to obtain the simulated absolute amount of wind.
#             solar_model (object): Model for solar efficiency (needs to implement a method simulate in order to work with this model). 
#                         See :func:`rivapy.models.SolarPowerModel` as an example for a solar model.
#             capacity_solar (object): The capacity of solar power. This is multiplied with the simulated efficiency to obtain the simulated absolute amount of solar.
#             power_model (object): The total demand, see :func:`rivapy.models.SupplyFunction` for an example.
#         """
#         pass
        
#     def simulate(self, timegrid: DateTimeGrid, 
#                     start_value_wind: float, 
#                     start_value_solar: float, 
#                     start_value_load: float,
#                     n_sims: int,
#                     rnd_wind: np.ndarray=None,
#                     rnd_solar: np.ndarray=None,
#                     rnd_load: float=None,
#                     rnd_state = None):
#         """Simulate the residual demand model on a given datetimegrid.

#         Args:
#             timegrid (DateTimeGrid): _description_
#             start_value_wind (float): _description_
#             start_value_solar (float): _description_
#             start_value_load (float): _description_
#             n_sims (int): _description_
#             rnd_wind (np.ndarray, optional): _description_. Defaults to None.
#             rnd_solar (np.ndarray, optional): _description_. Defaults to None.
#             rnd_load (float, optional): _description_. Defaults to None.
#             rnd_state (_type_, optional): _description_. Defaults to None.

#         Returns:
#             _type_: _description_
#         """
#         pass

