import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import abc
import sys
sys.path.append('C:/Users/doeltz/development/RiVaPy')
import warnings
from typing import Union, Callable, List, Tuple, Dict, Protocol, Set
from scipy.special import comb
from rivapy.tools.datetime_grid import DateTimeGrid, InterpolatedFunction, PeriodicFunction
from rivapy.models.ornstein_uhlenbeck import OrnsteinUhlenbeck
from rivapy.tools.interfaces import DateTimeFunction, FactoryObject

def _logit(x):
    return np.log(x/(1-x))

def _inv_logit(x):
    return 1.0/(1+np.exp(-x))

class ForwardSimulationResult(abc.ABC):
    @abc.abstractmethod
    def n_forwards(self)->float:
        pass
    @abc.abstractmethod
    def udls(self)->Set[str]:
        pass

    def keys(self)->List[str]:
        result = set()
        for udl in self.udls():
            for i in range(self.n_forwards()):
                result.add(ForwardSimulationResult.get_key(udl, i))
    
    def get(self, key: str)->np.ndarray:
        ...
    @staticmethod
    def get_key(udl: str, n_forward: int)->str:
        return 
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
    class ForwardSimulationResult(ForwardSimulationResult):
        def __init__(self, paths:np.ndarray, 
                        wind_forecast_model, 
                        timegrid, 
                        expiries, 
                        initial_forecasts):
            self._paths = paths
            self._timegrid = timegrid
            self._model = wind_forecast_model
            self._result = None
            self.expiries = expiries
            self.forecasts = initial_forecasts
            self._ou_additive_forward_corrections = np.empty((len(expiries)))
            for i in range(len(expiries)):
                correction = _logit(initial_forecasts[i])-self._model.ou.compute_expected_value(0.0, expiries[i])
                self._ou_additive_forward_corrections[i] = correction

        def n_forwards(self)->float:
            return len(self.expiries)

        def keys(self)->List[str]:
            return [self._model.region + '_' + str(i) for i in range(self.n_forwards())]

        def get(self, key: str)->np.ndarray:
            pass

        def get_fwd(self, index_t, index_T):
            return self._wind_forecast_model.get_forward(self._paths[index_t,:],self._timegrid[index_t], index_T)
            
        def _get_path(self, fwd_expiry: int = None):
            if fwd_expiry is None:
                return self._paths
            result = np.empty((self._timegrid.shape[0], self._paths.shape[1]))
            for i in range(self._timegrid.shape[0]):
                result[i,:] = self.get_fwd(i, fwd_expiry)
            return result

        def path_dict(self):
            if self._result is None:
                self._result = {}
                for i in range(self._model.n_forwards()):
                    self._result[self._model.region+'_'+str(i)] = self.get_path(i)
            return self._result

        def keys(self)->List[str]:
            return list(self.path_dict().keys())

        def get(self, key:str):
            return self.path_dict()[key]

        def get_forward(self, key:str, fwd_number: int)->np.ndarray:
            k = key+'_'+str(fwd_number)
            return self.get(k)

    def __init__(self, region: str = 'Wind_Germany',
                    speed_of_mean_reversion: float, 
                    volatility: float,
                    expiries: List[float],
                    initial_forecasts: List[float],
                    ):
        """Simple Model to simulate wind forecasts.

        The base model used within this model is the Ornstein-Uhlenbeck process and uses internally the class :class:`rivapy.models.OrnsteinUhlenbeck` with a mean level of zero to simulate the forecast.
        Here, the forecast is computed as the expected value of the Ornstein-Uhlenbeck process conditioned on current simulated value.

        Args:
            speed_of_mean_reversion (float): The speed of mean reversion of the underlying Ornstein-Uhlenbeck process (:class:`rivapy.models.OrnsteinUhlenbeck`).
            volatility (float): The volatility of the underlying Ornstein-Uhlenbeck process (:class:`rivapy.models.OrnsteinUhlenbeck`).
            expiries (List[float]): A list of the expiries of the futures that will be simulated.
            forecasts (List[float]): The forecasted efficiencies at each of the futures that will be simulated.
            name (str): The name of the respective wind region.
        """
        if len(expiries) != len(initial_forecasts):
            raise Exception('Number of forward expiries does not equal number of initial forecasts. Each forward expiry needs an own forecast.')
        self.ou = OrnsteinUhlenbeck(speed_of_mean_reversion, volatility, mean_reversion_level=0.0)
        self.expiries = expiries
        self.forecasts = initial_forecasts
        self.region = region
        self._timegrid = None
        
        self._ou_additive_forward_corrections = np.empty((len(expiries)))
        for i in range(len(expiries)):
            #mean_ou = _inv_logit(self.ou.compute_expected_value(0.0, expiries[i]))
            correction = _logit(initial_forecasts[i])-self.ou.compute_expected_value(0.0, expiries[i])
            self._ou_additive_forward_corrections[i] = correction

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
    
    def simulate(self, timegrid, 
                rnd: np.ndarray,
                expiries: List[float],
                forecast: List[float],
                startvalue=0.0)->SimulationResult:
        paths = self.ou.simulate(timegrid, startvalue, rnd)
        return WindPowerForecastModel.SimulationResult(paths, self, timegrid)

class MultiRegionWindForecastModel:
    class SimulationResult(SimulationResultForward):
        def __init__(self, model, regions_result: SimulationResult):
            self._results = regions_result
            self._model = model

        def path_dict(self):
            result = {}
            for r in self._results:
                result.update(r.path_dict())
            return result

        def keys(self)->List[str]:
            result = []
            for r in self._results:
                result.extend(r.keys())
            return result

        def path(self)->np.ndarray:
            raise NotImplementedError('This method is not implemented.')

        def n_forwards(self)->int:
            return self.model.n_forwards()
    
        def get_forward(self, key: str, fwd_number: int)->np.ndarray:
            pass    
        def get(self, key:str):
            pass
        
    class Region:
        def __init__(self, model: WindPowerForecastModel,  capacity: float, rnd_weights: List[float]):
            self.model = model
            self.capacity  = capacity
            self.rnd_weights = rnd_weights

        def name(self):
            return self.model.region

        def n_random(self):
            return len(self.rnd_weights)

    def __init__(self, name: str, region_forecast_models: List[Region]):
        self.name = name
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
    
    def simulate_new(self, timegrid, rnd: np.ndarray, startvalue=0.0):
        r = self.simulate(timegrid, rnd, startvalue)
        return MultiRegionWindForecastModel.SimulationResult(r)

    def rnd_shape(self, n_sims: int, n_timesteps: int)->tuple:
        return (len(self._region_forecast_models), n_timesteps-1, n_sims)

class ResidualDemandForwardModel(FactoryObject):
    class SimulationResult:
        def __init__(self, power_name, n_forwards, power_fwd, efficiencies):
            self.result = {}
            for i in range(n_forwards):
                self.result[power_name+str(i)] = power_fwd[:,:,i]
                for k,v in efficiencies.items():
                    self.result[k+str(i)] = v[:,:,i]
            
        def path_dict(self):
            return self.result

    def __init__(self, wind_power_forecast,
                        highest_price_ou_model, 
                        supply_curve: Callable[[float], float],
                        max_price: float,
                        forecast_hours: List[int]=None,
                        power_name:str = None):
        self.wind_power_forecast = wind_power_forecast
        self.highest_price_ou_model = highest_price_ou_model
        self.supply_curve = supply_curve
        self.forecast_hours = forecast_hours
        self.max_price = max_price
        if power_name is not None:
            self.power_name = power_name
        else:
            self.power_name = 'POWER'    
        #self.region_to_capacity = region_to_capacity
        
    def _to_dict(self)->dict:
        return {'wind_power_forecast': self.wind_power_forecast.to_dict(),
                'supply_curve': self.supply_curve.to_dict(),
                'highest_price_ou_model': self.highest_price_ou_model.to_dict(),
                'forecast_hours': self.forecast_hours,
                'max_price': self.max_price,
                'power_name': self.power_name,
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
        return self.wind_power_forecast.region

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
                    #print(i,timegrid[i], j,forecast.expiry(j))
                    #if abs(forecast.expiry(j)-timegrid[i])<1e-6:
                    #    print(i,timegrid[i], j,forecast.expiry(j))
                    if i in forecast_timepoints or i == 0 or abs(forecast.expiry(j)-timegrid[i])<1e-6:
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
    
    def _simulate_multi_region_new(self, timegrid: Union[np.ndarray, DateTimeGrid], 
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
        wind = self.wind_power_forecast.simulate_new(timegrid, rnd[1:,:])
        efficiency_forecast_total = np.zeros((timegrid.shape[0], rnd.shape[2], self.wind_power_forecast.n_forwards()))
        result_efficiencies = {}
        for region, forecast in wind._results.items():
            efficiency_forecast = np.empty((timegrid.shape[0], rnd.shape[2], self.wind_power_forecast.n_forwards()))
            multiplier = self.wind_power_forecast.region_relative_capacity(region)
            for i in range(timegrid.shape[0]):
                for j in range(self.wind_power_forecast.n_forwards()):
                    #print(i,timegrid[i], j,forecast.expiry(j))
                    #if abs(forecast.expiry(j)-timegrid[i])<1e-6:
                    #    print(i,timegrid[i], j,forecast.expiry(j))
                    if i in forecast_timepoints or i == 0 or abs(forecast.expiry(j)-timegrid[i])<1e-6:
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
        return ResidualDemandForwardModel.SimulationResult(self.power_name, self.wind_power_forecast.n_forwards(),
                            power_fwd, result_efficiencies)
    

    def compute_rlzd_qty(self, location:str, simulated_forecasts: Dict[str, np.ndarray])->np.ndarray:
        for k,v in simulated_forecasts.items():
            if location == k:
                return v[-1,:,:]

    def simulate(self, timegrid: Union[np.ndarray, DateTimeGrid], 
                rnd: np.ndarray, 
                forecast_timepoints: List[int]=None)->Tuple[np.ndarray, Dict[str, np.ndarray]]:
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
        wind = self.wind_power_forecast.simulate(timegrid, rnd[1])._paths
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
        return power_fwd, {self.wind_power_forecast.region: current_forecast}
        
    

if __name__=='__main__':
    from rivapy.models.residual_demand_model import MultiRegionWindForecastModel
    forward_expiries = [(24.0+23.0)/365.0, 24.0*2/365.0]
    regions = [ MultiRegionWindForecastModel.Region( 
                                    WindPowerForecastModel(speed_of_mean_reversion=0.5, 
                                                           volatility=1.80, 
                                                            expiries=forward_expiries,
                                                            forecasts = [0.8, 0.8],#*len(forward_expiries)
                                                            region = 'Onshore'
                                                            ),
                                    capacity=1000.0,
                                    rnd_weights=[0.7,0.3]
                                ),
           MultiRegionWindForecastModel.Region( 
                                    WindPowerForecastModel(speed_of_mean_reversion=0.5, 
                                                           volatility=1.80, 
                                                            expiries=forward_expiries,
                                                            forecasts = [0.6, 0.7],#*len(forward_expiries)
                                                            region = 'Offshore'
                                                            ),
                                    capacity=500.0,
                                    rnd_weights=[0.3,0.7]
                                )
           
          ]
    days = 2 
    timegrid = np.linspace(0.0, days*1.0/365.0, days*24)
    multi_region_model = MultiRegionWindForecastModel(regions)
    highest_price = OrnsteinUhlenbeck(speed_of_mean_reversion=1.0, volatility=0.01, mean_reversion_level=1.0)
    supply_curve = SmoothstepSupplyCurve(1.0, 0)
    rdm = ResidualDemandForwardModel(
                                    #wind_forecast_model, 
                                    multi_region_model,
                                    highest_price,
                                    supply_curve,
                                    max_price = 1.0,
                                    forecast_hours=None,#[6, 10, 14, 18], 
                                    #region_to_capacity=None
                                    )
    rnd = np.random.normal(size=rdm.rnd_shape(1000, timegrid.shape[0]))
    result = rdm._simulate_multi_region_new(timegrid, rnd, [10,20])
