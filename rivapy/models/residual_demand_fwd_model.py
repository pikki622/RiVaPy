import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import abc
from typing import Union, Callable, List, Tuple, Dict, Protocol, Set
from scipy.special import comb
from rivapy.tools.interfaces import FactoryObject
from rivapy.models.factory import create as _create
from rivapy.models.ornstein_uhlenbeck import OrnsteinUhlenbeck
from rivapy.models.base_model import BaseFwdModel

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
                result.add(BaseFwdModel.get_key(udl, i))
        return result

    @abc.abstractmethod
    def get(self, key: str)->np.ndarray:
        pass
    


class WindPowerForecastModel(BaseFwdModel):
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
            self.initial_forecasts = initial_forecasts
            self._ou_additive_forward_corrections = wind_forecast_model._compute_ou_additive_forward_correction(self.expiries, self.initial_forecasts)

        def n_forwards(self)->float:
            return len(self.expiries)

        def udls(self)->Set[str]:
            return self._model.udls()

        def get(self, key: str, forecast_timepoints: List[int]=None)->np.ndarray:
            expiry =  BaseFwdModel.get_expiry_from_key(key)
            return self._get(expiry, forecast_timepoints)

        def _get(self, expiry: int, forecast_timepoints: List[int])->np.ndarray:
            result = np.empty(self._paths.shape)
            for i in range(self._timegrid.shape[0]):
                #print(self.expiries[expiry]-self._timegrid[i])
                result[i,:] = self._model.get_forward(self._paths[i,:], self.expiries[expiry]-self._timegrid[i], self._ou_additive_forward_corrections[expiry])
            if forecast_timepoints is not None:
                ftp_prev = 1
                for ftp in forecast_timepoints:
                    for i in range(ftp_prev, ftp):
                        if abs(self._timegrid[i]-self.expiries[expiry])>1e-6:
                            result[i,:] = result[i-1,:]
                    ftp_prev = ftp+1
                for i in range(ftp_prev, self._timegrid.shape[0]):
                    if abs(self._timegrid[i]-self.expiries[expiry])>1e-6:
                        result[i,:] = result[i-1,:]
            return result
        
    def __init__(self, region: str,
                    speed_of_mean_reversion: float, 
                    volatility: float,
                    ):
        """Simple model to simulate forecasts of wind efficiencies (power production by wind as percentage of total wind capacity) based on the Ornstein-Uhlenbeck process.

        The base model used within this model is the Ornstein-Uhlenbeck process and uses internally the class :class:`rivapy.models.OrnsteinUhlenbeck` with a mean level of zero to simulate the forecast.
        Here, the forecast of wind efficiency :math:`w_{t,T}` at time :math:`T` is computed as the expected value of the standard logistic function (sigmoid) applied to the Ornstein-Uhlenbeck process conditioned on current simulated value,
        i.e.

        .. math:: w_{t, T} = \\frac{1}{1+e^{-(X_{t,T}+\\nu(T))}} 

        where :math:`X_{t,T}:=E[X_T\mid X_t]` with

        .. math:: dX = -\\lambda X dt + \\sigma dW_t, X(0) = 0

        and :math:`\\nu(T)` is a correction term to ensure that the initial value of :math:`w_{t,T}` is equal to 
        a given initial forecast :math:`\\bar{w}_{0,T}`. It is computed by 

        .. math:: \\nu(T) := \log\\frac{\\bar{w}_{0,T}}{(1.0-\\bar{w}_{0,T})}.

        The sigmoid function is applied to ensure that the forecast is between 0 and 1 (as percentage of total wind capacity)

        Args:
            speed_of_mean_reversion (float): The speed of mean reversion of the underlying Ornstein-Uhlenbeck process (:class:`rivapy.models.OrnsteinUhlenbeck`).
            volatility (float): The volatility of the underlying Ornstein-Uhlenbeck process (:class:`rivapy.models.OrnsteinUhlenbeck`).
            region (str): The name of the respective wind region.
        """
        #if len(expiries) != len(initial_forecasts):
        #    raise Exception('Number of forward expiries does not equal number of initial forecasts. Each forward expiry needs an own forecast.')
        self.ou = OrnsteinUhlenbeck(speed_of_mean_reversion, volatility, mean_reversion_level=0.0)
        self.region = region
        
        
    def _compute_ou_additive_forward_correction(self, expiries, initial_forecasts):
        result = np.empty((len(expiries)))
        for i in range(len(expiries)):
            #mean_ou = _inv_logit(self.ou.compute_expected_value(0.0, expiries[i]))
            correction = _logit(initial_forecasts[i]) #-self.ou.compute_expected_value(0.0, expiries[i])
            result[i] = correction
        return result

    def _to_dict(self)->dict:
        return {'speed_of_mean_reversion': self.ou.speed_of_mean_reversion, 
                'volatility': self.ou.volatility,
                'region': self.region}
        
    def get_forward(self, paths, ttm, ou_additive_forward_correction: float):
        expected_ou = self.ou.compute_expected_value(paths, ttm)#+correction
        return _inv_logit(expected_ou + ou_additive_forward_correction)
            
    def rnd_shape(self, n_sims: int, n_timesteps: int)->tuple:
        return (n_timesteps-1, n_sims)
    
    def simulate(self, timegrid, 
                rnd: np.ndarray,
                expiries: List[float],
                initial_forecasts: List[float],
                startvalue=0.0)->ForwardSimulationResult:
        #paths = np.empty((2,timegrid.shape[0], rnd.shape[2]))
        paths = self.ou.simulate(timegrid, startvalue, rnd)
        #paths[1,:,:] = self.ou.simulate(timegrid, startvalue, rnd[1])
        return WindPowerForecastModel.ForwardSimulationResult(paths, self, timegrid, expiries, initial_forecasts)

    def udls(self) -> Set[str]:
        return {self.region}

class MultiRegionWindForecastModel(BaseFwdModel):
    class ForwardSimulationResult(ForwardSimulationResult):
        def __init__(self, model, regions_result):
            self._results = regions_result
            self._model = model

        def n_forwards(self)->float:
            for v in self._results.values():
                return v.n_forwards()
            
        def udls(self)->Set[str]:
            return self._model.udls()

        def get(self, key: str, forecast_timepoints: List[int]=None) -> np.ndarray:
            udl = BaseFwdModel.get_udl_from_key(key)
            if udl == self._model.name:
                expiry = BaseFwdModel.get_expiry_from_key(key)
                return self._get(expiry, forecast_timepoints)
            for v in self._results.values():
                if udl in v.udls():
                    return v.get(key, forecast_timepoints)
            raise Exception(f'Cannot find key {key}')

        def _get(self, expiry: int, forecast_timepoints: List[int])->np.ndarray:
            result = None
            for udl in self.udls():
                if udl == self._model.name:
                    continue
                if result is None:
                    result = self._model.region_relative_capacity(udl)*np.copy(self.get(BaseFwdModel.get_key(udl, expiry), forecast_timepoints))
                else:
                    result += self._model.region_relative_capacity(udl)*np.copy(self.get(BaseFwdModel.get_key(udl, expiry), forecast_timepoints))
            return result
        
    class Region(FactoryObject):
        def __init__(self, model: WindPowerForecastModel,  capacity: float, rnd_weights: List[float]):
            self.model = _create(model)
            self.capacity  = capacity
            self.rnd_weights = rnd_weights

        def name(self):
            return self.model.region

        def n_random(self):
            return len(self.rnd_weights)

        def udls(self)->Set[str]:
            return self.model.udls()

        def _to_dict(self)->dict:
            return {'model': self.model.to_dict(), 'capacity': self.capacity, 'rnd_weights': self.rnd_weights}

    def __init__(self, name: str, region_forecast_models: List[Region]):
        """Simple model to simulate power forecasts for more then one wind region and a total efficiency over all regions.

        This model relates the wind models for the different regions within the simulation by just using a linear sum of normal random variates, i.e.
        for a region :math:`r` we have weights :math:`w_{r,i}`,  :math:`1\leq i\leq N`, so that within the simulation based on :math:`n` different normal random
        variates :math:`X_i` we compute the random variate for the region by

        .. math:: X_{r,i} = \\frac{\sum_i w_{r,i} X_i}{\sqrt{\sum_i w_{r,i}^2}}.

        Args:
            name (str): Name of the overall region.
            region_forecast_models (List[Region]): List of regions that will be simulated.

        Examples:
            >>> wind_onshore = WindPowerForecastModel(region='Onshore', speed_of_mean_reversion=0.1, volatility=4.80)
            >>> wind_offshore = WindPowerForecastModel(region='Offshore', speed_of_mean_reversion=0.5, volatility=4.80)
            >>> regions = [ MultiRegionWindForecastModel.Region( 
                                    wind_onshore,
                                    capacity=1000.0,
                                    rnd_weights=[0.8,0.2]
                                ),
                            MultiRegionWindForecastModel.Region( 
                                                        wind_offshore,
                                                        capacity=100.0,
                                                        rnd_weights=[0.2,0.8]
                                                    )
                            ]
            >>> wind = MultiRegionWindForecastModel('Wind_Germany', regions)
            >>> # after model setup we simulate forecasts
            >>> days = 10
            >>> timegrid = np.linspace(0.0, days*1.0/365.0, days*24)
            >>> forward_expiries = [timegrid[-1] + i/(365.0*24.0) for i in range(4)]
            >>> rnd = np.random.normal(size=wind.rnd_shape(n_sims, timegrid.shape[0]))
            >>> results = wind.simulate(timegrid, rnd, expiries=forward_expiries, 
                                       initial_forecasts={'Onshore': [0.8, 0.7,0.6,0.5],
                                                          'Offshore': [0.6, 0.6, 0.5, 0.5]}
                               )
        """
        self.name = name
        if not region_forecast_models:
            raise Exception('Empty list of models is not allowed')
        self._region_forecast_models = [ _create(r) for r in region_forecast_models]
        n_rnd_ref_model = self._region_forecast_models[0].n_random()
        for i in range(1, len(self._region_forecast_models)):
            if self._region_forecast_models[i].n_random() != n_rnd_ref_model:
                raise Exception('All regions must have the same number of random variables.')
        
        

    def _to_dict(self)->dict:
        return {'name': self.name, 
                'region_forecast_models':[v.to_dict() for v in self._region_forecast_models]
                }

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
        raise Exception(f'Model does not contain a region with name {region}')

    def region_names(self)->List[str]:
        return [r.name() for r in self._region_forecast_models]

    def simulate(self, timegrid, rnd: np.ndarray, expiries: List[float],
                initial_forecasts: Dict[str,List[float]], startvalue=0.0):
        results = {}
        for region in self._region_forecast_models:
            rnd_ = region.rnd_weights[0]*rnd[0,:,:]
            for i in range(1, region.n_random()):
                rnd_ += region.rnd_weights[i]*rnd[i,:,:]
            results[region.name()] = region.model.simulate(timegrid, rnd_, expiries, initial_forecasts[region.name()], startvalue)
        return MultiRegionWindForecastModel.ForwardSimulationResult(self, results)

    def rnd_shape(self, n_sims: int, n_timesteps: int)->tuple:
        return (len(self._region_forecast_models), n_timesteps-1, n_sims)

    def udls(self) -> Set[str]:
        result = {self.name}
        for v in self._region_forecast_models:
            result.update(v.udls())
        return result

class ResidualDemandForwardModel(BaseFwdModel):
    class ForwardSimulationResult(ForwardSimulationResult):
        def __init__(self, model, highest_price, wind_results):
            self._model = model
            self._highest_price = highest_price
            self._wind = wind_results
            
        def n_forwards(self)->float:
            return self._wind.n_forwards()
            
        def udls(self)->Set[str]:
            return self._model.udls()

        def get(self, key: str, forecast_timepoints: List[int]) -> np.ndarray:
            udl = BaseFwdModel.get_udl_from_key(key)
            if udl != self._model.power_name:
                return self._wind.get(key, forecast_timepoints)
            expiry = BaseFwdModel.get_expiry_from_key(key)
            return self._get(expiry, forecast_timepoints)

        def _get(self, expiry: int, forecast_timepoints: List[int])->np.ndarray:
            total_produced = self._wind.get(BaseFwdModel.get_key(self._wind._model.name, expiry), forecast_timepoints)
            power_fwd = np.empty((total_produced.shape[0], total_produced.shape[1]))
            for i in range(total_produced.shape[0]):
                power_fwd[i,:] =  self._model.supply_curve(1.0-total_produced[i,:] )*self._highest_price[i,:]
            return power_fwd
        
    def __init__(self, wind_power_forecast,
                        highest_price_ou_model, 
                        supply_curve: Callable[[float], float],
                        max_price: float,
                        forecast_hours: List[int]=None,
                        power_name:str = None):
        #print(wind_power_forecast)
        self.wind_power_forecast = _create(wind_power_forecast)
        self.highest_price_ou_model = _create(highest_price_ou_model)
        self.supply_curve = _create(supply_curve)
        self.forecast_hours = forecast_hours
        self.max_price = max_price
        self.power_name = power_name if power_name is not None else 'POWER'    
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

    def simulate(self, timegrid: np.ndarray, 
                rnd: np.ndarray, 
                expiries: List[float],
                initial_forecasts: Dict[str, List[float]]):
        highest_prices = self.highest_price_ou_model.simulate(timegrid, 1.0, rnd[0,:])*self.max_price
        simulated_wind = self.wind_power_forecast.simulate(timegrid, rnd[1:,:], expiries, initial_forecasts)
        return ResidualDemandForwardModel.ForwardSimulationResult(self, highest_prices, simulated_wind)
    
    def udls(self)->Set[str]:
        result = self.wind_power_forecast.udls()
        result.add(self.power_name)
        return result
        
    

if __name__=='__main__':
    from rivapy.models.residual_demand_model import MultiRegionWindForecastModel, SmoothstepSupplyCurve
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
