
import numpy as np
from numpy.core.fromnumeric import var
from rivapy.models.factory import _factory 
from rivapy.models.local_vol import LocalVol
from rivapy.models.heston import HestonModel
from rivapy.models.stoch_local_vol import StochasticLocalVol
from rivapy.models.scott_chesney import ScottChesneyModel
from rivapy.models.ornstein_uhlenbeck import OrnsteinUhlenbeck
from rivapy.models.residual_demand_model import ResidualDemandModel,  WindPowerModel, SolarPowerModel, SupplyFunction, LoadModel, SmoothstepSupplyCurve
from rivapy.models.residual_demand_fwd_model import WindPowerForecastModel, ResidualDemandForwardModel, MultiRegionWindForecastModel

def _add_to_factory(cls):
    factory_entries = _factory()
    factory_entries[cls.__name__] = cls

_add_to_factory(OrnsteinUhlenbeck)
_add_to_factory(SupplyFunction)
_add_to_factory(SmoothstepSupplyCurve)
_add_to_factory(WindPowerForecastModel)
_add_to_factory(ResidualDemandForwardModel)
_add_to_factory(MultiRegionWindForecastModel.Region)
_add_to_factory(MultiRegionWindForecastModel)