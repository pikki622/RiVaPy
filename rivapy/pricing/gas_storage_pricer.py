import numpy as np
import datetime as dt
from typing import Callable, Union

from rivapy.instruments.gasstorage_specification import GasStorageSpecification

class _PolynomialRegressionFunction:
    def __init__(self, deg: int):
        self.deg = deg

    def fit(self, S: np.array, C: np.array):
        return np.polyfit(S, C, self.deg)

    def predict(self, R: np.array, S: np.array):
        return np.polyval(R, S)

class PricingParameter:
    def __init__(self, n_time_steps:int, n_actions: int, n_vol_levels: int, regression: object = _PolynomialRegressionFunction):
        self.n_time_steps = n_time_steps    
        self.n_actions = n_actions         
        self.n_vol_levels = n_vol_levels
        self.regression = regression

def _payoff_func(S, delta_v, a1=0, a2=0, b1=0, b2=0, action=1):
    #payoff function h (with bid-ask spread & transaction cost)
    # -c*delta_v     inject         c=(1+a1)S +b1
    # 0              do nothing
    # -p*delta_v     withdraw       p=(1-a2)S -b2
    
    #if a1=a2=b1=b2=0 -> c=p=S
    if a1==0 and a2==0 and b1==0 and b2==0:
        return -S*delta_v
    else:
        if action == 1: #inject
            return -((1+a1)*S + b1) * delta_v
        elif action == -1: #withdraw
            return -((1-a2)*S - b2) * delta_v
        else: #do nothing
            return 0

def _penalty_func(S, v): #final condition
    def indicator_function(v, y):
        return np.array([0 if x == y else 1 for x in v])
    # v(T)=0 should be satisfied, else: penalty: S(T)*100_000
    return -100_000 * S * indicator_function(v, 0) 

def pricing_lsmc(storage: GasStorageSpecification, 
                pricing_parameters: PricingParameter,
                prices: np.ndarray, 
                nb_sims: int,
                penalty_func: Callable = _penalty_func) -> Union[np.ndarray, np.ndarray]:
    """ Least-Squares Monte Carlo Method for Pricing the Gas Storage

    Args:
        storage (GasStorageSpecification): the specification 
        pricing_parameters (PricingParameter): the parameters
        prices (np.ndarray): the prices
        nb_sims (int): number of simulations

    Returns:
        np.ndarray: the accumulated cash flows
        np.ndarray: the optimal volume levels
    """
    
    #discretization of possible volume levels
    v = np.linspace(storage.min_level, storage.storage_capacity, pricing_parameters.n_vol_levels) 
    
    # Assign a value to the contract at maturity according to the final condition
    acc_cashflows = np.empty((len(storage.timegrid)+1, len(v), nb_sims))

    # For t=T+1: final condition
    for i in range(nb_sims):    
        acc_cashflows[-1,:,i] = penalty_func(prices[-1,i], v) #size v
    
    # Apply backward induction for t=T...1 
    # For each t, step over N allowed volume levels v(t,n) 
    regression = pricing_parameters.regression(deg=2)
    total_vol_levels = np.empty((len(storage.timegrid), len(v), nb_sims))
    for t in range(len(storage.timegrid),1,-1):
        for vol_t in range(len(v)):
            dec_func = np.empty((len(v), nb_sims))
            for vol_tplus1 in range(len(v)): #for all volumes by themselves
                # - Run an OLS regression to find an approx. of the continuation value
                cont_val = acc_cashflows[t,vol_tplus1,:] # assumed cont. value for t+1, no disc. factor
                cv_fit = regression.fit(prices[t-1,:], cont_val)
                cv_pred = regression.predict(cv_fit, prices[t-1,:])

                # - Combine the cont. values C into a decision rule for each volume level
                max_withdraw = max(storage.withdrawal_rate, -(v[vol_t]-storage.min_level))
                max_inject = min(storage.injection_rate, storage.storage_capacity - v[vol_t])
                
                dv = v[vol_tplus1] - v[vol_t]

                # check against constraints to find achievable actions:
                if dv >= max_withdraw and dv <= max_inject:
                    dec_func[vol_tplus1,:] = _payoff_func(prices[t-1,:],dv) + cv_pred
                else:
                    dec_func[vol_tplus1, :] = -1e12

            #argmax for decision rule for specified volume level at time step t+1
            ind = np.argmax(dec_func, axis=0)
            #dv_max = v[vol_t]*np.ones(nb_sims) - v[ind] #size nb_sims
            total_vol_levels[t-1, vol_t,:] = ind

            # - Calculate the accumulated future cash flows Y^b
            acc_cashflows[t-1,vol_t,:] = np.max(dec_func, axis=0) #no disc. factor

    # "extrapolate" to first level
    acc_cashflows[0,:,:] = acc_cashflows[1,:,:]
    total_vol_levels[0,:,:] = total_vol_levels[1,:,:]
    
    #forward sweep for optimal path
    ind_level = np.empty((len(storage.timegrid), nb_sims))
    total_volume = np.empty((len(storage.timegrid), nb_sims))
    ind_level[0,:] = 0 #on the grid, index of startLevel = 0
    for t in range(1,len(storage.timegrid)):
        for m in range(nb_sims):
            ind_level[t,m] = total_vol_levels[t-1,int(ind_level[t-1,m]),m]
            total_volume[t,m] = v[int(ind_level[t,m])]

    return acc_cashflows, total_volume