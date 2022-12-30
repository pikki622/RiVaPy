import numpy as np
import datetime as dt
from typing import Callable, Union

from rivapy.instruments.gasstorage_specification import GasStorageSpecification

class PricingParameter:
    def __init__(self, n_time_steps:int, n_actions: int, n_vol_levels: int, regression):
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

class _PolynomialRegressionFunction:
    def __init__(self, deg: int):
        self.deg = deg

    def fit(self, S: np.array, C: np.array):
        return np.polyfit(S, C, self.deg)

    def predict(self, R: np.array, S: np.array):
        return np.polyval(R, S)

def _penalty_func(S, v): #final condition
    def indicator_function(v, y):
        return np.array([0 if x == y else 1 for x in v])
    # v(T)=0 should be satisfied, else: penalty: S(T)*100_000
    return -100_000 * S * indicator_function(v, 0) 

def pricing_lsmc(storage: GasStorageSpecification, 
                pricing_parameters: PricingParameter,
                penalty_func: Callable,
                prices: np.ndarray, 
                nb_sims: int) -> Union[np.ndarray, np.ndarray]:
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

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def create_contract_dates(startdate: dt.datetime, enddate: dt.datetime, datestep:dt.timedelta)->list:
        dates=[startdate]
        while dates[-1] <= enddate-datestep:
            dates.append(dates[-1]+datestep)
        
        #dates=[startdate]*n #nb_timesteps 
        #for i in range(1,n):
        #    dates[i] = dates[i-1] + dateStep
        
        return dates

    class GeometricBrownianMotionSimulator:

        """Simulate a 1D Geometric Brownian Motion for a datetime timegrid"""

        def __init__(self, timegrid: list, mu: float, sigma: float):
            
            self.timegrid = timegrid
            self.mu = mu
            self.sigma = sigma

        def create_gbm(self, X0: float, seed=None) -> np.array:

            if seed is not None:
                np.random.seed(seed)

            dtt = []
            for i in range(len(self.timegrid)-1):
                dti = self.timegrid[i+1] - self.timegrid[i]
                dtt.append(dti.days/365.0)
            dt = np.array(dtt)
            rnd = np.random.normal(size=(len(self.timegrid)-1))
            Y = np.exp((self.mu - self.sigma**2 / 2) * dt + self.sigma * np.sqrt(dt) * rnd)
            R = X0 * np.cumprod(Y) 
            return np.insert(R, 0, X0) #add start value X0

    ## Setting the parameters
    nomination = 1 #daily nomination
    #n = 365 # one year to maturity
    M = 20 #number of independent price paths simulated
    S0 = 1.0 #starting value
    #kappa = 0.05
    sigma = 0.00945
    mu = 0.2
    
    n_vol_levels = 11 #101
    min_volume = 0.0
    max_volume = 10.0 #1_000
    start_volume = 0.0 #100_000
    end_volume = 0.0 #100_000
    max_withdrawal = -1.0 #-7500
    max_injection = 1.0 #2500

    startdate = dt.datetime.fromisoformat('2021-01-01')
    enddate = dt.datetime.fromisoformat('2021-12-31')
    dateStep = dt.timedelta(days=nomination)
    contractdates = create_contract_dates(startdate, enddate, dateStep)
    #fwd_times = [(date - contractdates)/n for date in contractDates]

    # Simulate M independent price paths S^b(1), S^b(T+1) for b = 1...M starting at S(0)
    gbm_sim = GeometricBrownianMotionSimulator(contractdates, mu, sigma)
    gbm = np.empty((len(contractdates), M))
    for i in range(M):
        gbm[:,i] = gbm_sim.create_gbm(S0, seed=i) 

    params = PricingParameter(n_time_steps = 0, n_actions = 0, n_vol_levels = n_vol_levels, regression = _PolynomialRegressionFunction)
    
    store = GasStorageSpecification(contractdates, max_volume, max_withdrawal, max_injection)  
    gas_cashflows, vol_levels = pricing_lsmc(store, params, _penalty_func, gbm, M)
    
    #avg_gas_cashflow = np.average(gas_cashflows, axis=2)

    if True: 
        plt.figure()
        plt.plot(gbm)
        plt.show()

        plt.figure(figsize=(12,8))
        plt.plot(vol_levels)
        plt.show()
        
        for i in range(n_vol_levels):
            plt.figure(figsize=(12,8))
            plt.plot(gas_cashflows[:-10,i,:])
            plt.show()

