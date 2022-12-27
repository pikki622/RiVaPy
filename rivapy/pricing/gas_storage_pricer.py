import numpy as np
import datetime as dt
from typing import Callable

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

def old_price_lsmc(storage: GasStorageSpecification, 
                pricing_parameters: PricingParameter,
                prices: np.ndarray):
    """ Least-Squares Monte Carlo Method for Pricing the Gas Storage

    Args:
        storage (GasStorageSpecification): the specification 
        pricing_parameters (PricingParameter): the parameters
        prices (np.ndarray): the prices

    Returns:
        _type_: _description_
    """
    
    #discretization of possible volume levels
    v = np.linspace(min_volume, max_volume, pricing_parameters.n_vol_levels) 
    
    # Assign a value to the contract at maturity according to the final condition
    acc_cashflows = np.empty((len(storage.timegrid)+1, len(v), M)) #For t=T+1: final condition
    #t = self.timegrid[-1]+dt.timedelta(days=1)

    for i in range(M):    
        acc_cashflows[-1,:,i] = _penalty_func(prices[-1,i], v) #size v
    
    # Apply backward induction for t=T...1 
    # For each t, step over N allowed volume levels v(t,n) 
    for t in range(len(storage.timegrid),1,-1):

        for vol_tplus1 in range(len(v)): #for all volumes by themselves
                    
            # - Run an OLS regression to find an approx. of the continuation value
            cont_val = acc_cashflows[t,vol_tplus1,:] # assumed cont. value for t+1, no disc. factor
            regression = pricing_parameters.regression(deg=2)
            cv_fit = regression.fit(prices[t-1,:], cont_val) #regression = np.polyfit(S[:,t-1], cont_val, deg=5)
            cv_pred = regression.predict(cv_fit, prices[t-1,:]) #cv_pred = np.polyval(regression, S[:,t-1])
            
            # - Combine the cont. values C into a decision rule for each volume level
            max_withdraw = max(storage.max_withdrawal, storage.min_volume - v[vol_tplus1])
            max_inject = min(storage.max_injection, storage.min_volume - v[vol_tplus1])

            dec_func = np.empty((len(v), M))
            for vol_t in range(len(v)):
                dv = v[vol_tplus1] - v[vol_t]

                # check against constraints to find achievable actions:
                if dv > max_withdraw or dv < max_inject:
                    dec_func[vol_t, :] = -1e12

                dec_func[vol_t,:] = _payoff_func(prices[t-1,:],dv) + cv_pred

            #argmax for decision rule for specified volume level at time step t+1
            ind = np.argmax(dec_func, axis=0)
            vol_t_max = v[ind] #size M
            dv_max = v[vol_tplus1]*np.ones(M) - vol_t_max
        
            # - Calculate the accumulated future cash flows Y^b
            acc_cashflows[t-1,vol_tplus1,:] = _payoff_func(prices[t-1,:], dv_max) + acc_cashflows[t, vol_tplus1, :] #no disc. factor

    return acc_cashflows

def pricing_lsmc(storage: GasStorageSpecification, 
                pricing_parameters: PricingParameter,
                penalty_func: Callable,
                prices: np.ndarray, nb_sims: int) -> np.ndarray:
    """ Least-Squares Monte Carlo Method for Pricing the Gas Storage

    Args:
        storage (GasStorageSpecification): the specification 
        pricing_parameters (PricingParameter): the parameters
        prices (np.ndarray): the prices
        nb_sims (int): number of simulations

    Returns:
        np.ndarray: the accumulated cash flows
    """
    
    #discretization of possible volume levels
    v = np.linspace(storage.min_level, storage.storage_capacity, pricing_parameters.n_vol_levels) 
    
    # Assign a value to the contract at maturity according to the final condition
    acc_cashflows = np.empty((len(storage.timegrid)+1, len(v), nb_sims)) #For t=T+1: final condition
    #t = self.timegrid[-1]+dt.timedelta(days=1)

    for i in range(nb_sims):    
        acc_cashflows[-1,:,i] = penalty_func(prices[-1,i], v) #size v

    # Apply backward induction for t=T...1 
    # For each t, step over N allowed volume levels v(t,n) 
    regression = pricing_parameters.regression(deg=2)
    for t in range(len(storage.timegrid),1,-1):
        for vol_t in range(len(v)): #for all volumes by themselves
            dec_func = np.empty((len(v), nb_sims))
            for vol_tplus1 in range(len(v)):
                # - Run an OLS regression to find an approx. of the continuation value
                cont_val = acc_cashflows[t,vol_tplus1,:] # assumed cont. value for t+1, no disc. factor
                cv_fit = regression.fit(prices[t-1,:], cont_val) #regression = np.polyfit(S[:,t-1], cont_val, deg=5)
                cv_pred = regression.predict(cv_fit, prices[t-1,:]) #cv_pred = np.polyval(regression, S[:,t-1])
            
                # - Combine the cont. values C into a decision rule for each volume level
                max_withdraw = max(storage.withdrawal_rate, -(v[vol_t]-storage.min_level))
                max_inject = min(storage.injection_rate, storage.storage_capacity - v[vol_t])

                dv = v[vol_tplus1] - v[vol_t]

            # check against constraints to find achievable actions:
            if dv >= max_withdraw or dv <= max_inject:
                dec_func[vol_tplus1,:] = _payoff_func(prices[t-1,:],dv) + cv_pred
            else:
                dec_func[vol_tplus1, :] = -1e12
                
        #argmax for decision rule for specified volume level at time step t+1
        ind = np.argmax(dec_func, axis=0)
        vol_t_max = v[ind] #size M
        #dv_max = v[vol_tplus1]*np.ones(M) - vol_t_max
    
        # - Calculate the accumulated future cash flows Y^b
        acc_cashflows[t-1,vol_t,:] = np.max(dec_func, axis=0)#_payoff_func(prices[t-1,:], dv_max) + acc_cashflows[t, vol_tplus1, :] #no disc. factor

    return acc_cashflows

class GasStorageLSMC:
    #TODO: Remove this class
    def __init__(self, timegrid, min_volume, max_volume, 
                start_volume, end_volume, 
                max_withdrawal, max_injection):
        self.timegrid = timegrid
        self.min_volume = min_volume
        self.max_volume = max_volume
        self.start_volume = start_volume
        self.end_volume = end_volume
        self.max_withdrawal = max_withdrawal
        self.max_injection = max_injection

    def pricing(self, S, M, penalty_func, pricing_parameters: PricingParameter):
            
        #discretization of possible volume levels
        v = np.linspace(min_volume, max_volume, pricing_parameters.n_vol_levels) 
        
        # Assign a value to the contract at maturity according to the final condition
        acc_cashflows = np.empty((len(self.timegrid)+1, len(v), M)) #For t=T+1: final condition
        #t = self.timegrid[-1]+dt.timedelta(days=1)

        for i in range(M):    
            acc_cashflows[-1,:,i] = penalty_func(S[-1,i], v) #size v
        
        regression = pricing_parameters.regression(deg=2)
        # Apply backward induction for t=T...1 
        # For each t, step over N allowed volume levels v(t,n) 
        for t in range(len(self.timegrid),1,-1):
            for vol_t in range(len(v)):
                dec_func = np.empty((len(v), M))
                for vol_tplus1 in range(len(v)): #for all volumes by themselves
                    # - Run an OLS regression to find an approx. of the continuation value
                    cont_val = acc_cashflows[t,vol_tplus1,:] # assumed cont. value for t+1, no disc. factor
                    cv_fit = regression.fit(S[t-1,:], cont_val) #regression = np.polyfit(S[:,t-1], cont_val, deg=5)
                    cv_pred = regression.predict(cv_fit, S[t-1,:]) #cv_pred = np.polyval(regression, S[:,t-1])
                    
                    # - Combine the cont. values C into a decision rule for each volume level
                    max_withdraw = max(self.max_withdrawal, -(v[vol_t]-self.min_volume))
                    max_inject = min(self.max_injection, self.max_volume - v[vol_t])
                    
                    dv = v[vol_tplus1] - v[vol_t]

                    # check against constraints to find achievable actions:
                    if dv >= max_withdraw and dv <= max_inject:
                        dec_func[vol_tplus1,:] = _payoff_func(S[t-1,:],dv) + cv_pred
                    else:
                        dec_func[vol_tplus1, :] = -1e12

                #argmax for decision rule for specified volume level at time step t+1
                ind = np.argmax(dec_func, axis=0)
                vol_t_max = v[ind] #size M
                #dv_max = v[vol_t]*np.ones(M) - vol_t_max
                # - Calculate the accumulated future cash flows Y^b
                acc_cashflows[t-1,vol_t,:] = np.max(dec_func, axis=0)#_payoff_func(S[t-1,:], dv_max) + cv_pred#dec_func[:,ind] #acc_cashflows[t, vol_t, :] #no disc. factor

        return acc_cashflows

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
    
    gas = GasStorageLSMC(contractdates, min_volume, max_volume, start_volume, end_volume, max_withdrawal, max_injection)
    gas_cashflows = gas.pricing(gbm, M, penalty_func=_penalty_func, pricing_parameters=params)
    #store = GasStorageSpecification(contractdates, max_volume, max_withdrawal, max_injection)   
    #gas_cashflows = pricing_lsmc(store, params, _penalty_func, gbm, M)
    
    #avg_gas_cashflow = np.average(gas_cashflows, axis=2)

    if True:

        plt.figure()
        plt.plot(gbm)
        plt.show()

        for i in range(n_vol_levels):
            plt.figure(figsize=(12,8))
            plt.plot(gas_cashflows[:-20,i,0])
            plt.show()
            #if i==1:
            #    break        
