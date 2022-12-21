import numpy as np

class GasStorageSpecification:
    def __init__(self, 
                timegrid: np.array, # wie heißt das bei Zafir? -> objID, startP & endP
                storageCapacity: float, 
                withdrawalRate: float, 
                injectionRate: float, 
                withdrawalCost: float, 
                injectionCost: float,
                minLevel: float = 0.0,
                startLevel: float = 0.0, #additional (storageLevel?)
                endLevel: float = 0.0  #additional (payoff?)
                ):
        """Constructor for gas storage specification

        Args:
            timegrid (np.array): Array of timesteps (between 0 and 1).
            storageCapacity (float): Maximum possible level for the gas storage.
            withdrawalRate (float): Maximum withdrawal rate.
            injectionRate (float): Maximum injection rate.
            withdrawalCost (float): Relative cost of withdrawal.
            injectionCost (float): Relative cost of injection.
            minLevel (float, optional): Minimum level for the gas storage. Defaults to 0.0.
            startLevel (float, optional): Start level for the gas storage. Defaults to 0.0.
            endLevel (float, optional): End level for gas storage. Defaults to 0.0.
        """
        
        self.timegrid = timegrid
        self.minLevel = minLevel
        self.storageCapacity = storageCapacity
        self.startLevel = startLevel
        self.endLevel = endLevel
        self.withdrawalRate = withdrawalRate
        self.injectionRate = injectionRate
        self.withdrawalCost = withdrawalCost
        self.injectionCost = injectionCost