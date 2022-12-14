import numpy as np
import datetime as dt

    
class GasStorageSpecification:
    def __init__(self, timegrid, # wie heißt das bei Zafir?
                min_volume: float, max_volume: float, 
                start_volume: float, end_volume: float, 
                max_withdrawal: float, max_injection: float,
                withdrawal_cost: float = 0.0, 
                injection_cost: float = 0.0
                ):
        """_summary_

        Args:
            timegrid (_type_): _description_
            max_volume (float): _description_
            start_volume (float): _description_
            end_volume (float): _description_
            max_withdrawal (float): _description_
            max_injection (float): _description_
            withdrawal_cost (float, optional): _description_. Defaults to 0.0.
            injection_cost (float, optional): _description_. Defaults to 0.0.
        """
        self.timegrid = timegrid
        self.min_volume = min_volume
        self.max_volume = max_volume
        self.start_volume = start_volume
        self.end_volume = end_volume
        self.max_withdrawal = max_withdrawal
        self.max_injection = max_injection
        self.withdrawal_cost = withdrawal_cost
        self.injection_cost = injection_cost