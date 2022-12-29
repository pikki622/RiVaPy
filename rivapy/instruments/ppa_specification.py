from typing import Union, Tuple, Iterable, Set, List
import abc
import numpy as np
import pandas as pd
import datetime as dt
import rivapy.tools.interfaces as interfaces
from rivapy.instruments.factory import create as _create


class SimpleSchedule(interfaces.FactoryObject):
	def __init__(self, start: dt.datetime, end:dt.datetime, 
					freq: str='1H', weekdays: Set[int] = None, 
					hours: Set[int] = None, tz: str=None ):
		"""Simple schedule of fixed datetime points.

		Args:
			start (dt.datetime): Start of schedule (including this timepoint).
			end (dt.datetime): End of schedule (excluding this timepoint).
			freq (str, optional): Frequency of timepoints. Defaults to '1H'. See documentation for pandas.date_range for further details on freq.
			weekdays (Set[int], optional): List of integers representing the weekdays where the schedule is defined. 
											Integers according to datetime weekdays (0->Monay, 1->Tuesday,...,6->Sunday). 
											If None, all weekdays are used. Defaults to None.
			hours (Set[int], optional): List of hours where schedule is defined. If None, all hours are included. Defaults to None.
			tz (str or tzinfo): Time zone name for returning localized datetime points, for example ‘Asia/Hong_Kong’. 
								By default, the resulting datetime points are timezone-naive. See documentation for pandas.date_range for further details on tz.
		Examples:
		
		.. highlight:: python
		.. code-block:: python

			>>> simple_schedule = SimpleSchedule(dt.datetime(2023,1,1), dt.datetime(2023,1,1,4,0,0), freq='1H')
			>>> simple_schedule.get_schedule()
			[datetime(2023,1,1,0,0,0), datetime(2023,1,1,1,0,0), datetime(2023,1,1,2,0,0), datetime(2023,1,1,3,0,0)]

			# We include only hours 2 and 3 into schedule
			>>> simple_schedule = SimpleSchedule(dt.datetime(2023,1,1), dt.datetime(2023,1,1,4,0,0), freq='1H', hours=[2,3])
			>>> simple_schedule.get_schedule()
			[datetime.datetime(2023, 1, 1, 2, 0), datetime.datetime(2023, 1, 1, 3, 0)]

			# We restrict further to only mondays as weekdays included
			>>> simple_schedule = SimpleSchedule(dt.datetime(2023,1,1), dt.datetime(2023,1,2,4,0,0), freq='1H', hours=[2,3], weekdays=[0])
			>>> simple_schedule.get_schedule()
			[datetime.datetime(2023, 1, 2, 2, 0), datetime.datetime(2023, 1, 2, 3, 0)]


		"""
		self.start = start
		self.end = end
		self.freq = freq
		self.weekdays = weekdays
		self.hours = hours
		self.tz = tz
		self._df = None

	def _to_dict(self)->dict:
		return {
					'start': self.start, 'end':self.end, 
					'freq': self.freq, 'weekdays': self.weekdays, 
					'hours': self.hours, 'tz': self.tz
		}
 
	def get_schedule(self, refdate: dt.datetime = None)->List[dt.datetime]:
		"""Return list of datetime values belonging to the schedule.

		Args:
			refdate (dt.datetime): All schedule dates are ignored before this reference dats. If None, all schedule dates are returned. Defaults to None.

		Returns:
			List[dt.datetime]: List of all datetimepoints of the schedule.
		"""
		d_ = pd.date_range(self.start, self.end, freq=self.freq, tz=self.tz, closed='left').to_pydatetime()
		if self.weekdays is not None:
			d_ = [d for d in d_ if d.weekday() in self.weekdays]
		if self.hours is not None:
			d_ = [d for d in d_ if d.hour in self.hours]
		if refdate is not None:
			d_ = [d for d in d_ if d >= refdate]
		return d_

	def get_df(self)->pd.DataFrame:
		if self._df is None:
			self._df = pd.DataFrame({'dates': pd.date_range(self.start, self.end, freq=self.freq, tz=self.tz, closed='left').to_pydatetime()}).reset_index()
		return self._df

	def get_params(self)->dict:
		"""Return all params as json serializable dictionary.

		Returns:
			dict: Dictionary of all parameters.
		"""
		return {'start': self.start, 'end': self.end, 'freq': self.freq, 'weekdays': self.weekdays, 'hours': self.hours, 'tz': self.tz}

class PPASpecification(interfaces.FactoryObject):
	def __init__(self, 
				amount: Union[float, np.ndarray], 
				schedule: Union[SimpleSchedule, List[dt.datetime]],
				fixed_price: float,
				id:str = None):
		"""Specification for a simple power purchase agreement (PPA).

		Args:
			amount (Union[None, float, np.ndarray]): Amount of power delivered at each timepoint/period. Either a single value s.t. all volumes delivered are constant or a load table.
			schedule (Union[SimpleSchedule, List[dt.datetime]): Schedule describing when power is delivered.
			fixed_price (float): The fixed price paif for the power.
			id (str): Simple id of the specification. If None, a uuid will be generated. Defaults to None.
		"""
		self.id = id
		if id is None:
			self.id = type(self).__name__+'/'+str(dt.datetime.now())
		self.amount = amount
		if isinstance(schedule, dict): #if schedule is a dict we try to create it from factory
			self.schedule = _create(schedule)
		else:
			self.schedule = schedule
		self.fixed_price = fixed_price
		if isinstance(schedule, list):
			self._schedule_df = pd.DataFrame({'dates': self.schedule}).reset_index()
		else:
			self._schedule_df = self.schedule.get_df().set_index(['dates']).sort_index()
		self._schedule_df['amount'] = amount
		self._schedule_df['flow'] = None

	def _to_dict(self)->dict:
		try: # if isinstance(self.schedule, interfaces.FactoryObject):
			schedule = self.schedule.to_dict()
		except  Exception as e:
			schedule = self.schedule
		return {
			'id': self.id,
			'amount': self.amount,
			'schedule': schedule,
			'fixed_price': self.fixed_price
		}


	def set_amount(self, amount):
		self.amount = amount
		self._schedule_df['amount'] = amount

	def compute_flows(self, refdate: dt.datetime, pfc, result: pd.DataFrame=None, result_col = None)->pd.DataFrame:
		df = pfc.get_df()
		if result is None:
			self._schedule_df['flow'] = self._schedule_df['amount']*(df.loc[self._schedule_df.index]['values']-self.fixed_price)
			return self._schedule_df
		result[result_col] = self._schedule_df['amount']*(df.loc[self._schedule_df.index]['values']-self.fixed_price)

class GreenPPASpecification(PPASpecification):
	def __init__(self,schedule: Union[SimpleSchedule, List[dt.datetime]],
				technology: str,
				fixed_price: float,
				id:str = None):
		super().__init__(None, schedule, fixed_price, id)
		self.technology = technology

	def _to_dict(self)->dict:
		result = super()._to_dict()
		del result['amount']
		result['technology'] = self.technology
		return result

	def compute_flows(self, refdate: dt.datetime, pfc, forecast_amount: np.ndarray, result: pd.DataFrame=None, result_col = None)->pd.DataFrame:
		self.set_amount(forecast_amount)
		return super().compute_flows(refdate, pfc, result, result_col)

