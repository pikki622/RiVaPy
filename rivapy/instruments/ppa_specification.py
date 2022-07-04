from typing import Union, Tuple, Iterable, Set, List
import abc
import numpy as np
import pandas as pd
import datetime as dt


class SimpleSchedule:
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

	def get_schedule(self)->List[dt.datetime]:
		"""Return list of datetime values belonging to the schedule.

		Returns:
			List[dt.datetime]: List of all datetimepoints of the schedule.
		"""
		d_ = pd.date_range(self.start, self.end, freq=self.freq, tz=self.tz, closed='left').to_pydatetime()
		if self.weekdays is not None:
			d_ = [d for d in d_ if d.weekday() in self.weekdays]
		if self.hours is not None:
			d_ = [d for d in d_ if d.hour in self.hours]
		return d_

class PPASpecification:
	def __init__(self, 
				amount: Union[float, np.ndarray], 
				schedule: Union[SimpleSchedule, List[dt.datetime]]):
		"""Specification for a power purchase agreement (PPA).

		Args:
			amount (Union[None, float, np.ndarray]): Amount of power delivered at each timepoint/period. Either a single value s.t. all volumes delivered are constant or a load table.
			schedule (Union[SimpleSchedule, List[dt.datetime]): Schedule describing when power is delivered.
		"""
		self.amount = amount
		self.schedule = schedule