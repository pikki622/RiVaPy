# -*- coding: utf-8 -*-

from datetime import datetime, date
from typing import List as _List, Tuple as _Tuple, Union as _Union
from holidays import  HolidayBase as _HolidayBase, CountryHoliday as _CountryHoliday
from holidays.utils import list_supported_countries as _list_supported_countries
from rivapy.tools.enums import DayCounter, RollConvention, SecuritizationLevel

# from iso4217parse import \
#     by_alpha3 as _iso4217_by_alpha3, \
#     by_code_num as _iso4217_by_code_num, \
#     Currency as _Currency

def _datetime_to_date(date_time: _Union[datetime, date]
                      ) -> date:
    """
    Converts type of date from datetime to date or leaves it unchanged if it is already of type date.

    Args:
        date_time (_Union[datetime, date]): Date(time) to be converted.

    Returns:
        date: (Potentially) Converted date(time).
    """
    if isinstance(date_time, datetime):
        return date_time.date()
    elif isinstance(date_time, date):
        return date_time
    else:
        raise TypeError("'" + str(date_time) + "' must be of type datetime or date!")

def _check_positivity(value: float
                      ) -> float:
    """
    Checks if value is positive.

    Args:
        value (float): value to be checked for positivity.

    Returns:
        float: positive value
    """
    if value > 0.0:
        return value
    else:
        raise Exception(str(value) + ' must be positive!')


# def check_non_negativity(value: float
#                          ) -> float:
#     """
#     Checks if value is non-negative.
#
#     Args:
#         value (float): value to be checked for non-negativity.
#
#     Returns:
#         float: non-negative value
#     """
#     if value < 0.0:
#         raise Exception(str(value) + ' must not be negative!')
#     else:
#         return value


def _check_relation(less: float,
                    more: float
                    ) -> _Tuple[float, float]:
    """
    Checks if the relative size of two floating numbers is as expected.

    Args:
        less (float): Number expected to be smaller.
        more (float): Number expected to be bigger.

    Returns:
        Tuple[float, float]: Tuple of (two) ascending numbers.
    """
    if less < more:
        return less, more
    else:
        raise Exception("'" + str(less) + "' must be smaller than '" + str(more) + "'.")


def _is_start_before_end(start: date,
                         end: date,
                         strictly: bool = True
                         ) -> bool:
    """
    Checks if the start date is before (strictly = True) of not after (strictly = False) the end date, respectively.

    Args:
        start (date): Start date
        end (date: End date
        strictly (bool): Flag defining if the start date shall be strictly before or not after the end date,
                         respectively.

    Returns:
        bool: True if start date <(=) end date. False otherwise.
    """
    if start < end:
        return True
    elif start == end:
        if strictly:
            print("WARNING: '" + str(start) + "' must not be after '" + str(end) + "'!")
            return False
        else:
            return True
    else:
        print("WARNING: '" + str(start) + "' must be earlier than '" + str(end) + "'!")
        return False

def _check_start_before_end(start: _Union[date, datetime],
                           end: _Union[date, datetime]
                           ) -> _Tuple[date, date]:
    """
    Converts the two input dates from datetime to date format it necessary and checks if the first date is earlier
    than the second one.

    Args:
        start (_Union[date, datetime]): Start date
        end (_Union[date, datetime]): End date

    Returns:
        Tuple[date, date]: start date, end date
    """
    start_date = _datetime_to_date(start)
    end_date = _datetime_to_date(end)
    if start_date < end_date:
        return start_date, end_date
    else:
        raise Exception("'" + str(start) + "' must be earlier than '" + str(end) + "'!")

def _is_chronological(start_date: date,
                      end_date: date,
                      dates: _List[date] = None,
                      strictly_start: bool = True,
                      strictly_between: bool = True,
                      strictly_end: bool = False
                      ) -> bool:
    """
    Checks if a given set of dates fulfills the following requirements:
    - start date <(=) end date
    - start date <(=) dates[0] <(=) dates[1] <(=) ... <(=) dates[n] <(=) end date

    Flags will control if the chronological order shall be fulfilled strictly or not.

    Args:
        start_date (date): First day of the interval the dates shall chronologically fall in.
        end_date (date): Last day of the interval the dates shall chronologically fall in.
        dates (List[date], optional): List of dates to be tested if they are ascending and between start and end date.
                                      Defaults to None.
        strictly_start(bool, optional): True, if start date must be strictly before following date, False otherwise.
                                        Defaults to True.
        strictly_between(bool, optional): True, if dates must be strictly monotonically ascending, False otherwise.
                                          Defaults to True.
        strictly_end(bool, optional): True, if end date must be strictly after previous date, False otherwise.
                                      Defaults to False.
    Returns:
        bool: True, if all chronological requirements w.r.t. given dates are fulfilled. False, otherwise.
    """
    if dates is None:
        return _is_start_before_end(start_date, end_date, (strictly_start & strictly_end))
    else:
        if ~_is_start_before_end(start_date, dates[0], strictly_start):
            return False

        for i in range(1, len(dates)):
            if ~_is_start_before_end(dates[i], dates[i - 1], strictly_between):
                return False

        if ~_is_start_before_end(dates[-1], end_date, strictly_end):
            return False

        return True

def check_start_before_end(start: _Union[date, datetime],
                           end: _Union[date, datetime]
                           ) -> _Tuple[date, date]:
    """
    Converts the two input dates from datetime to date format it necessary and checks if the first date is earlier
    than the second one.

    Args:
        start (_Union[date, datetime]): Start date
        end (_Union[date, datetime]): End date

    Returns:
        Tuple[date, date]: start date, end date
    """
    start_date = _datetime_to_date(start)
    end_date = _datetime_to_date(end)
    if start_date < end_date:
        return start_date, end_date
    else:
        raise Exception("'" + str(start) + "' must be earlier than '" + str(end) + "'!")

def _is_ascending_date_list(start_date: date,
                           dates: _List[date],
                           end_date: date,
                           exclude_start: bool = True,
                           exclude_end: bool = False
                           ) -> bool:
    """
    Checks if all specified dates, e.g. coupon payment dates, fall between start date and end date. Start and end date
    are excluded dependent on the corresponding boolean flags. Moreover, the dates are verified to be ascending.

    Args:
        start_date (date): First day of the interval the dates shall foll in.
        dates (List[date]): List of dates to be tested if they are ascending and between start and end date.
        end_date (date): Last day of the interval the dates shall foll in.
        exclude_start (bool, optional): True, if start date does not belong to the interval. False, otherwise.
                                        Defaults to True.
        exclude_end (bool, optional): True, if end date does not belong to the interval. False, otherwise.
                                      Defaults to False.

    Returns:
        bool: True, if dates are ascending and fall between the interval given by start and end date. False, otherwise.
    """
    if dates[0] < start_date:
        return False
    elif exclude_start & (dates[0] == start_date):
        return False

    for i in range(1, len(dates)):
        if dates[i] <= dates[i-1]:
            return False

    if dates[-1] > end_date:
        return False
    elif exclude_end & (dates[-1] == end_date):
        return False

    return True


def _currency_to_string(currency: str) -> str:
    """
    Checks if currency provided as ISO4217 three letter or numeric code, respectively, is known and converts it if
    necessary into the three letter ISO4217 currency code.

    Args:
        currency (Union[str, int, Currency]): Currency as ISO4217 three letter or numeric code or Currency object.

    Returns:
        str: Three letter ISO4217 currency code.
    """
    return currency
    if isinstance(currency, str):
        if _iso4217_by_alpha3(currency) is not None:
            return currency
        else:
            raise Exception("Unknown currency '" + str(currency) + "'!")
    elif isinstance(currency, int):
        if _iso4217_by_code_num(currency) is not None:
            return _iso4217_by_code_num(currency)[0]
        else:
            raise Exception("Unknown currency '" + str(currency) + "'!")
    elif isinstance(currency, _Currency):
        if currency[0] is not None:
            return currency[0]
        else:
            raise Exception("Unknown currency '" + str(currency) + "'!")
    else:
        raise TypeError("The currency '" + str(currency) + "' must be provided as string or integer!")


def _day_count_convention_to_string(day_count_convention: _Union[DayCounter, str]
                                    ) -> str:
    """
    Checks if day count convention is known, i.e. part of the enums list, and converts it if necessary into a sting.

    Args:
        day_count_convention (_Union[DayCounter, str]): Day count convention as DayCounter or string.

    Returns:
        str: Day count convention as string.
    """
    if isinstance(day_count_convention, DayCounter):
        try:
            return day_count_convention.value
        except AttributeError:
            # TODO: Clarify why this is not triggered.
            raise Exception("Unknown day count convention '" + str(day_count_convention) + "'!")
    elif isinstance(day_count_convention, str):
        if DayCounter.has_value(day_count_convention):
            return day_count_convention
        else:
            raise Exception("Unknown day count convention '" + day_count_convention + "'!")
    else:
        raise TypeError("The day count convention '" + str(day_count_convention)
                        + "' must be provided as DayCounter or string!")


def _roll_convention_to_string(business_day_convention: _Union[RollConvention, str]
                               ) -> str:
    """
    Checks if business day convention is known, i.e. part of the enums list, and converts it if necessary into a sting.

    Args:
        business_day_convention (_Union[RollConvention, str]): Business day convention as RollConvention or string.

    Returns:
        str: Business day convention as string.
    """
    if isinstance(business_day_convention, RollConvention):
        return business_day_convention.value
    elif isinstance(business_day_convention, str):
        if RollConvention.has_value(business_day_convention):
            return business_day_convention
        else:
            raise Exception("Unknown business day convention '" + str(business_day_convention) + "'!")
    else:
        raise TypeError("The business day convention '" + str(business_day_convention)
                        + "' must be provided as RollConvention or string!")


def _securitisation_level_to_string(securitisation_level: _Union[SecuritizationLevel, str]
                                    ) -> str:
    """
    Checks if securitisation level is known, i.e. part of the enums list, and converts it if necessary into a sting.

    Args:
        securitisation_level (_Union[SecuritizationLevel, str]): Securitisation level as SecuritizationLevel or string.

    Returns:
        str: Securitisation level as string.
    """
    if isinstance(securitisation_level, SecuritizationLevel):
        return securitisation_level.value
    elif isinstance(securitisation_level, str):
        if SecuritizationLevel.has_value(securitisation_level):
            return securitisation_level
        else:
            raise Exception("Unknown securitisation level '" + str(securitisation_level) + "'!")
    else:
        raise TypeError("The securitisation level '" + str(securitisation_level)
                        + "' must be provided as SecuritizationLevel or string!")


def _string_to_calendar(calendar: _Union[_HolidayBase, str]
                        ) -> _HolidayBase:
    """
    Checks if calendar provided as _HolidayBase or string (of corresponding country), respectively, is known and
    converts it if necessary into the HolidayBse format.

    Args:
        calendar (_Union[_HolidayBase, str]): Calendar provided as _HolidayBase or (country) string.

    Returns:
        _HolidayBase: (Potentially) converted calendar.
    """
    if isinstance(calendar, _HolidayBase):
        return calendar
    elif isinstance(calendar, str):
        if calendar in _list_supported_countries():
            return _CountryHoliday(calendar)
        else:
            raise Exception('Unknown calendar ' + calendar + "'!")
    else:
        raise TypeError("The holiday calendar '" + str(calendar)
                        + "' must be provided as HolidayBase or string!")
