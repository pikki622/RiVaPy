# -*- coding: utf-8 -*-


from typing import Union
from holidays import HolidayBase, CountryHoliday
from holidays.utils import list_supported_countries
from RiVaPy.tools.enums import Day_Counter, Roll_Convention
from iso4217parse import \
    by_alpha3 as iso4217_by_alpha3, \
    by_code_num as iso4217_by_code_num


def check_positivity(value: float) -> float:
    """
    Checks if value is positive.

    Args:
        value (float): value to be checked for positivity.

    Returns:
        float: value
    """
    if value > 0:
        return value
    else:
        raise Exception(str(value) + ' must be positive!')


def currency_to_string(currency: Union[str, int]) -> str:
    """
    Checks if currency provided as ISO4217 three letter or numeric code, respectively, is known and converts it if
    necessary into the three letter ISO4217 currency code.

    Args:
        currency (Union[str, int]): Currency as ISO4217 three letter or numeric code

    Returns:
        str: Three letter ISO4217 currency code
    """
    if isinstance(currency, str) and (iso4217_by_alpha3(currency) is not None):
        return currency
    elif isinstance(currency, int) and (iso4217_by_code_num(currency) is not None):
        return iso4217_by_code_num(currency)[0]
    else:
        raise Exception("Unknown currency '" + str(currency) + "'!")


def day_count_convention_to_string(day_count_convention: Union[Day_Counter, str]) -> str:
    """
    Checks if day count convention is known, i.e. part of the enums list, and converts it if necessary into a sting.

    Args:
        day_count_convention (Union[Day_Counter, str]): Day count convention as Day_Counter or string.

    Returns:
        str: Day count convention as string.
    """
    if isinstance(day_count_convention, Day_Counter):
        return day_count_convention.value
    elif isinstance(day_count_convention, str) and Day_Counter.has_value(day_count_convention):
        return day_count_convention
    else:
        raise Exception("Unknown day count convention '" + str(day_count_convention) + "'!")


def roll_convention_to_string(business_day_convention: Union[Roll_Convention, str]) -> str:
    """
    Checks if business day convention is known, i.e. part of the enums list, and converts it if necessary into a sting.

    Args:
        business_day_convention (Union[Roll_Convention, str]): Business day convention as Roll_Convention or string.

    Returns:
        str: Business day convention as string.
    """
    if isinstance(business_day_convention, Roll_Convention):
        return business_day_convention.value
    elif isinstance(business_day_convention, str) and Roll_Convention.has_value(business_day_convention):
        return business_day_convention
    else:
        raise Exception("Unknown business day convention '" + str(business_day_convention) + "'!")


def string_to_calendar(calendar: Union[HolidayBase, str]) -> HolidayBase:
    """
    Checks if calendar provided as HolidayBase or string (of corresponding country), respectively, is known and converts
    it if necessary into the HolidayBse format.

    Args:
        calendar (Union[HolidayBase, str]): Calendar provided as HolidayBase or (country) string.

    Returns:
        HolidayBase: (Potentially) converted calendar.
    """
    if isinstance(calendar, HolidayBase):
        return calendar
    if calendar in list_supported_countries():
        return CountryHoliday(calendar)
    else:
        raise Exception('Unknown calendar: ' + str(calendar))
