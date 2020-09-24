# -*- coding: utf-8 -*-


from typing import Union
from holidays import HolidayBase, CountryHoliday
from holidays.utils import list_supported_countries
from RiVaPy.tools.enums import \
    DayCounter, \
    RollConvention, \
    SecuritizationLevel
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
    if isinstance(currency, str):
        if iso4217_by_alpha3(currency) is not None:
            return currency
        else:
            raise Exception("Unknown currency '" + str(currency) + "'!")
    elif isinstance(currency, int):
        if iso4217_by_code_num(currency) is not None:
            return iso4217_by_code_num(currency)[0]
        else:
            raise Exception("Unknown currency '" + str(currency) + "'!")
    else:
        raise TypeError("The currency '" + str(currency) + "' must be provided as string or integer!")


def day_count_convention_to_string(day_count_convention: Union[DayCounter, str]) -> str:
    """
    Checks if day count convention is known, i.e. part of the enums list, and converts it if necessary into a sting.

    Args:
        day_count_convention (Union[DayCounter, str]): Day count convention as DayCounter or string.

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


def roll_convention_to_string(business_day_convention: Union[RollConvention, str]) -> str:
    """
    Checks if business day convention is known, i.e. part of the enums list, and converts it if necessary into a sting.

    Args:
        business_day_convention (Union[RollConvention, str]): Business day convention as RollConvention or string.

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


def securitisation_level_to_string(securitisation_level: Union[SecuritizationLevel, str]) -> str:
    """
    Checks if securitisation level is known, i.e. part of the enums list, and converts it if necessary into a sting.

    Args:
        securitisation_level (Union[SecuritizationLevel, str]): Securitisation level as SecuritizationLevel or string.

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
    elif isinstance(calendar, str):
        if calendar in list_supported_countries():
            return CountryHoliday(calendar)
        else:
            raise Exception('Unknown calendar ' + calendar + "'!")
    else:
        raise TypeError("The holiday calendar '" + str(calendar)
                        + "' must be provided as HolidayBase or string!")
