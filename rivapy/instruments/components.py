# -*- coding: utf-8 -*-
from typing import Union as _Union
from datetime import datetime, date
from rivapy.tools.datetools import datetime_to_date, _is_ascending_date_list
from rivapy.tools._validators import \
    _day_count_convention_to_string as _day_count_convention_to_string, _check_start_before_end, _check_positivity, \
    _check_relation,_is_chronological
from rivapy.tools.enums import \
    DayCounter, \
    Rating, \
    Sector


class Coupon:
    def __init__(self,
                 accrual_start: _Union[date, datetime],
                 accrual_end: _Union[date, datetime],
                 payment_date: _Union[date, datetime],
                 day_count_convention: _Union[DayCounter, str],
                 annualised_fixed_coupon: float,
                 fixing_date: _Union[date, datetime],
                 floating_period_start: _Union[date, datetime],
                 floating_period_end: _Union[date, datetime],
                 floating_spread: float = 0.0,
                 floating_rate_cap: float = 1e10,
                 floating_rate_floor: float = -1e10,
                 floating_reference_index: str = 'dummy_reference_index',
                 amortisation_factor: float = 1.0):
        # accrual start and end date as well as payment date
        if _is_chronological(accrual_start, [accrual_end], payment_date):
            self.__accrual_start = accrual_start
            self.__accrual_end = accrual_end
            self.__payment_date = payment_date

        self.__day_count_convention = _day_count_convention_to_string(day_count_convention)

        self.__annualised_fixed_coupon = _check_positivity(annualised_fixed_coupon)

        self.__fixing_date = datetime_to_date(fixing_date)

        # spread on floating rate
        self.__spread = floating_spread

        # cap/floor on floating rate
        self.__floating_rate_floor, self.__floating_rate_cap = _check_relation(floating_rate_floor, floating_rate_cap)

        # reference index for fixing floating rates
        if floating_reference_index == '':
            # do not leave reference index empty as this causes pricer to ignore floating rate coupons!
            self.floating_reference_index = 'dummy_reference_index'
        else:
            self.__floating_reference_index = floating_reference_index
        self.__amortisation_factor = _check_positivity(amortisation_factor)


class Issuer:
    def __init__(self,
                 issuer_id: str,
                 issuer_name: str,
                 issuer_rating: Rating,
                 issuer_country: str,
                 issuer_sector: Sector):
        self.__issuer_id = issuer_id
        self.__issuer_name = issuer_name
        self.issuer_rating = issuer_rating
        self.__issuer_country = issuer_country
        self.__issuer_sector = issuer_sector

    @property
    def issuer_id(self) -> str:
        """
        Getter for issuer id.

        Returns:
            str: Issuer id.
        """
        return self.__issuer_id

    @property
    def issuer_name(self) -> str:
        """
        Getter for issuer name.

        Returns:
            str: Issuer name.
        """
        return self.__issuer_name

    @property
    def issuer_rating(self) -> Rating:
        """
        Getter for issuer's rating.

        Returns:
            Rating: Issuer's rating.
        """
        return self.__issuer_rating

    @issuer_rating.setter
    def issuer_rating(self, rating: Rating):
        """
        Setter for issuer's rating.

        Args:
            rating: Rating of issuer.
        """
        self.__issuer_rating = rating

    @property
    def issuer_country(self) -> str:
        """
        Getter for issuer's country.

        Returns:
            Country: Issuer's country.
        """
        return self.__issuer_country

    @property
    def issuer_sector(self) -> Sector:
        """
        Getter for issuer's sector.

        Returns:
            Sector: Issuer's sector.
        """
        return self.__issuer_sector
