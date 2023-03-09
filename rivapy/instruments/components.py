# -*- coding: utf-8 -*-
from typing import Union as _Union, List
import numpy as np
from datetime import datetime, date
import rivapy.tools.interfaces as interfaces
from rivapy.tools.datetools import _date_to_datetime
from rivapy.tools._validators import _check_positivity, _check_relation, _is_chronological
from rivapy.tools.enums import DayCounterType, Rating, Sector, Country, ESGRating


class Coupon:
    def __init__(self,
                 accrual_start: _Union[date, datetime],
                 accrual_end: _Union[date, datetime],
                 payment_date: _Union[date, datetime],
                 day_count_convention: _Union[DayCounterType, str],
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

        self.__day_count_convention = DayCounterType.to_string(day_count_convention)

        self.__annualised_fixed_coupon = _check_positivity(annualised_fixed_coupon)

        self.__fixing_date = _date_to_datetime(fixing_date)

        # spread on floating rate
        self.__spread = floating_spread

        # cap/floor on floating rate
        self.__floating_rate_floor, self.__floating_rate_cap = _check_relation(floating_rate_floor, floating_rate_cap)

        # reference index for fixing floating rates
        if not floating_reference_index:
            # do not leave reference index empty as this causes pricer to ignore floating rate coupons!
            self.floating_reference_index = 'dummy_reference_index'
        else:
            self.__floating_reference_index = floating_reference_index
        self.__amortisation_factor = _check_positivity(amortisation_factor)


class Issuer(interfaces.FactoryObject):
    def __init__(self,
                 obj_id: str,
                 name: str,
                 rating: _Union[Rating, str],
                 esg_rating: _Union[ESGRating, str],
                 country: str,
                 sector: Sector):
        self.__obj_id = obj_id
        self.__name = name
        self.__rating = Rating.to_string(rating)
        self.__esg_rating = ESGRating.to_string(esg_rating)
        self.__country = country
        self.__sector = Sector.to_string(sector)

    @staticmethod
    def _create_sample(n_samples: int, seed: int = None, issuer: List[str] = None) -> List:
        """Just sample some test data

        Args:
            n_samples (int): _description_
            seed (int, optional): _description_. Defaults to None.
            issuer (List[str], optional): _description_. Defaults to None.

        Raises:
            Exception: _description_

        Returns:
            List: _description_
        """
        if seed is not None:
            np.random.seed(seed)
        ratings = list(Rating)
        esg_ratings = list(ESGRating)
        sectors = list(Sector)
        country = list(Country)
        if issuer is None:
            issuer = [f'Issuer_{str(i)}' for i in range(n_samples)]
        elif (n_samples is not None) and (n_samples !=  len(issuer)):
            raise Exception('Cannot create data since length of issuer list does not equal number of sampled. Set n_namples to None.')
        return [
            Issuer(
                f'Issuer_{str(i)}',
                issuer[i],
                np.random.choice(ratings),
                np.random.choice(esg_ratings),
                np.random.choice(country).value,
                np.random.choice(sectors),
            )
            for i in range(n_samples)
        ]

    def _to_dict(self) -> dict:
        return {'obj_id': self.obj_id, 
                'name': self.name, 'rating': self.rating,
                'esg_rating': self.esg_rating, 
                'country': self.country, 'sector': self.sector}

    @property
    def obj_id(self) -> str:
        """
        Getter for issuer id.

        Returns:
            str: Issuer id.
        """
        return self.__obj_id

    @property
    def name(self) -> str:
        """
        Getter for issuer name.

        Returns:
            str: Issuer name.
        """
        return self.__name

    @property
    def rating(self) -> str:
        """
        Getter for issuer's rating.

        Returns:
            Rating: Issuer's rating.
        """
        return self.__rating

    @rating.setter
    def rating(self, rating: _Union[Rating, str]):
        """
        Setter for issuer's rating.

        Args:
            rating: Rating of issuer.
        """
        self.__rating =Rating.to_string(rating)

    @property
    def esg_rating(self) -> str:
        """
        Getter for issuer's rating.

        Returns:
            Rating: Issuer's rating.
        """
        return self.__esg_rating

    @esg_rating.setter
    def esg_rating(self, esg_rating: _Union[ESGRating, str]):
        """
        Setter for issuer's rating.

        Args:
            rating: Rating of issuer.
        """
        self.__esg_rating = ESGRating.to_string(esg_rating)

    @property
    def country(self) -> str:
        """
        Getter for issuer's country.

        Returns:
            Country: Issuer's country.
        """
        return self.__country

    @property
    def sector(self) -> str:
        """
        Getter for issuer's sector.

        Returns:
            Sector: Issuer's sector.
        """
        return self.__sector

    @sector.setter
    def sector(self, sector:_Union[Sector, str]) -> str:
        """
        Setter for issuer's sector.

        Returns:
            Sector: Issuer's sector.
        """
        self.__sector = Sector.to_string(sector)

