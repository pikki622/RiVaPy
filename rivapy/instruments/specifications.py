# -*- coding: utf-8 -*-


from abc import \
    abstractmethod as _abstractmethod
from typing import \
    List as _List, \
    Union as _Union
from datetime import datetime, date
from iso4217parse import Currency
from holidays import \
    HolidayBase as _HolidayBase, \
    ECB as _ECB
from RiVaPy.basics.base import BaseObject
from RiVaPy.tools.datetools import \
    Period, \
    Schedule, \
    _datetime_to_date, \
    _datetime_to_date_list, \
    _term_to_period
from RiVaPy.tools.enums import \
    DayCounter, \
    RollConvention, \
    SecuritizationLevel
from RiVaPy.tools._validators import \
    _check_positivity, \
    _is_chronological, \
    _currency_to_string, \
    _day_count_convention_to_string, \
    _roll_convention_to_string, \
    _securitisation_level_to_string, \
    _string_to_calendar


class IssuedInstrument(BaseObject):
    def __init__(self,
                 obj_id: str,
                 issuer: str = None,
                 securitisation_level: _Union[SecuritizationLevel, str] = None):
        """
        Abstract instrument to capture information that is relevant for instruments with issuer default risk.

        Args:
            issuer (str, optional): Issuer of the instrument. Defaults to None.
            securitisation_level (Union[SecuritizationLevel, str], optional): Securitisation level of the instrument.
                                                                             Defaults to None.
        """
        super().__init__(obj_id)
        if issuer is not None:
            self.issuer = issuer
        if securitisation_level is not None:
            self.securitisation_level = securitisation_level

    def _validate_derived_base_object(self):
        pass

    @property
    def issuer(self) -> str:
        """
        Getter for instrument's issuer.

        Returns:
            str: Instrument's issuer.
        """
        return self.__issuer

    @issuer.setter
    def issuer(self, issuer: str):
        """
        Setter for instrument's issuer.

        Args:
            issuer(str): Issuer of the instrument.
        """
        self.__issuer = issuer

    @property
    def securitisation_level(self) -> str:
        """
        Getter for instrument's securitisation level.

        Returns:
            str: Instrument's securitisation level.
        """
        return self.__securitisation_level

    @securitisation_level.setter
    def securitisation_level(self, securitisation_level:  _Union[SecuritizationLevel, str]):
        self.__securitisation_level = securitisation_level_to_string(securitisation_level)

    @_abstractmethod
    def _validate_derived_issued_instrument(self):
        pass


class Bond(IssuedInstrument):
    def __init__(self,
                 obj_id: str,
                 issue_date: _Union[date, datetime],
                 maturity_date: _Union[date, datetime],
                 currency: _Union[str, int, Currency] = 'EUR',
                 notional: float = 100.0,
                 issuer: str = None,
                 securitisation_level: _Union[SecuritizationLevel, str] = None):
        """
        Abstract bond specification.

        Args:
            obj_id (str): (Preferably) Unique label of the bond, e.g. ISIN.
            issue_date (Union[date, datetime]): Date of bond issuance.
            maturity_date (Union[date, datetime]): Bond's maturity/expiry date. Must lie after the issue_date.
            currency (Union[str, int, Currency], optional): Currency as alphabetic or numeric code according to iso
                                                            currency code ISO 4217
                                                            (cf. https://www.iso.org/iso-4217-currency-codes.html).
                                                            Defaults to 'EUR'.
            notional (float, optional): Bond's notional/face value. Must be positive. Defaults to 100.0.
        """
        super().__init__(obj_id, issuer, securitisation_level)
        self.issue_date = issue_date
        self.maturity_date = maturity_date
        self.currency = currency
        self.notional = notional
        # validate dates
        self._validate_derived_issued_instrument()

    def _validate_derived_issued_instrument(self):
        self.__issue_date, self.__maturity_date = _check_start_before_end(self.__issue_date, self.__maturity_date)

    @property
    def issue_date(self) -> date:
        """
        Getter for bond's issue date.

        Returns:
            date: Bond's issue date.
        """
        return self.__issue_date

    @issue_date.setter
    def issue_date(self, issue_date: _Union[datetime, date]):
        """
        Setter for bond's issue date.

        Args:
            issue_date (Union[datetime, date]): Bond's issue date.
        """
        self.__issue_date = _datetime_to_date(issue_date)

    @property
    def maturity_date(self) -> date:
        """
        Getter for bond's maturity date.

        Returns:
            date: Bond's maturity date.
        """
        return self.__maturity_date

    @maturity_date.setter
    def maturity_date(self, maturity_date: _Union[datetime, date]):
        """
        Setter for bond's maturity date.

        Args:
            maturity_date (Union[datetime, date]): Bond's maturity date.
        """
        self.__maturity_date = _datetime_to_date(maturity_date)

    @property
    def currency(self) -> str:
        """
        Getter for bond's currency.

        Returns:
            str: Bond's ISO 4217 currency code
        """
        return self.__currency

    @currency.setter
    def currency(self,currency: _Union[str, int, Currency]):
        self.__currency = currency_to_string(currency)

    @property
    def notional(self) -> float:
        """
        Getter for bond's face value.

        Returns:
            float: Bond's face value.
        """
        return self.__notional

    @notional.setter
    def notional(self, notional):
        self.__notional = check_positivity(notional)

    @_abstractmethod
    def _validate_derived_bond(self):
        pass


class ZeroCouponBond(Bond):
    def __init__(self,
                 obj_id: str,
                 issue_date: _Union[date, datetime],
                 maturity_date: _Union[date, datetime],
                 currency: _Union[str, int, Currency] = 'EUR',
                 notional: float = 100.0,
                 issuer: str = None,
                 securitisation_level: _Union[SecuritizationLevel, str] = None):
        """
        Zero coupon bond specification.
        """
        super().__init__(obj_id,
                         issue_date,
                         maturity_date,
                         currency,
                         notional,
                         issuer,
                         securitisation_level)

    def _validate_derived_bond(self):
        pass


class FixedRateBond(Bond):
    def __init__(self,
                 obj_id: str,
                 issue_date: _Union[date, datetime],
                 maturity_date: _Union[date, datetime],
                 coupon_payment_dates: _List[_Union[date, datetime]],
                 coupons: _List[float],
                 currency: _Union[str, int, Currency] = 'EUR',
                 notional: float = 100.0,
                 issuer: str = None,
                 securitisation_level: _Union[SecuritizationLevel, str] = None):
        """
        Fixed rate bond specification by providing coupons and coupon payment dates directly.

        Args:
            coupon_payment_dates (List[Union[date, datetime]]): List of annualised coupon payment dates.
            coupons (List[float]): List of annualised coupon amounts as fraction of notional.
        """
        super().__init__(obj_id, issue_date, maturity_date, currency, notional, issuer, securitisation_level)
        self.__issue_date = issue_date
        self.__maturity_date = maturity_date
        self.__coupon_payment_dates = coupon_payment_dates
        self.__coupons = coupons


        # validation of dates' consistency
        if not _is_ascending_date_list(issue_date, coupon_payment_dates, maturity_date):
            raise Exception("Inconsistent combination of issue date '" + str(issue_date)
                            + "', payment dates '" + str(coupon_payment_dates)
                            + "', and maturity date '" + str(maturity_date) + "'.")
            # TODO: Clarify if inconsistency should be shown explicitly.
        if len(coupon_payment_dates) == len(coupons):
            self.__coupons = coupons
        else:
            raise Exception('Number of coupons ' + str(coupons) +
                            ' is not equal to number of coupon payment dates ' + str(coupon_payment_dates))

    def _validate_derived_bond(self):
        self.__coupon_payment_dates = _datetime_to_date_list(self.__coupon_payment_dates)
        # validation of dates' consistency
        if not _is_ascending_date_list(self.__issue_date, self.__coupon_payment_dates, self.__maturity_date):
            raise Exception("Inconsistent combination of issue date '" + str(self.__issue_date)
                            + "', payment dates '" + str(self.__coupon_payment_dates)
                            + "', and maturity date '" + str(self.__maturity_date) + "'.")
            # TODO: Clarify if inconsistency should be shown explicitly.
        if len(self.__coupon_payment_dates) != len(self.__coupons):
            raise Exception('Number of coupons ' + str(self.__coupons) +
                            ' is not equal to number of coupon payment dates ' + str(self.__coupon_payment_dates))

    @classmethod
    def from_master_data(cls,
                         obj_id: str,
                         issue_date: _Union[date, datetime],
                         maturity_date: _Union[date, datetime],
                         coupon: float,
                         tenor: _Union[Period, str],
                         backwards: bool = True,
                         stub: bool = False,
                         business_day_convention: _Union[RollConvention, str] = RollConvention.FOLLOWING,
                         calendar: _Union[_HolidayBase, str] = None,
                         currency: _Union[str, int, Currency] = 'EUR',
                         notional: float = 100.0,
                         issuer: str = None,
                         securitisation_level: _Union[SecuritizationLevel, str] = None):
        """
        Fixed rate bond specification based on bond's master data.

        Args:
            # TODO: How can we avoid repeating ourselves here?
            obj_id (str): (Preferably) Unique label of the bond, e.g. ISIN.
            issue_date (Union[date, datetime]): Date of bond issuance.
            maturity_date (Union[date, datetime]): Bond's maturity/expiry date. Must lie after the issue_date.

            coupon (float): Annualised coupon amount as fraction of notional, e.g. 0.0125 for fixed rate coupon of
                            1.25%.
            tenor: (Union[period, str]): Time distance between two coupon payment dates.
            backwards (bool, optional): Defines direction for rolling out the schedule. True means the schedule will be
                                        rolled out (backwards) from maturity date to issue date. Defaults to True.
            stub (bool, optional): Defines if the first/last period is accepted (True), even though it is shorter than
                                   the others, or if it remaining days are added to the neighbouring period (False).
                                   Defaults to True.
            business_day_convention (Union[RollConvention, str], optional): Set of rules defining the adjustment of
                                                                            days to ensure each date being a business
                                                                            day with respect to a given holiday
                                                                            calendar. Defaults to
                                                                            RollConvention.FOLLOWING
            calendar (Union[HolidayBase, str], optional): Holiday calendar defining the bank holidays of a country or
                                                           province (but not all non-business days as for example
                                                           Saturdays and Sundays).
                                                           Defaults (through constructor) to holidays.ECB
                                                           (= Target2 calendar) between start_day and end_day.
            # TODO: How can we avoid repeating ourselves here?
            currency (Union[str, int, Currency], optional): Currency as alphabetic or numeric code according to iso
                                                            currency code ISO 4217
                                                            (cf. https://www.iso.org/iso-4217-currency-codes.html).
                                                            Defaults to 'EUR'.
            notional (float, optional): Bond's notional/face value. Must be positive. Defaults to 100.0.
            issuer (str, optional): Issuer of the instrument. Defaults to None.
            securitisation_level (Union[SecuritizationLevel, str], optional): Securitisation level of the instrument.
                                                                              Defaults to None.

        Returns:
            FixedRateBond: Corresponding fixed rate bond with already generated schedule for coupon payments.
        """
        coupon = check_positivity(coupon)
        tenor = _tenor_to_period(tenor)
        business_day_convention = roll_convention_to_string(business_day_convention)
        if calendar is None:
            calendar = _ECB(years=range(issue_date.year, maturity_date.year + 1))
        else:
            calendar = string_to_calendar(calendar)
        schedule = Schedule(issue_date, maturity_date, tenor, backwards, stub, business_day_convention, calendar)
        coupon_payment_dates = schedule.generate_dates(True)
        coupons = [coupon] * len(coupon_payment_dates)
        securitisation_level = securitisation_level_to_string(securitisation_level)
        return FixedRateBond(obj_id, issue_date, maturity_date, coupon_payment_dates, coupons, currency, notional,
                             issuer, securitisation_level)

    @property
    def coupon_payment_dates(self) -> _List[date]:
        """
        Getter for payment dates for fixed coupons.

        Returns:
            List[date]: List of dates for fixed coupon payments.
        """
        return self.__coupon_payment_dates

    @property
    def coupons(self) -> _List[float]:
        """
        Getter for fixed coupon payments.

        Returns:
            List[float]: List of coupon amounts expressed as annualised fractions of bond's face value.
        """
        return self.__coupons


class FloatingRateNote(Bond):
    def __init__(self, 
                 obj_id: str,
                 issue_date: _Union[date, datetime],
                 maturity_date: _Union[date, datetime],
                 coupon_period_dates: _List[_Union[date, datetime]],
                 day_count_convention: _Union[DayCounter, str] = DayCounter.ThirtyU360,
                 spreads: _List[float] = None,
                 reference_index: str = 'dummy_curve',
                 currency: _Union[str, int, Currency] = 'EUR',
                 notional: float = 100.0,
                 issuer: str = None,
                 securitisation_level: _Union[SecuritizationLevel, str] = None):
        """
        Floating rate note specification by providing coupon periods directly.

        Args:
            coupon_period_dates (List[_Union[date, datetime]): Floating rate note's coupon periods, i.e. beginning and
                                                               ends of the accrual periods for the floating rate coupon
                                                               payments.
            day_count_convention (Union[DayCounter, str], optional): Day count convention for determining period
                                                                     length. Defaults to DayCounter.ThirtyU360.
            spreads (List[float], optional): List of spreads added to the floating rates derived from fixing the
                                             reference curve as fraction of notional. Defaults to None.
            reference_index (str, optional): Floating rate note underlying reference curve used for fixing the floating
                                             rate coupon amounts. Defaults to 'dummy_curve'.
                                             Note: A reference curve could also be provided later at the pricing stage.
        """
        # super().__init__(obj_id, issue_date, maturity_date, currency, notional, issuer, securitisation_level)
        Bond.__init__(self, obj_id, issue_date, maturity_date, currency, notional, issuer, securitisation_level)
        self.__coupon_period_dates = _datetime_to_date_list(coupon_period_dates)
        # validation of dates' consistency
        if not _is_ascending_date_list(issue_date, coupon_period_dates, maturity_date, False):
            raise Exception("Inconsistent combination of issue date '" + str(issue_date)
                            + "', payment dates '" + str(coupon_period_dates)
                            + "', and maturity date '" + str(maturity_date) + "'.")
            # TODO: Clarify if inconsistency should be shown explicitly.
        self.__day_count_convention = day_count_convention_to_string(day_count_convention)
        if spreads is None:
            self.__spreads = [0.0] * (len(coupon_period_dates) - 1)
        elif len(spreads) == len(coupon_period_dates) - 1:
            self.__spreads = spreads
        else:
            raise Exception('Number of spreads ' + str(spreads) +
                            ' does not fit to number of coupon periods ' + str(coupon_period_dates))
        if reference_index == '':
            # do not leave reference curve empty as this causes pricer to ignore floating rate coupons!
            self.__reference_index = 'dummy_curve'
        else:
            self.__reference_index = reference_index

    @classmethod
    def from_master_data(cls, 
                         obj_id: str,
                         issue_date: _Union[date, datetime],
                         maturity_date: _Union[date, datetime],
                         tenor: _Union[Period, str],
                         backwards: bool = True,
                         stub: bool = False,
                         business_day_convention: _Union[RollConvention, str] = RollConvention.FOLLOWING,
                         calendar: _Union[_HolidayBase, str] = None,
                         day_count_convention: _Union[DayCounter, str] = DayCounter.ThirtyU360,
                         spread: float = 0.0,
                         reference_index: str = 'dummy_curve',
                         currency: _Union[str, int, Currency] = 'EUR',
                         notional: float = 100.0,
                         issuer: str = None,
                         securitisation_level: _Union[SecuritizationLevel, str] = None):
        """
        Floating rate note specification based on master data.

        Args:
            # TODO: How can we avoid repeating ourselves here?
            obj_id (str): (Preferably) Unique label of the bond, e.g. ISIN.
            issue_date (Union[date, datetime]): Date of bond issuance.
            maturity_date (Union[date, datetime]): Bond's maturity/expiry date. Must lie after the issue_date.

            tenor: (Union[period, str]): Time distance between two coupon payment dates.
            backwards (bool, optional): Defines direction for rolling out the schedule. True means the schedule will be
                                        rolled out (backwards) from maturity date to issue date. Defaults to True.
            stub (bool, optional): Defines if the first/last period is accepted (True), even though it is shorter than
                                   the others, or if it remaining days are added to the neighbouring period (False).
                                   Defaults to True.
            business_day_convention (Union[RollConvention, str], optional): Set of rules defining the adjustment of
                                                                            days to ensure each date being a business
                                                                            day with respect to a given holiday
                                                                            calendar. Defaults to
                                                                            RollConvention.FOLLOWING
            calendar (Union[HolidayBase, str], optional): Holiday calendar defining the bank holidays of a country or
                                                          province (but not all non-business days as for example
                                                          Saturdays and Sundays).
                                                          Defaults (through constructor) to holidays.ECB
                                                          (= Target2 calendar) between start_day and end_day.
            # TODO: How can we avoid repeating ourselves here?
            day_count_convention (Union[DayCounter, str], optional): Day count convention for determining period
                                                                     length. Defaults to DayCounter.ThirtyU360.
            spread (float, optional): Spread added to floating rate derived from fixing the reference curve as fraction
                                      of notional, i.e. 0.0025 for 25 basis points. Defaults to 0.0.
            reference_index (str, optional): Floating rate note underlying reference curve used for fixing the floating
                                             rate coupon amounts. Defaults to 'dummy_curve'.
                                             Note: A reference curve could also be provided later at the pricing stage.
            currency (Union[str, int, Currency], optional): Currency as alphabetic or numeric code according to iso
                                                            currency code ISO 4217
                                                            (cf. https://www.iso.org/iso-4217-currency-codes.html).
                                                            Defaults to 'EUR'.
            notional (float, optional): Bond's notional/face value. Must be positive. Defaults to 100.0.
            issuer (str, optional): Issuer of the instrument. Defaults to None.
            securitisation_level (Union[SecuritizationLevel, str], optional): Securitisation level of the instrument.
                                                                             Defaults to None.
        Returns:
            FloatingRateNote: Corresponding floating rate note with already generated schedule for coupon payments.
        """
        tenor = _tenor_to_period(tenor)
        business_day_convention = roll_convention_to_string(business_day_convention)
        if calendar is None:
            calendar = _ECB(years=range(issue_date.year, maturity_date.year + 1))
        else:
            calendar = string_to_calendar(calendar)
        schedule = Schedule(issue_date, maturity_date, tenor, backwards, stub, business_day_convention, calendar)
        coupon_period_dates = schedule.generate_dates(False)
        spreads = [spread] * (len(coupon_period_dates) - 1)
        return FloatingRateNote(obj_id, issue_date, maturity_date, coupon_period_dates, day_count_convention, spreads,
                                reference_index, currency, notional, issuer, securitisation_level)

    @property
    def coupon_period_dates(self) -> _List[date]:
        """
        Getter for accrual periods for floating rate coupons.
        
        Returns:
            List[date]: List of accrual periods for floating rate coupons.
        """
        return self.__coupon_period_dates

    @property
    def day_count_convention(self) -> str:
        """
        Getter for bond's day count convention.
        
        Returns:
            str: Bond's day count convention. 
        """
        return self.__day_count_convention

    @property
    def spreads(self) -> _List[float]:
        """
        Getter for spreads added to the floating rates determined by fixing of reference index.
        
        Returns:
            List[float]: List of spreads added to the floating rates determined by fixing of reference index.
        """
        return self.__spreads

    @property
    def reference_index(self) -> str:
        """
        Getter for reference index for fixing floating rates.

        Returns:
            str: Reference index for fixing floating rates.
        """
        return self.__reference_index


class FixedToFloatingRateNote(FixedRateBond, FloatingRateNote):
    def __init__(self,
                 obj_id: str,
                 issue_date: _Union[date, datetime],
                 maturity_date: _Union[date, datetime],
                 coupon_payment_dates: _List[_Union[date, datetime]],
                 coupons: _List[float],
                 coupon_period_dates: _List[_Union[date, datetime]],
                 day_count_convention: _Union[DayCounter, str] = DayCounter.ThirtyU360,
                 spreads: _List[float] = None,
                 reference_index: str = 'dummy_curve',
                 currency: _Union[str, int, Currency] = 'EUR',
                 notional: float = 100.0,
                 issuer: str = None,
                 securitisation_level: _Union[SecuritizationLevel, str] = None):
        """
        Fixed-to-floating rate note specification by providing fixed rate coupons and fixed rate coupon payment dates
        as well as floating rate coupon periods directly.
        """
        FixedRateBond.__init__(self, obj_id, issue_date, maturity_date, coupon_payment_dates, coupons,
                               currency, notional, issuer, securitisation_level)

        FloatingRateNote.__init__(self, obj_id, issue_date, maturity_date, coupon_period_dates,
                                  day_count_convention, spreads, reference_index, currency, notional, issuer,
                                  securitisation_level)

    @classmethod
    def from_master_data(cls, obj_id: str,
                         issue_date: _Union[date, datetime],
                         fixed_to_float_date: _Union[date, datetime],
                         maturity_date: _Union[date, datetime],
                         coupon: float,
                         tenor_fixed: _Union[Period, str],
                         tenor_float: _Union[Period, str],
                         backwards_fixed: bool = True,
                         backwards_float: bool = True,
                         stub_fixed: bool = False,
                         stub_float: bool = False,
                         business_day_convention_fixed: _Union[RollConvention, str] = RollConvention.FOLLOWING,
                         business_day_convention_float: _Union[RollConvention, str] = RollConvention.FOLLOWING,
                         calendar_fixed: _Union[_HolidayBase, str] = None,
                         calendar_float: _Union[_HolidayBase, str] = None,
                         day_count_convention: _Union[DayCounter, str] = DayCounter.ThirtyU360,
                         spread: float = 0.0,
                         reference_index: str = 'dummy_curve',
                         currency: _Union[str, int] = 'EUR',
                         notional: float = 100.0,
                         issuer: str = None,
                         securitisation_level: _Union[SecuritizationLevel, str] = None):
        """
        Fixed-to-floating rate note specification based on master data.

        Args:
            # TODO: How can we avoid repeating ourselves here?
            obj_id (str): (Preferably) Unique label of the bond, e.g. ISIN.
            issue_date (_Union[date, datetime]): Date of bond issuance.
            fixed_to_float_date (_Union[date, datetime]): Date where fixed schedule changes into floating one.
            maturity_date (_Union[date, datetime]): Bond's maturity/expiry date. Must lie after the issue_date.
            coupon (float): Annualised coupon amount as fraction of notional, e.g. 0.0125 for fixed rate coupon of
                            1.25%.
            tenor_fixed (_Union[period, str]): Time distance between two fixed rate coupon payment dates.
            tenor_float (_Union[period, str]): Time distance between two floating rate coupon payment dates.
            backwards_fixed (bool, optional): Defines direction for rolling out the schedule for the fixed rate part.
                                              True means the schedule will be rolled out (backwards) from maturity date
                                              to issue date. Defaults to True.
            backwards_float (bool, optional): Defines direction for rolling out the schedule for the floating rate part.
                                              True means the schedule will be rolled out (backwards) from maturity date
                                              to issue date. Defaults to True.
            stub_fixed (bool, optional): Defines if the first/last period is accepted (True) in the fixed rate schedule,
                                         even though it is shorter than the others, or if it remaining days are added to
                                         the neighbouring period (False). Defaults to True.
            stub_float (bool, optional): Defines if the first/last period is accepted (True) in the float rate schedule,
                                         even though it is shorter than the others, or if it remaining days are added to
                                         the neighbouring period (False). Defaults to True.
            business_day_convention_fixed (_Union[RollConvention, str], optional): Set of rules defining the adjustment
                                                                                   of days to ensure each date in the
                                                                                   fixed rate schedule being a business
                                                                                   day with respect to a given holiday
                                                                                   calendar. Defaults to
                                                                                   RollConvention.FOLLOWING
            business_day_convention_float (_Union[RollConvention, str], optional): Set of rules defining the adjustment
                                                                                   of days to ensure each date in the
                                                                                   float rate schedule being a business
                                                                                   day with respect to a given holiday
                                                                                   calendar. Defaults to
                                                                                   RollConvention.FOLLOWING
            calendar_fixed (_Union[__HolidayBase, str], optional): Holiday calendar defining the bank holidays of a
                                                                  country or province (but not all non-business days as
                                                                  for example Saturdays and Sundays).
                                                                  Defaults (through constructor) to holidays.ECB
                                                                  (= Target2 calendar) between start_day and end_day.
            calendar_float (_Union[__HolidayBase, str], optional): Holiday calendar defining the bank holidays of a
                                                                  country or province (but not all non-business days as
                                                                  for example Saturdays and Sundays).
                                                                  Defaults (through constructor) to holidays.ECB
                                                                  (= Target2 calendar) between start_day and end_day.
            day_count_convention (_Union[DayCounter, str], optional): Day count convention for determining period
                                                                      length.Defaults to DayCounter.ThirtyU360.
            spread (float, optional): Spread added to floating rate derived from fixing the reference curve as fraction
                                      of notional, i.e. 0.0025 for 25 basis points. Defaults to 0.0.
            reference_index (str, optional): Floating rate note underlying reference curve used for fixing the floating
                                             rate coupon amounts. Defaults to 'dummy_curve'.
                                             Note: A reference curve could also be provided later at the pricing stage.
            currency (_Union[str, int], optional): Currency as alphabetic or numeric code according to iso currency code
                                                   ISO 4217 (cf. https://www.iso.org/iso-4217-currency-codes.html).
                                                   Defaults to 'EUR'.
            notional (float, optional): Bond's notional/face value. Must be positive. Defaults to 100.0.
            issuer (str, optional): Issuer of the instrument. Defaults to None.
            securitisation_level (_Union[SecuritizationLevel, str], optional): Securitisation level of the instrument.
                                                                               Defaults to None.

        Returns:
            FixedToFloatingRateNote: Corresponding fixed-to-floating rate note with already generated schedules for
                                     fixed rate and floating rate coupon payments.
        """
        fixed_rate_part = FixedRateBond.from_master_data(obj_id, issue_date, fixed_to_float_date, coupon, tenor_fixed,
                                                         backwards_fixed, stub_fixed, business_day_convention_fixed,
                                                         calendar_fixed, currency, notional, issuer,
                                                         securitisation_level)
        floating_rate_part = FloatingRateNote.from_master_data(obj_id, fixed_to_float_date, maturity_date, tenor_float,
                                                               backwards_float, stub_float,
                                                               business_day_convention_float, calendar_float,
                                                               day_count_convention, spread, reference_index, currency,
                                                               notional, issuer, securitisation_level)
        return FixedToFloatingRateNote(obj_id, issue_date, maturity_date, fixed_rate_part.coupon_payment_dates,
                                       fixed_rate_part.coupons, floating_rate_part.coupon_period_dates,
                                       day_count_convention, floating_rate_part.spreads, reference_index, currency,
                                       notional, issuer, securitisation_level)
