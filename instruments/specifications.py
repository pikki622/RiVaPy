# -*- coding: utf-8 -*-
from typing import List, Union
from datetime import datetime, date
from holidays import HolidayBase, ECB
from RiVaPy.tools.datetools import \
    Period, \
    Schedule, \
    check_start_before_end, \
    datetime_to_date_list, \
    is_ascending_date_list, \
    tenor_to_period
from RiVaPy.tools.enums import \
    DayCounter, \
    RollConvention, \
    SecuritizationLevel
from RiVaPy.tools.validators import \
    check_positivity, \
    currency_to_string, \
    day_count_convention_to_string, \
    roll_convention_to_string, \
    securitisation_level_to_string, \
    string_to_calendar


# TODO: Decide if module abc shall be used
class IssuedInstrument:
    def __init__(self, issuer: str = None,
                 securitisation_level: Union[SecuritizationLevel, str] = None):
        """
        Abstract instrument to capture information that is relevant for instruments with issuer default risk.

        Args:
            issuer (str, optional): Issuer of the instrument. Defaults to None.
            securitisation_level (Union[SecuritizationLevel, str], optional): Securitisation level of the instrument.
                                                                              Defaults to None.
        """
        if issuer is not None:
            self.__issuer = issuer
        if securitisation_level is not None:
            self.__securitisation_level = securitisation_level_to_string(securitisation_level)

    @property
    def issuer(self):
        return self.__issuer

    @property
    def securitisation_level(self):
        return self.__securitisation_level


# TODO: Decide if module abc shall be used
class Bond(IssuedInstrument):
    def __init__(self, obj_id: str,
                 issue_date: Union[date, datetime],
                 maturity_date: Union[date, datetime],
                 currency: Union[str, int] = 'EUR',
                 notional: float = 100.0,
                 issuer: str = None,
                 securitisation_level: Union[SecuritizationLevel, str] = None):
        """
        Abstract bond specification.

        Args:
            obj_id (str): (Preferably) Unique label of the bond, e.g. ISIN
            issue_date (Union[date, datetime]): Date of bond issuance.
            maturity_date (Union[date, datetime]): Bond's maturity/expiry date. Must lie after the issue_date.
            currency (Union[str, int], optional): Currency as alphabetic or numeric code according to iso currency code
                                                  ISO 4217 (cf. https://www.iso.org/iso-4217-currency-codes.html).
                                                  Defaults to 'EUR'.
            notional (float, optional): Bond's notional/face value. Must be positive. Defaults to 100.0.
        """
        super().__init__(issuer, securitisation_level)
        self.__obj_id = obj_id
        self.__issue_date, self.__maturity_date = check_start_before_end(issue_date, maturity_date)
        self.__currency = currency_to_string(currency)
        self.__notional = check_positivity(notional)

    @property
    def obj_id(self):
        return self.__obj_id

    @property
    def issue_date(self):
        return self.__issue_date

    @property
    def maturity_date(self):
        return self.__maturity_date

    @property
    def currency(self):
        return self.__currency

    @property
    def notional(self):
        return self.__notional


class ZeroCouponBond(Bond):
    def __init__(self, obj_id: str,
                 issue_date: Union[date, datetime],
                 maturity_date: Union[date, datetime],
                 currency: Union[str, int] = 'EUR',
                 notional: float = 100.0,
                 issuer: str = None,
                 securitisation_level: Union[SecuritizationLevel, str] = None):
        """
        Zero coupon bond specification.
        """
        super().__init__(obj_id, issue_date, maturity_date, currency, notional, issuer, securitisation_level)


class FixedRateBond(Bond):
    def __init__(self, obj_id: str,
                 issue_date: Union[date, datetime],
                 maturity_date: Union[date, datetime],
                 coupon_payment_dates: List[Union[date, datetime]],
                 coupons: List[float],
                 currency: Union[str, int] = 'EUR',
                 notional: float = 100.0,
                 issuer: str = None,
                 securitisation_level: Union[SecuritizationLevel, str] = None):
        """
        Fixed rate bond specification by providing coupons and coupon payment dates directly.

        Args:
            coupon_payment_dates (List[Union[date, datetime]]): List of coupon payment dates.
            coupons (List[float]): List of coupon amounts as fraction of notional.
        """
        super().__init__(obj_id, issue_date, maturity_date, currency, notional, issuer, securitisation_level)
        self.__coupon_payment_dates = datetime_to_date_list(coupon_payment_dates)
        # validation of dates' consistency
        if not is_ascending_date_list(issue_date, coupon_payment_dates, maturity_date):
            raise Exception("Inconsistent combination of issue date '" + str(issue_date)
                            + "', payment dates '" + str(coupon_payment_dates)
                            + "', and maturity date '" + str(maturity_date) + "'.")
            # TODO: Clarify if inconsistency should be shown explicitly.
        if len(coupon_payment_dates) == len(coupon_payment_dates):
            self.__coupons = coupons
        else:
            raise Exception('Number of coupons ' + str(coupons) +
                            ' is not equal to number of coupon payment dates ' + str(coupon_payment_dates))

    @classmethod
    def from_master_data(cls, obj_id: str,
                         issue_date: Union[date, datetime],
                         maturity_date: Union[date, datetime],
                         coupon: float,
                         tenor: Union[Period, str],
                         backwards: bool = True,
                         stub: bool = False,
                         business_day_convention: Union[RollConvention, str] = RollConvention.FOLLOWING,
                         calendar: Union[HolidayBase, str] = None,
                         currency: Union[str, int] = 'EUR',
                         notional: float = 100.0,
                         issuer: str = None,
                         securitisation_level: Union[SecuritizationLevel, str] = None):
        """
        Fixed rate bond specification based on bond's master data.

        Args:
            # TODO: How can we avoid repeating ourselves here?
            obj_id (str): (Preferably) Unique label of the bond, e.g. ISIN
            issue_date (Union[date, datetime]): Date of bond issuance.
            maturity_date (Union[date, datetime]): Bond's maturity/expiry date. Must lie after the issue_date.

            coupon (float): Coupon amount as fraction of notional, e.g. 0.0125 for fixed rate coupon of 1.25%.
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
            currency (Union[str, int], optional): Currency as alphabetic or numeric code according to iso currency code
                                                  ISO 4217 (cf. https://www.iso.org/iso-4217-currency-codes.html).
                                                  Defaults to 'EUR'.
            notional (float, optional): Bond's notional/face value. Must be positive. Defaults to 100.0.
            issuer (str, optional): Issuer of the instrument. Defaults to None.
            securitisation_level (Union[SecuritizationLevel, str], optional): Securitisation level of the instrument.
                                                                              Defaults to None.

        Returns:
            FixedRateBond: Corresponding fixed rate bond with already generated schedule for coupon payments.
        """
        coupon = check_positivity(coupon)
        tenor = tenor_to_period(tenor)
        business_day_convention = roll_convention_to_string(business_day_convention)
        if calendar is None:
            calendar = ECB(years=range(issue_date.year, maturity_date.year + 1))
        else:
            calendar = string_to_calendar(calendar)
        schedule = Schedule(issue_date, maturity_date, tenor, backwards, stub, business_day_convention, calendar)
        coupon_payment_dates = schedule.generate_dates(True)
        coupons = [coupon] * len(coupon_payment_dates)
        securitisation_level = securitisation_level_to_string(securitisation_level)
        return FixedRateBond(obj_id, issue_date, maturity_date, coupon_payment_dates, coupons, currency, notional,
                             issuer, securitisation_level)

    @property
    def coupon_payment_dates(self):
        return self.__coupon_payment_dates

    @property
    def coupons(self):
        return self.__coupons


class FloatingRateNote(Bond):
    def __init__(self, obj_id: str,
                 issue_date: Union[date, datetime],
                 maturity_date: Union[date, datetime],
                 coupon_period_dates: List[Union[date, datetime]],
                 day_count_convention: Union[DayCounter, str] = DayCounter.ThirtyU360,
                 spreads: List[float] = None,
                 reference_curve: str = 'dummy_curve',
                 currency: Union[str, int] = 'EUR',
                 notional: float = 100.0,
                 issuer: str = None,
                 securitisation_level: Union[SecuritizationLevel, str] = None):
        """
        Floating rate note specification by providing coupon periods directly.

        Args:
            coupon_period_dates (List[Union[date, datetime]): Floating rate note's coupon periods, i.e. beginning and
                                                              ends of the accrual periods for the floating rate coupon
                                                              payments.
            day_count_convention (Union[DayCounter, str], optional): Day count convention for determining period length.
                                                                     Defaults to DayCounter.ThirtyU360.
            spreads (List[float], optional): List of spreads added to the floating rates derived from fixing the
                                             reference curve as fraction of notional. Defaults to None.
            reference_curve (str, optional): Floating rate note underlying reference curve used for fixing the floating
                                             rate coupon amounts. Defaults to 'dummy_curve'.
                                             Note: A reference curve could also be provided later at the pricing stage.
        """

        super().__init__(obj_id, issue_date, maturity_date, currency, notional, issuer, securitisation_level)
        self.__coupon_period_dates = datetime_to_date_list(coupon_period_dates)
        # validation of dates' consistency
        if not is_ascending_date_list(issue_date, coupon_period_dates, maturity_date, False):
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
        if reference_curve == '':
            # do not leave reference curve empty as this causes pricer to ignore floating rate coupons!
            self.__reference_curve = 'dummy_curve'
        else:
            self.__reference_curve = reference_curve

    @classmethod
    def from_master_data(cls, obj_id: str,
                         issue_date: Union[date, datetime],
                         maturity_date: Union[date, datetime],
                         tenor: Union[Period, str],
                         backwards: bool = True,
                         stub: bool = False,
                         business_day_convention: Union[RollConvention, str] = RollConvention.FOLLOWING,
                         calendar: Union[HolidayBase, str] = None,
                         day_count_convention: Union[DayCounter, str] = DayCounter.ThirtyU360,
                         spread: float = 0.0,
                         reference_curve: str = 'dummy_curve',
                         currency: Union[str, int] = 'EUR',
                         notional: float = 100.0,
                         issuer: str = None,
                         securitisation_level: Union[SecuritizationLevel, str] = None):
        """
        Floating rate note specification based on master data.

        Args:
            # TODO: How can we avoid repeating ourselves here?
            obj_id (str): (Preferably) Unique label of the bond, e.g. ISIN
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
            day_count_convention (Union[DayCounter, str], optional): Day count convention for determining period length.
                                                                     Defaults to DayCounter.ThirtyU360.
            spread (float, optional): Spread added to floating rate derived from fixing the reference curve as fraction
                                      of notional, i.e. 0.0025 for 25 basis points. Defaults to 0.0.
            reference_curve (str, optional): Floating rate note underlying reference curve used for fixing the floating
                                             rate coupon amounts. Defaults to 'dummy_curve'.
                                             Note: A reference curve could also be provided later at the pricing stage.
            currency (Union[str, int], optional): Currency as alphabetic or numeric code according to iso currency code
                                                  ISO 4217 (cf. https://www.iso.org/iso-4217-currency-codes.html).
                                                  Defaults to 'EUR'.
            notional (float, optional): Bond's notional/face value. Must be positive. Defaults to 100.0.
            issuer (str, optional): Issuer of the instrument. Defaults to None.
            securitisation_level (Union[SecuritizationLevel, str], optional): Securitisation level of the instrument.
                                                                              Defaults to None.
        Returns:
            FloatingRateNote: Corresponding floating rate note with already generated schedule for coupon payments.
        """
        tenor = tenor_to_period(tenor)
        business_day_convention = roll_convention_to_string(business_day_convention)
        if calendar is None:
            calendar = ECB(years=range(issue_date.year, maturity_date.year + 1))
        else:
            calendar = string_to_calendar(calendar)
        schedule = Schedule(issue_date, maturity_date, tenor, backwards, stub, business_day_convention, calendar)
        coupon_period_dates = schedule.generate_dates(False)
        spreads = [spread] * (len(coupon_period_dates) - 1)
        return FloatingRateNote(obj_id, issue_date, maturity_date, coupon_period_dates, day_count_convention, spreads,
                                reference_curve, currency, notional, issuer, securitisation_level)

    @property
    def coupon_period_dates(self):
        return self.__coupon_period_dates

    @property
    def day_count_convention(self):
        return self.__day_count_convention

    @property
    def spreads(self):
        return self.__spreads

    @property
    def reference_curve(self):
        return self.__reference_curve


class FixedToFloatingRateNote(FixedRateBond, FloatingRateNote):
    def __init__(self, obj_id: str,
                 issue_date: Union[date, datetime],
                 maturity_date: Union[date, datetime],
                 coupon_payment_dates: List[Union[date, datetime]],
                 coupons: List[float],
                 coupon_period_dates: List[Union[date, datetime]],
                 day_count_convention: Union[DayCounter, str] = DayCounter.ThirtyU360,
                 spreads: List[float] = None,
                 reference_curve: str = 'dummy_curve',
                 currency: Union[str, int] = 'EUR',
                 notional: float = 100.0,
                 issuer: str = None,
                 securitisation_level: Union[SecuritizationLevel, str] = None):
        """
        Fixed-to-floating rate note specification by providing fixed rate coupons and fixed rate coupon payment dates
        as well as floating rate coupon periods directly.
        """
        FixedRateBond.__init__(self, obj_id, issue_date, maturity_date, coupon_payment_dates, coupons, currency,
                               notional, issuer, securitisation_level)
        FloatingRateNote.__init__(self, obj_id, issue_date, maturity_date, coupon_period_dates, day_count_convention,
                                  spreads, reference_curve, currency, notional, issuer, securitisation_level)

    @classmethod
    def from_master_data(cls, obj_id: str,
                         issue_date: Union[date, datetime],
                         fixed_to_float_date: Union[date, datetime],
                         maturity_date: Union[date, datetime],
                         coupon: float,
                         tenor_fixed: Union[Period, str],
                         tenor_float: Union[Period, str],
                         backwards_fixed: bool = True,
                         backwards_float: bool = True,
                         stub_fixed: bool = False,
                         stub_float: bool = False,
                         business_day_convention_fixed: Union[RollConvention, str] = RollConvention.FOLLOWING,
                         business_day_convention_float: Union[RollConvention, str] = RollConvention.FOLLOWING,
                         calendar_fixed: Union[HolidayBase, str] = None,
                         calendar_float: Union[HolidayBase, str] = None,
                         day_count_convention: Union[DayCounter, str] = DayCounter.ThirtyU360,
                         spread: float = 0.0,
                         reference_curve: str = 'dummy_curve',
                         currency: Union[str, int] = 'EUR',
                         notional: float = 100.0,
                         issuer: str = None,
                         securitisation_level: Union[SecuritizationLevel, str] = None):
        """
        Fixed-to-floating rate note specification based on master data.

        Args:
            # TODO: How can we avoid repeating ourselves here?
            obj_id (str): (Preferably) Unique label of the bond, e.g. ISIN
            issue_date (Union[date, datetime]): Date of bond issuance.
            fixed_to_float_date:
            maturity_date (Union[date, datetime]): Bond's maturity/expiry date. Must lie after the issue_date.
            coupon (float): Coupon amount as fraction of notional, e.g. 0.0125 for fixed rate coupon of 1.25%.
            tenor_fixed: (Union[period, str]): Time distance between two fixed rate coupon payment dates.
            tenor_float:
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
            business_day_convention_fixed (Union[RollConvention, str], optional): Set of rules defining the adjustment
                                                                                  of days to ensure each date in the
                                                                                  fixed rate schedule being a business
                                                                                  day with respect to a given holiday
                                                                                  calendar. Defaults to
                                                                                  RollConvention.FOLLOWING
            business_day_convention_float (Union[RollConvention, str], optional): Set of rules defining the adjustment
                                                                                  of days to ensure each date in the
                                                                                  float rate schedule being a business
                                                                                  day with respect to a given holiday
                                                                                  calendar. Defaults to
                                                                                  RollConvention.FOLLOWING
            calendar_fixed (Union[HolidayBase, str], optional): Holiday calendar defining the bank holidays of a country
                                                                or province (but not all non-business days as for
                                                                example Saturdays and Sundays).
                                                                Defaults (through constructor) to holidays.ECB
                                                                (= Target2 calendar) between start_day and end_day.
            calendar_float (Union[HolidayBase, str], optional): Holiday calendar defining the bank holidays of a country
                                                                or province (but not all non-business days as for
                                                                example Saturdays and Sundays).
                                                                Defaults (through constructor) to holidays.ECB
                                                                (= Target2 calendar) between start_day and end_day.
            day_count_convention (Union[DayCounter, str], optional): Day count convention for determining period length.
                                                                     Defaults to DayCounter.ThirtyU360.
            spread (float, optional): Spread added to floating rate derived from fixing the reference curve as fraction
                                      of notional, i.e. 0.0025 for 25 basis points. Defaults to 0.0.
            reference_curve (str, optional): Floating rate note underlying reference curve used for fixing the floating
                                             rate coupon amounts. Defaults to 'dummy_curve'.
                                             Note: A reference curve could also be provided later at the pricing stage.
            currency (Union[str, int], optional): Currency as alphabetic or numeric code according to iso currency code
                                                  ISO 4217 (cf. https://www.iso.org/iso-4217-currency-codes.html).
                                                  Defaults to 'EUR'.
            notional (float, optional): Bond's notional/face value. Must be positive. Defaults to 100.0.
            issuer (str, optional): Issuer of the instrument. Defaults to None.
            securitisation_level (Union[SecuritizationLevel, str], optional): Securitisation level of the instrument.
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
                                                               day_count_convention, spread, reference_curve, currency,
                                                               notional, issuer, securitisation_level)
        return FixedToFloatingRateNote(obj_id, issue_date, maturity_date, fixed_rate_part.coupon_payment_dates,
                                       fixed_rate_part.coupons, floating_rate_part.coupon_period_dates,
                                       day_count_convention, floating_rate_part.spreads, reference_curve, currency,
                                       notional, issuer, securitisation_level)
