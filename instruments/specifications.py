# -*- coding: utf-8 -*-
import pyvacon.analytics as _analytics
from typing import List, Union
from datetime import datetime, date
from holidays import HolidayBase, ECB
from RiVaPy.tools._converter import _add_converter
from RiVaPy.tools.datetools import \
    Period, \
    Schedule, \
    check_start_before_end, \
    is_ascending_date_list, \
    tenor_to_period
from RiVaPy.tools.enums import \
    Roll_Convention
from RiVaPy.tools.validators import \
    check_positivity, \
    currency_to_string, \
    roll_convention_to_string, \
    string_to_calendar

InflationIndexForwardCurve = _add_converter(_analytics.InflationIndexForwardCurve)

ComboSpecification = _add_converter(_analytics.ComboSpecification)
# Equity/FX
PayoffStructure = _add_converter(_analytics.PayoffStructure)
ExerciseSchedule = _add_converter(_analytics.ExerciseSchedule)
BarrierDefinition = _add_converter(_analytics.BarrierDefinition)
BarrierSchedule = _add_converter(_analytics.BarrierSchedule)
BarrierPayoff = _add_converter(_analytics.BarrierPayoff)
BarrierSpecification = _add_converter(_analytics.BarrierSpecification)
EuropeanVanillaSpecification = _add_converter(_analytics.EuropeanVanillaSpecification)
AmericanVanillaSpecification = _add_converter(_analytics.AmericanVanillaSpecification)
RainbowUnderlyingSpec = _add_converter(_analytics.RainbowUnderlyingSpec)
RainbowBarrierSpec = _add_converter(_analytics.RainbowBarrierSpec)
LocalVolMonteCarloSpecification = _add_converter(_analytics.LocalVolMonteCarloSpecification)
RainbowSpecification = _add_converter(_analytics.RainbowSpecification)
MultiMemoryExpressSpecification = _add_converter(_analytics.MultiMemoryExpressSpecification)
MemoryExpressSpecification = _add_converter(_analytics.MemoryExpressSpecification)
ExpressPlusSpecification = _add_converter(_analytics.ExpressPlusSpecification)
AsianVanillaSpecification = _add_converter(_analytics.AsianVanillaSpecification)
RiskControlStrategy = _add_converter(_analytics.RiskControlStrategy)
AsianRiskControlSpecification = _add_converter(_analytics.AsianRiskControlSpecification)


# Interest Rates
IrSwapLegSpecification = _add_converter(_analytics.IrSwapLegSpecification)
IrFixedLegSpecification = _add_converter(_analytics.IrFixedLegSpecification)
IrFloatLegSpecification = _add_converter(_analytics.IrFloatLegSpecification)
InterestRateSwapSpecification = _add_converter(_analytics.InterestRateSwapSpecification)
InterestRateBasisSwapSpecification = _add_converter(_analytics.InterestRateBasisSwapSpecification)
DepositSpecification = _add_converter(_analytics.DepositSpecification)
InterestRateFutureSpecification = _add_converter(_analytics.InterestRateFutureSpecification)


class IssuedInstrument:
    def __init__(self, issuer: str = 'dummy_issuer', sec_level: str = 'NONE'):
        if issuer is not None:
            self.__issuer = issuer
        if sec_level is not None:
            self.__sec_level = sec_level

    @property
    def issuer(self):
        return self.__issuer

    @property
    def sec_level(self):
        return self.__sec_level


class Bond:
    def __init__(self, obj_id: str,
                 currency: Union[str, int],
                 issue_date: Union[date, datetime],
                 maturity_date: Union[date, datetime],
                 notional: float = 100.0):
        """
        Abstract bond specification.

        Args:
            obj_id (str): (Preferably) Unique label of the bond, e.g. ISIN
            currency (Union[str, int]): Currency as alphabetic or numeric code according to sio currency code ISO 4217.
                                        (cf. https://www.iso.org/iso-4217-currency-codes.html)
            issue_date (Union[date, datetime]): Date of bond issuance.
            maturity_date (Union[date, datetime]): Bond's maturity/expiry date. Must lie after the issue_date.
            notional (float, optional): Bond's notional/face value. Must be positive. Defaults to 100.0.
        """
        self.__obj_id = obj_id
        self.__currency = currency_to_string(currency)
        self.__issue_date, self.__maturity_date = check_start_before_end(issue_date, maturity_date)
        self.__notional = check_positivity(notional)

    @property
    def obj_id(self):
        return self.__obj_id

    @property
    def currency(self):
        return self.__currency

    @property
    def issue_date(self):
        return self.__issue_date

    @property
    def maturity_date(self):
        return self.__maturity_date

    @property
    def notional(self):
        return self.__notional


class ZeroCouponBond(Bond):
    def __init__(self, obj_id: str,
                 currency: Union[str, int],
                 issue_date: Union[date, datetime],
                 maturity_date: Union[date, datetime],
                 notional):
        """
        Zero coupon bond specification.
        """
        super().__init__(obj_id, currency, issue_date, maturity_date, notional)


class FixedRateBond(Bond):
    def __init__(self, obj_id: str,
                 currency: Union[str, int],
                 issue_date: Union[date, datetime],
                 maturity_date: Union[date, datetime],
                 notional: float,
                 coupon_payment_dates: List[date],
                 coupons: List[float]):
        super().__init__(obj_id, currency, issue_date, maturity_date, notional)
        self.__coupon_payment_dates = is_ascending_date_list(issue_date, coupon_payment_dates, maturity_date)
        if len(coupon_payment_dates) == len(coupon_payment_dates):
            self.__coupons = coupons
        else:
            raise Exception('Number of coupons ' + str(coupons) +
                            ' is not equal to number of coupon payment dates ' + str(coupon_payment_dates))

    @classmethod
    def from_master_data(cls, obj_id: str,
                         currency: Union[str, int],
                         issue_date: Union[date, datetime],
                         maturity_date: Union[date, datetime],
                         notional: float,
                         coupon: float,
                         tenor: Union[Period, str],
                         backwards: bool = True,
                         stub: bool = False,
                         business_day_convention: Union[Roll_Convention, str] = Roll_Convention.FOLLOWING,
                         calendar: Union[HolidayBase, str] = None):
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
        return FixedRateBond(obj_id, currency, issue_date, maturity_date, notional, coupon_payment_dates, coupons)

    @property
    def coupon_payment_dates(self):
        return self.__coupon_payment_dates

    @property
    def coupons(self):
        return self.__coupons


class FloatingRateNote(Bond):
    def __init__(self, obj_id: str,
                 currency: Union[str, int],
                 issue_date: Union[date, datetime],
                 maturity_date: Union[date, datetime],
                 notional: float,
                 coupon_period_dates: List[date],
                 spreads: List[float]):
        super().__init__(obj_id, currency, issue_date, maturity_date, notional)
        self.__coupon_period_dates = is_ascending_date_list(issue_date, coupon_period_dates, maturity_date)
        if len(coupon_period_dates) == len(coupon_period_dates):
            self.__spreads = spreads
        else:
            raise Exception('Number of spreads ' + str(spreads) +
                            ' does not fir to number of coupon periods ' + str(coupon_period_dates))

    @classmethod
    def from_master_data(cls, obj_id: str,
                         currency: Union[str, int],
                         issue_date: Union[date, datetime],
                         maturity_date: Union[date, datetime],
                         notional: float,
                         spread: float,
                         tenor: Union[Period, str],
                         backwards: bool = True,
                         stub: bool = False,
                         business_day_convention: Union[Roll_Convention, str] = Roll_Convention.FOLLOWING,
                         calendar: Union[HolidayBase, str] = None):
        tenor = tenor_to_period(tenor)
        business_day_convention = roll_convention_to_string(business_day_convention)
        if calendar is None:
            calendar = ECB(years=range(issue_date.year, maturity_date.year + 1))
        else:
            calendar = string_to_calendar(calendar)
        schedule = Schedule(issue_date, maturity_date, tenor, backwards, stub, business_day_convention, calendar)
        coupon_period_dates = schedule.generate_dates(False)
        spreads = [spread] * (len(coupon_period_dates) - 1)
        return FloatingRateNote(obj_id, currency, issue_date, maturity_date, notional, coupon_period_dates, spreads)

    @property
    def coupon_period_dates(self):
        return self.__coupon_period_dates

    @property
    def spreads(self):
        return self.__spreads


# Bonds/Credit
CouponDescription = _add_converter(_analytics.CouponDescription)
BondSpecification = _add_converter(_analytics.BondSpecification)
InflationLinkedBondSpecification = _add_converter(_analytics.InflationLinkedBondSpecification)
CallableBondSpecification = _add_converter(_analytics.CallableBondSpecification)

GasStorageSpecification = _add_converter(_analytics.GasStorageSpecification)

ScheduleSpecification = _add_converter(_analytics.ScheduleSpecification)

SpecificationManager = _add_converter(_analytics.SpecificationManager)

vectorCouponDescription = _analytics.vectorCouponDescription
vectorRainbowBarrierSpec = _analytics.vectorRainbowBarrierSpec
vectorRainbowUdlSpec = _analytics.vectorRainbowUdlSpec

# ProjectToCorrelation = _analytics.ProjectToCorrelation
