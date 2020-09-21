# -*- coding: utf-8 -*-
import pyvacon.analytics as _analytics
from typing import Union
from datetime import datetime, date
from holidays import HolidayBase, ECB
from RiVaPy.tools._converter import _add_converter
from RiVaPy.tools.datetools import \
    Period, \
    Schedule, \
    check_start_before_end, \
    tenor_to_period
from RiVaPy.tools.enums import \
    Day_Counter, \
    Roll_Convention
from RiVaPy.tools.validators import \
    check_positivity, \
    currency_to_string, \
    day_count_convention_to_string, \
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
    def __init__(self, obj_id: str, currency: Union[str, int], issue_date: Union[date, datetime],
                 maturity_date: Union[date, datetime], notional: float = 100.0):
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
    def __init__(self, obj_id: str, currency: str, issue_date: Union[date, datetime],
                 maturity_date: Union[date, datetime], notional):
        """
        Zero coupon bond specification.
        """
        super().__init__(obj_id, currency, issue_date, maturity_date, notional)


class FixedRateBond(Bond):
    def __init__(self, obj_id: str, currency: str, issue_date: Union[date, datetime],
                 maturity_date: Union[date, datetime], notional, coupon: float, tenor: Union[Period, str],
                 spot_lag: int = 2, backwards: bool = True, stub: bool = False,
                 business_day_convention: Union[Roll_Convention, str] = Roll_Convention.FOLLOWING,
                 calendar: Union[HolidayBase, str] = None,
                 day_count_convention: Union[Day_Counter, str] = Day_Counter.ThirtyU360):
        super().__init__(obj_id, currency, issue_date, maturity_date, notional)
        self.__coupon = check_positivity(coupon)
        self.__tenor = tenor_to_period(tenor)
        self.__spot_lag = check_positivity(spot_lag)
        self.__backwards = backwards
        self.__stub = stub
        self.__business_day_convention = roll_convention_to_string(business_day_convention)
        if calendar is None:
            self.__calendar = ECB(years=range(issue_date.year, maturity_date.year + 1))
        else:
            self.__calendar = string_to_calendar(calendar)
        self.__day_count_convention = day_count_convention_to_string(day_count_convention)
        # automatically generate coupon payment schedule:
        self.__schedule = Schedule(self.issue_date, self.maturity_date, self.__tenor, self.__backwards, self.__stub,
                                   self.__business_day_convention, self.__calendar)
        self.__coupon_payment_dates = self.__schedule.generate_dates()
        self.__coupons = [self.__coupon] * len(self.__coupon_payment_dates)

    @property
    def coupon(self):
        return self.__coupon

    @property
    def tenor(self):
        return self.__tenor

    @property
    def spot_lag(self):
        return self.__spot_lag

    @property
    def backwards(self):
        return self.__backwards

    @property
    def stub(self):
        return self.__stub

    @property
    def business_day_convention(self):
        return self.__business_day_convention

    @property
    def calendar(self):
        return self.__calendar

    @property
    def day_count_convention(self):
        return self.__day_count_convention

    @property
    def schedule(self):
        return self.__schedule

    @property
    def coupon_payment_dates(self):
        return self.__coupon_payment_dates

    @property
    def coupons(self):
        return self.__coupons


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
