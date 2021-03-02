# -*- coding: utf-8 -*-
from datetime import datetime as _datetime
from rivapy import _pyvacon_available
if _pyvacon_available:
    import pyvacon.finance.specification as _spec
    
    ComboSpecification = _spec.ComboSpecification
    #Equity/FX
    PayoffStructure = _spec.PayoffStructure
    ExerciseSchedule = _spec.ExerciseSchedule
    BarrierDefinition = _spec.BarrierDefinition
    BarrierSchedule = _spec.BarrierSchedule
    BarrierPayoff = _spec.BarrierPayoff
    BarrierSpecification = _spec.BarrierSpecification
    EuropeanVanillaSpecification = _spec.EuropeanVanillaSpecification
    AmericanVanillaSpecification = _spec.AmericanVanillaSpecification
    #RainbowUnderlyingSpec = _spec.RainbowUnderlyingSpec
    #RainbowBarrierSpec = _spec.RainbowBarrierSpec
    LocalVolMonteCarloSpecification = _spec.LocalVolMonteCarloSpecification
    RainbowSpecification = _spec.RainbowSpecification
    MultiMemoryExpressSpecification = _spec.MultiMemoryExpressSpecification
    MemoryExpressSpecification = _spec.MemoryExpressSpecification
    ExpressPlusSpecification = _spec.ExpressPlusSpecification
    AsianVanillaSpecification = _spec.AsianVanillaSpecification
    RiskControlStrategy = _spec.RiskControlStrategy
    AsianRiskControlSpecification = _spec.AsianRiskControlSpecification


    #Interest Rates
    IrSwapLegSpecification = _spec.IrSwapLegSpecification
    IrFixedLegSpecification = _spec.IrFixedLegSpecification
    IrFloatLegSpecification = _spec.IrFloatLegSpecification
    InterestRateSwapSpecification = _spec.InterestRateSwapSpecification
    InterestRateBasisSwapSpecification = _spec.InterestRateBasisSwapSpecification
    DepositSpecification = _spec.DepositSpecification
    InterestRateFutureSpecification = _spec.InterestRateFutureSpecification
        
    InflationLinkedBondSpecification = _spec.InflationLinkedBondSpecification
    CallableBondSpecification = _spec.CallableBondSpecification

    #GasStorageSpecification = _spec.GasStorageSpecification

    #ScheduleSpecification = _spec.ScheduleSpecification

    #SpecificationManager = _spec.SpecificationManager

    #Bonds/Credit
    CouponDescription = _spec.CouponDescription
    BondSpecification = _spec.BondSpecification
else:
    #empty placeholder...
    class BondSpecification:
        pass

def ZeroBondSpecification(obj_id: str, curr: str,  issue_date: _datetime, expiry: _datetime, notional: float = 100.0, 
                        issuer: str = 'dummy_issuer', sec_level: str='NONE')->BondSpecification:
    """[summary]

    Args:
        obj_id (str: [description]
        curr (str: [description]
        issue_date (_datetime: [description]
        expiry (_datetime: [description]
        notional (float, optional: [description]. Defaults to 100.0.
        issuer (str, optional: [description]. Defaults to 'dummy_issuer'.
        sec_level (str, optional: [description]. Defaults to 'NONE'.

    Returns:
        BondSpecification: [description]
    """
    return BondSpecification(obj_id, issuer, sec_level, curr, expiry, issue_date, notional, 'ACT365FIXED', [], [], '', [], [])



#ProjectToCorrelation = _analytics.ProjectToCorrelation

# test