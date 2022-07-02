# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 17:51:16 2016

@author: oeltz
"""
from rivapy import _pyvacon_available
if _pyvacon_available:
    from pyvacon.finance.definition import DayCounter as _DayCounter
    DayCounterType = _DayCounter.Type

    from pyvacon.numerics.interpolation import InterpolationType
    from pyvacon.numerics.extrapolation import ExtrapolationType
else:
    class DayCounterType:
        pass
    class InterpolationType:
        HAGAN_DF = 'HAGAN_DF'

    class ExtrapolationType:
        NONE = 'NONE'
        

class SecuritizationLevel:
    NONE = 'NONE'
    COLLATERALIZED = 'COLLATERALIZED' #,,,'','SUBORDINATED','MEZZANINE','EQUITY']
    SENIOR_SECURED = 'SENIOR_SECURED'
    SENIOR_UNSECURED = 'SENIOR_UNSECURED'
    SUBORDINATED = 'SUBORDINATED'
    MEZZANINE = 'MEZZANINE'
    EQUITY = 'EQUITY'


class ProductType:
       BOND = 'BOND'
       CALLABLE_BOND = 'CALLABLE_BOND'
       
       
class PricerType:
    ANALYTIC = 'ANALYTIC'
    PDE = 'PDE'
    MONTE_CARLO = 'MONTE_CARLO'
    COMBO = 'COMBO'
    
       
class Model:
    BLACK76 = 'BLACK76'
    CIR ='CIR'
    HULL_WHITE = 'HULL_WHITE'
    HESTON = 'HESTON'
    LV = 'LV'
    GBM = 'GBM'
    G2PP = 'G2PP'
    VASICEK = 'VASICEK'
    
class Period:
    A = 'A'
    SA = 'SA'
    Q = 'Q'
    M = 'M'
    D = 'D'
    
class RollConvention:
    FOLLOWING = 'Following'
    MODIFIED_FOLLOWING = 'ModifiedFollowing'
    MODIFIED_FOLLOWING_EOM = 'ModifiedFollowingEOM'
    PRECEDING = 'Preceding'
    MODIFIED_PRECEDING = 'ModifiedPreceding'
    UNADJUSTED = 'Unadjusted'
    
class VolatilityStickyness:
    NONE = 'NONE'
    StickyStrike = 'StickyStrike'
    StickyXStrike = 'StickyXStrike'
    StickyFwdMoneyness = 'StickyFwdMoneyness'

class InflationInterpolation:
    UNDEFINED = 'UNDEFINED'
    GERMAN = 'GERMAN'
    JAPAN = 'JAPAN'
    CONSTANT = 'CONSTANT'
    
class Currency:
    AED =  'AED'
    AFN =  'AFN'
    ALL =  'ALL'
    AMD =  'AMD'
    ANG =  'ANG'
    AOA =  'AOA'
    ARS =  'ARS'
    AUD =  'AUD'
    AWG =  'AWG'
    AZN =  'AZN'
    BAM =  'BAM'
    BBD =  'BBD'
    BDT =  'BDT'
    BGN =  'BGN'
    BHD =  'BHD'
    BIF =  'BIF'
    BMD =  'BMD'
    BND =  'BND'
    BOB =  'BOB'
    BRL =  'BRL'
    BSD =  'BSD'
    BTN =  'BTN'
    BWP =  'BWP'
    BYR =  'BYR'
    BZD =  'BZD'
    CAD =  'CAD'
    CDF =  'CDF'
    CHF =  'CHF'
    CLP =  'CLP'
    CNH =  'CNH'
    CNY =  'CNY'
    COP =  'COP'
    CRC =  'CRC'
    CUC =  'CUC'
    CUP =  'CUP'
    CVE =  'CVE'
    CZK =  'CZK'
    DJF =  'DJF'
    DKK =  'DKK'
    DOP =  'DOP'
    DZD =  'DZD'
    EGP =  'EGP'
    ERN =  'ERN'
    ETB =  'ETB'
    EUR =  'EUR'
    FJD =  'FJD'
    FKP =  'FKP'
    GBP =  'GBP'
    GEL =  'GEL'
    GGP =  'GGP'
    GHS =  'GHS'
    GIP =  'GIP'
    GMD =  'GMD'
    GNF =  'GNF'
    GTQ =  'GTQ'
    GYD =  'GYD'
    HKD =  'HKD'
    HNL =  'HNL'
    HRK =  'HRK'
    HTG =  'HTG'
    HUF =  'HUF'
    IDR =  'IDR'
    ILS =  'ILS'
    IMP =  'IMP'
    INR =  'INR'
    IQD =  'IQD'
    IRR =  'IRR'
    ISK =  'ISK'
    JEP =  'JEP'
    JMD =  'JMD'
    JOD =  'JOD'
    JPY =  'JPY'
    KES =  'KES'
    KGS =  'KGS'
    KHR =  'KHR'
    KMF =  'KMF'
    KPW =  'KPW'
    KRW =  'KRW'
    KWD =  'KWD'
    KYD =  'KYD'
    KZT =  'KZT'
    LAK =  'LAK'
    LBP =  'LBP'
    LKR =  'LKR'
    LRD =  'LRD'
    LSL =  'LSL'
    LTL =  'LTL'
    LVL =  'LVL'
    LYD =  'LYD'
    MAD =  'MAD'
    MDL =  'MDL'
    MGA =  'MGA'
    MKD =  'MKD'
    MMK =  'MMK'
    MNT =  'MNT'
    MOP =  'MOP'
    MRO =  'MRO'
    MUR =  'MUR'
    MVR =  'MVR'
    MWK =  'MWK'
    MXN =  'MXN'
    MYR =  'MYR'
    MZN =  'MZN'
    NAD =  'NAD'
    NGN =  'NGN'
    NIO =  'NIO'
    NOK =  'NOK'
    NPR =  'NPR'
    NZD =  'NZD'
    OMR =  'OMR'
    PAB =  'PAB'
    PEN =  'PEN'
    PGK =  'PGK'
    PHP =  'PHP'
    PKR =  'PKR'
    PLN =  'PLN'
    PYG =  'PYG'
    QAR =  'QAR'
    RON =  'RON'
    RSD =  'RSD'
    RUB =  'RUB'
    RWF =  'RWF'
    SAR =  'SAR'
    SBD =  'SBD'
    SCR =  'SCR'
    SDG =  'SDG'
    SEK =  'SEK'
    SGD =  'SGD'
    SHP =  'SHP'
    SLL =  'SLL'
    SOS =  'SOS'
    SPL =  'SPL'
    SRD =  'SRD'
    STD =  'STD'
    SVC =  'SVC'
    SYP =  'SYP'
    SZL =  'SZL'
    THB =  'THB'
    TJS =  'TJS'
    TMT =  'TMT'
    TND =  'TND'
    TOP =  'TOP'
    TRY =  'TRY'
    TTD =  'TTD'
    TVD =  'TVD'
    TWD =  'TWD'
    TZS =  'TZS'
    UAH =  'UAH'
    UGX =  'UGX'
    USD =  'USD'
    UYU =  'UYU'
    UZS =  'UZS'
    VEF =  'VEF'
    VND =  'VND'
    VUV =  'VUV'
    WST =  'WST'
    XAF =  'XAF'
    XAG =  'XAG'
    XAU =  'XAU'
    XPD =  'XPD'
    XPT =  'XPT'
    XCD =  'XCD'
    XDR =  'XDR'
    XOF =  'XOF'
    XPF =  'XPF'
    YER =  'YER'
    ZAR =  'ZAR'
    ZMW =  'ZMW'
    ZWD =  'ZWD'



