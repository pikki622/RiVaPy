# -*- coding: utf-8 -*-


from enum import Enum, unique
"""
The following Enum sub-classes replace to corresponding former classes one-on-one. The main reason for this replacement
is the more comfortable iterations over the enumeration class members. Moreover, the Enum class provides potentially
useful functionalities like comparisons, pickling, ... Finally, the decorator @unique ensures unique enumeration values.
"""


class MyEnum(Enum):
    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class DBUdl(MyEnum):
    DAX = '13021'
    STOXX50E = '3343'
    DOWJOWNS = '27930'
    SPX = '3383'
    NASDAX = '11227'

    APPLE = '460'
    EXXON = '0'
    COMMERZBANK = '5'
    DB11 = '156'  # Deutsche Börse
    LUFTHANSA = '9'
    PRO_SIEBEN = '10'
    THYSSEN = '6'
    # Dax
    ADS = '269'  # Adidas
    ALV = '184'  # Allianz
    BASF = '51'
    BAYER = '58'
    BEIERSDORF = '97'
    BMW = '59'
    CONTINENTAL = '92'
    CONVESTRO = '37'
    DAIMLER = '42'
    DBK = '8'  # Deutsche Bank
    DEUTSCHE_BOERSE = '156'
    DEUTSCHE_POST = '38'
    DTE = '15'  # Deutsche Telekom
    DEUTSCHE_WOHNEN = '44'
    EON = '10'
    FRESENIUS_MEDICAL_CARE = '73'
    FRESENIUS = '39'
    HEIDELBER_CEMENT = '52'
    HENKEL = '86'
    INFINEON = '22'
    LINDE = '211'
    MERCK = '116'
    MTU = '154'
    MUV = '237'
    RWE = '129'
    SAP = '34'
    SIEMENS = '118'
    VW = '139'
    VONOVIA = '59'
    WIRECARD = '1'
    DAX_list = {'DAX': DAX, 'ADS': ADS, 'ALV': ALV, 'BASF': BASF, 'BAYER': BAYER, 'BEIERSDORF': BEIERSDORF, 'BMW': BMW,
                'CONTINENTAL': CONTINENTAL, 'CONVESTRO': CONVESTRO, 'DAIMLER': DAIMLER, 'DBK': DBK,
                'DEUTSCHE_BOERSE': DEUTSCHE_BOERSE, 'DEUTSCHE_POST': DEUTSCHE_POST, 'DTE': DTE,
                'DEUTSCHE_WOHNEN': DEUTSCHE_WOHNEN, 'EON': EON, 'FRESENIUS_MEDICAL_CARE': FRESENIUS_MEDICAL_CARE,
                'FRESENIUS': FRESENIUS, 'HEIDELBER_CEMENT': HEIDELBER_CEMENT, 'HENKEL': HENKEL, 'INFINEON': INFINEON,
                'LINDE': LINDE, 'MERCK': MERCK, 'MTU': MTU, 'MUV': MUV, 'RWE': RWE, 'SAP': SAP, 'SIEMENS': SIEMENS,
                'VW': VW, 'VONOVIA': VONOVIA, 'WIRECARD': WIRECARD}


@unique
class InterpolationType(MyEnum):
    CONSTANT = 'CONSTANT'
    LINEAR = 'LINEAR'
    LINEAR_LOG = 'LINEARLOG'
    CONSTRAINED_SPLINE = 'CONSTRAINED_SPLINE'
    HAGAN = 'HAGAN'
    HAGAN_DF = 'HAGAN_DF'


@unique
class ExtrapolationType(MyEnum):
    NONE = 'NONE'
    CONSTANT = 'CONSTANT'
    LINEAR = 'LINEAR'
    LINEAR_LOG = 'LINEARLOG'


@unique
class SecuritizationLevel(MyEnum):
    NONE = 'NONE'
    COLLATERALIZED = 'COLLATERALIZED'
    SENIOR_SECURED = 'SENIOR_SECURED'
    SENIOR_UNSECURED = 'SENIOR_UNSECURED'
    SUBORDINATED = 'SUBORDINATED'
    MEZZANINE = 'MEZZANINE'
    EQUITY = 'EQUITY'
    PREFERRED_SENIOR = 'PREFERRED_SENIOR'
    NON_PREFERRED_SENIOR = 'NON_PREFERRED_SENIOR'


@unique
class ProductType(MyEnum):
    BOND = 'BOND'
    CALLABLE_BOND = 'CALLABLE_BOND'


@unique
class PricerType(MyEnum):
    ANALYTIC = 'ANALYTIC'
    PDE = 'PDE'
    MONTE_CARLO = 'MONTE_CARLO'
    COMBO = 'COMBO'


@unique
class Model(MyEnum):
    BLACK76 = 'BLACK76'
    CIR = 'CIR'
    HULL_WHITE = 'HULL_WHITE'
    HESTON = 'HESTON'
    LV = 'LV'
    GBM = 'GBM'
    G2PP = 'G2PP'
    VASICEK = 'VASICEK'


@unique
class Period(MyEnum):
    A = 'A'
    SA = 'SA'
    Q = 'Q'
    M = 'M'
    D = 'D'


@unique
class RollConvention(MyEnum):
    FOLLOWING = 'Following'
    MODIFIED_FOLLOWING = 'ModifiedFollowing'
    MODIFIED_FOLLOWING_EOM = 'ModifiedFollowingEOM'
    MODIFIED_FOLLOWING_BIMONTHLY = 'ModifiedFollowingBimonthly'
    PRECEDING = 'Preceding'
    MODIFIED_PRECEDING = 'ModifiedPreceding'
    NEAREST = 'Nearest'
    UNADJUSTED = 'Unadjusted'


@unique
class DayCounter(MyEnum):
    ACT_ACT = 'ActAct'
    ACT365_FIXED = 'ACT365FIXED'
    ACT360 = 'Act360'
    ThirtyU360 = '30U360'
    ThirtyE360 = '30E360'
    ACT252 = 'Act252'


@unique
class VolatilityStickyness(MyEnum):
    NONE = 'NONE'
    Sticky_Strike = 'StickyStrike'
    Sticky_X_Strike = 'StickyXStrike'
    Sticky_Fwd_Moneyness = 'StickyFwdMoneyness'


@unique
class InflationInterpolation(MyEnum):
    UNDEFINED = 'UNDEFINED'
    GERMAN = 'GERMAN'
    JAPAN = 'JAPAN'
    CONSTANT = 'CONSTANT'
