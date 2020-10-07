import pyvacon.version as version
from RiVaPy import enums
from pyvacon.analytics import setLogLevel as set_log_level


from pyvacon.analytics import registerSerialization as _register_serialization
_register_serialization('depp')

import RiVaPy.instruments as instruments
import RiVaPy.pricing as pricing
import pyvacon.marketdata as marketdata

if version.is_beta:
    import warnings
    warnings.warn('Imported pyvacon is just beta version.')
