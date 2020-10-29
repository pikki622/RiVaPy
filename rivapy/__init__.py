import pyvacon.version as version
from rivapy import enums
from pyvacon.analytics import setLogLevel as set_log_level


from pyvacon.analytics import registerSerialization as _register_serialization
_register_serialization('depp')

import rivapy.instruments as instruments
import rivapy.pricing as pricing
import pyvacon.marketdata as marketdata

if version.is_beta:
    import warnings
    warnings.warn('Imported pyvacon is just beta version.')
