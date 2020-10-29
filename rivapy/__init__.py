import warnings
    
try:
    import pyvacon.version as version
    from pyvacon.analytics import setLogLevel as set_log_level
    from pyvacon.analytics import registerSerialization as _register_serialization
    import pyvacon.marketdata as marketdata
    _register_serialization('depp')
    _pyvacon_available = True
    if version.is_beta:
        warnings.warn('Imported pyvacon is just beta version.')
except:
    _pyvacon_available = False
    warnings.warn('No pyvacon available, only limited functionality supported.')

from rivapy import enums
import rivapy.instruments as instruments
import rivapy.pricing as pricing

