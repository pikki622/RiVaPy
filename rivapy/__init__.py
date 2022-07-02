import warnings

_pyvacon_available = False
try:
    import pyvacon
    _pyvacon_available = True
except Exception as e:
    warnings.warn('The pyacon module is not available. You may not use all functionality without this module. Consider installing pyvacon.')

if _pyvacon_available:
    import pyvacon.version as version
    if version.is_beta:
        warnings.warn('Imported pyvacon is just beta version.')


from rivapy import enums
import rivapy.instruments as instruments
import rivapy.pricing as pricing
import rivapy.marketdata as marketdata
#import rivapy


