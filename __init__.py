from RiVaPy.config.logging_config_manager import setup_logging
from pyvacon.analytics import registerSerialization as _registerSerialization
from pyvacon import version

# Should be the first statement in the module to avoid circular dependency issues.
setup_logging('config\\logging_config.yaml')  # TODO: Check why this causes a warning message.

_registerSerialization('depp')

if version.is_beta:
    import warnings
    warnings.warn('Imported pyvacon is just beta version.')

__all__ = ['config', 'instruments', 'marketdata', 'pricing', 'tools', 'unit_tests']
