import rivapy.instruments.factory as _factory

from rivapy.instruments.specifications import *
from rivapy.instruments.cds_specification import CDSSpecification
from rivapy.instruments.ppa_specification import SimpleSchedule
from rivapy.instruments.ppa_specification import PPASpecification, GreenPPASpecification



def _add_to_factory(cls):
    if _factory._factory_entries is None:
        _factory._factory_entries = {}
    _factory._factory_entries[cls.__name__] = cls


_add_to_factory(SimpleSchedule)
_add_to_factory(PPASpecification)
_add_to_factory(GreenPPASpecification)


