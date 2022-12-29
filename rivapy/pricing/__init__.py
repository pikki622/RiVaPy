
#from pyvacon.pricing import *
#del price

#__all__ = ['pricer', 'pricing_data', 'pricing_request']

from rivapy.pricing.pricing_data import CDSPricingData
from rivapy.pricing.pricing_data import Black76PricingData, ResultType, AmericanPdePricingData
from rivapy.pricing.pricer import price

from rivapy import _pyvacon_available
if _pyvacon_available:
	from pyvacon.finance.pricing import *

import rivapy.pricing.analytics

# from pyvacon.pricing import price as _price

# def price(pr_data):
#     if hasattr(pr_data, 'price'):
#         return pr_data.price()
#     else:
#         return _price(pr_data)