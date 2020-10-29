from rivapy import _pyvacon_available
if _pyvacon_available:
    from pyvacon.pricing import price as _price

def price(pr_data):
    if hasattr(pr_data, 'price'):
        return pr_data.price()
    else:
        if _pyvacon_available:
            return _price(pr_data)
        else:
            raise Exception('Cannot price: pyvacon not avilable.')