from pyvacon.pricing import price as _price

def price(pr_data):
    return pr_data.price() if hasattr(pr_data, 'price') else _price(pr_data)