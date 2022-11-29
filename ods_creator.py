from collections import OrderedDict
from pyexcel_ods import save_data

results = OrderedDict()
results['DEV'] = []
results['TEST'] = []
save_data("results.ods", results)