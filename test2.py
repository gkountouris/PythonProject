# import libraries
import pickle
import json
import sys
import os
import numpy as np

import utils.utils

# # open pickle file
# with open(sys.argv[1], 'rb') as infile:
#     obj = pickle.load(infile)
#
# # convert pickle object to json object
# json_obj = json.loads(json.dumps(obj, default=str))
#
# # write the json file
# with open(
#         os.path.splitext(sys.argv[1])[0] + '.json',
#         'w',
#         encoding='utf-8'
#     ) as outfile:
#     json.dump(json_obj, outfile, ensure_ascii=False, indent=4)
zeros = np.zeros(5)

list = [0.09316565445608868, 0.004929577464788732, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
result_list = [0, 0, 2, 3]
for thr in list:
    result_list.append(thr)

for i in range(len(list)):
    print(type(result_list[i]))

print(result_list)
utils.utils.results_to_ods('TEST', result_list)

