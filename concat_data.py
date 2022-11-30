import json
import nltk
from nltk.corpus import stopwords
import pathlib
import pickle
from tqdm import tqdm
import os


# path = '/home/gk/PycharmProjects/Tucker_graph/data3/pubmed_squad_covid_data_test_graph_{}.json'
#
# concat_data = []
# for idx in range(7):
#     with open(path.format(idx), 'r') as f:
#         chunk = json.load(f)
#     concat_data.append(chunk)
#
# final_data = []
# for data in concat_data:
#     for (qq, anss, context, type, id, graph_emb) in data:
#         nodes = []
#         for triplets in graph_emb:
#             nodes.append(triplets[0])
#             nodes.append(triplets[3])
#         final_data.append([qq, anss, context, type, id, nodes])
#
# print(len(final_data))
# #
# with open(os.path.join('final_pubmed_squad_covid_data_test_graph.json'), 'w') as file:
#     json.dump(final_data, file)

with open(os.path.join('final_pubmed_squad_covid_data_test_graph.json'), 'w') as file:
    data = json.load(file)




