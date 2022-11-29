import json
import nltk
from nltk.corpus import stopwords
import pathlib
import pickle
from tqdm import tqdm
import os


path = '/home/gk/PycharmProjects/Tucker_graph/data/pubmed_squad_covid_data_train_graph_{}.json'

final_data = []
for i in range(99):
    with open(path.format(i), 'r') as f:
        chunk = json.load(f)
    final_data.append(chunk)
    break

print(final_data[0][15])

# print('final: ', len(final_data))
#
# with open(os.path.join('pubmed_squad_covid_data_train_graph.json'), 'w') as file:
#     json.dump(final_data, file)
