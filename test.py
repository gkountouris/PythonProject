# gradient descent optimization with nadam for a two-dimensional test function
import logging
import colorama
from colorama import Fore

from math import sqrt
from numpy import asarray
from numpy.random import rand
from numpy.random import seed
import json
import pickle
import numpy as np
import torch

from utils import utils
# with open('/media/gkoun/BioASQ/BioASQ-data/bioasq_factoid/Graph/drkg_enhanced_logic_3_200_30_entity_embeddings.json', 'r') as f:
#   emb = json.load(f)
#
# data = pickle.load(open('/media/gkoun/BioASQ/BioASQ-data/bioasq_factoid/Graph/pubmed_factoid_extracted_data_train_triplets_plus_embeddings.p', 'rb'))

node_embeddings = torch.zeros(200)
ids = np.zeros(34)
# for (qq, anss, context, type, graph_emb) in data:
#     for emb_keys in graph_emb['nodes_original']:
#         node_embeddings = torch.add(node_embeddings, torch.FloatTensor(emb[emb_keys]), out=None)
#     node_embeddings = torch.div(node_embeddings, len(graph_emb))
#     break

token_embed = torch.zeros(1, len(ids), 1024)
print(token_embed.shape)

node_embeddings = node_embeddings.view(1, 1, 200)
print(node_embeddings.shape)
node_embeddings = node_embeddings.repeat(1, len(ids), 1)
print(node_embeddings.shape)

final_embeddings = torch.cat((token_embed, node_embeddings), 2)

print(final_embeddings.shape)

# torch.cat((x, x, x), 0)