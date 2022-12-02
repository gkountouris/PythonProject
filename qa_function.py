# from graph_answerable_original import shortest_paths
from transformers import AutoTokenizer, AutoModel
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import numpy as np
from utils import utils
from utils import parser_utils
import json
import random
import os
from tqdm import tqdm
import torch
import pathlib
from modelling import my_models

def qa_function(question, snipets):

    data = []

    for snipet in snipets:
        # triplets = shortest_paths(question, snipet)
        triplets = []
        nodes = []
        for triplet in triplets:
            nodes.append(triplet[0])
            nodes.append(triplet[3])
        data.append([question, snipet, nodes])

    all_devices = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]

    if len(all_devices) > 1:
        first_device = torch.device("cuda:0")
        rest_device = torch.device("cuda:1")
    elif len(all_devices) == 1:
        first_device = torch.device("cuda:0")
        rest_device = torch.device("cuda:0")
    else:
        first_device = torch.device("cpu")
        rest_device = torch.device("cpu")

    filename = 'None_michiyasunaga__BioLinkBERT-large_MLP_100_20_5e-05_AUG.pth.tar'

    try:
        my_data_path = pathlib.Path('/home/gk/Documents/BioASQ/BioASQ-data/bioasq_factoid/Graph')
        with open(my_data_path.joinpath('drkg_enhanced_logic_3_200_30_entity_embeddings.json'), 'r') as f:
            embed = json.load(f)
    except:
        my_data_path = pathlib.Path('/media/gkoun/BioASQ/BioASQ-data/bioasq_factoid/Graph/')
        with open(my_data_path.joinpath('drkg_enhanced_logic_3_200_30_entity_embeddings.json'), 'r') as f:
            embed = json.load(f)

    lm_tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-large")
    lm_model = AutoModel.from_pretrained("michiyasunaga/BioLinkBERT-large").to(first_device)
    for param in lm_model.parameters():
        param.requires_grad = False

    my_model = my_models.OnTopModeler(1224, 100).to(rest_device)

    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
        my_model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> could not find path !!! '{}'".format(filename))

    method = 'OnTopModeler'

    my_model.eval()
    with torch.no_grad():
        for q_text, snip, g_emb in data:
            sent_ids = lm_tokenizer.encode(snip.lower())[1:]
            quest_ids = lm_tokenizer.encode(q_text.lower())

            #######################################################################
            lm_input = torch.tensor([quest_ids + sent_ids]).to(first_device)
            lm_out = lm_model(lm_input)[0].to(first_device)
            #######################################################################
            begin_y, end_y = utils.model_choose(g_emb, embed, lm_out, quest_ids, my_model,
                                          rest_device, method)

    print(begin_y, end_y)

question = 'what is the recommended vitamin d serum levels?'
snipet = 'Levels of 50 nmol/L (20 ng/mL) or more are sufficient for most people. ' \
         'In contrast, the Endocrine Society stated that, for clinical practice, ' \
         'a serum 25(OH)D concentration of more than 75 nmol/L (30 ng/mL) is ' \
         'necessary to maximize the effect of vitamin D on calcium, bone, and muscle metabolism'

qa_function(question, snipet)


