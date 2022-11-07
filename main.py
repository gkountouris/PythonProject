from transformers import AutoTokenizer, AutoModel
import zipfile, json, pickle, random, os
from tqdm import tqdm
from pprint import pprint
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import stopwords
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc as sklearn_auc
import numpy as np
import argparse
import xlrd
from utils import utils
from utils import parser_utils
import zipfile, json, pickle, random, os
from tqdm import tqdm
from nltk.corpus import stopwords
import torch
import logging, re
import pathlib
from modelling import my_model


if __name__ == '__main__':
    __spec__ = None

    parser = parser_utils.get_parser()
    args, _ = parser.parse_known_args()

    args.train_path = pathlib.Path('/media/gkoun/BioASQ/BioASQ-data/bioasq_factoid/Graph').resolve()
    args.dev_path = pathlib.Path('/media/gkoun/BioASQ/BioASQ-data/bioasq_factoid/Graph').resolve()
    args.test_path = pathlib.Path('/media/gkoun/BioASQ/BioASQ-data/bioasq_factoid/Graph').resolve()

    info_logger = utils.set_up_logger('info', 'a')
    error_logger = utils.set_up_logger('error', 'a')

    train_data = utils.load_data(args.train_path, args.keep_only, info_logger)
    dev_data = utils.load_data(args.train_path, args.keep_only, info_logger)
    test_data = utils.load_data(args.train_path, args.keep_only, info_logger)

    random.seed(args.my_seed)
    torch.manual_seed(args.my_seed)

    use_cuda = torch.cuda.is_available()

    # device              = torch.device("cuda") if(use_cuda) else torch.device("cpu")
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

    print('DEVICE:')
    print((use_cuda, first_device))
    print((use_cuda, rest_device))
    pprint(all_devices)

    info_logger.info('DEVICE:', (use_cuda, first_device), (use_cuda, rest_device), all_devices)

    # model_name = args.model_name
    # bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    # pprint(bert_tokenizer.special_tokens_map)
    # bert_model = AutoModel.from_pretrained(model_name).to(first_device)
    # bert_model.eval()
    # for param in bert_model.parameters():
    #     param.requires_grad = False
    #
    # random.shuffle(train_data)
    #
    # num_training_steps = args.total_epochs * (len(train_data) // args.batch_size)
    #
    # my_model = my_model.Ontop_Modeler(args.transformer_size+200, args.hidden_nodes).to(rest_device)
    # optimizer = optim.AdamW(my_model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # lr_scheduler = utils.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
    #                                                      num_training_steps=num_training_steps)



