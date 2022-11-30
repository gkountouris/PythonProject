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
# import lovely_tensors as lt
# lt.monkey_patch()

# Importing the StringIO module.
from io import StringIO
# A Python program to demonstrate working of OrderedDict
from collections import OrderedDict
from pyexcel_ods import save_data
from pyexcel_ods import get_data

def train_one(the_data):
    pbar = tqdm(the_data)
    overall_losses = []
    my_model.train()
    optimizer.zero_grad()
    for q_text, exact_answers, snip, _, _, g_emb in pbar:
        if args.graph:
            g_emb = []
        sent_ids = lm_tokenizer.encode(snip.lower())[1:]
        quest_ids = lm_tokenizer.encode(q_text.lower())
        #######################################################################
        lm_input = torch.tensor([quest_ids + sent_ids]).to(first_device)
        lm_out = lm_model(lm_input)[0].to(first_device)
        #######################################################################
        begin_y, end_y = utils.model_choose(g_emb, embed, lm_out, quest_ids, my_model,
                                      rest_device, method)
        #######################################################################
        target_b = torch.zeros(len(sent_ids) - 1).to(rest_device)
        target_e = torch.zeros(len(sent_ids) - 1).to(rest_device)
        #######################################################################
        for ea in exact_answers:
            if len(ea) == 0:
                continue
            ea_ids = lm_tokenizer.encode(ea.lower())[1:-1]
            for b, e in utils.find_sub_list(ea_ids, sent_ids):
                target_b[b] = 1
                target_e[e] = 1
        if sum(target_b) == 0:
            # error_logger.error('error_in_target_b')
            continue
        if sum(target_e) == 0:
            # error_logger.error('error_in_target_e')
            continue
        #######################################################################
        loss_begin = my_model.loss(begin_y, target_b)
        loss_end = my_model.loss(end_y, target_e)
        overall_loss = (loss_begin + loss_end) / 2.0
        overall_losses.append(overall_loss)
        #######################################################################
        if len(overall_losses) >= args.batch_size:
            cost_ = (sum(overall_losses) / float(len(overall_losses)))
            cost_.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            overall_losses = []
            pbar.set_description('{}'.format(round(cost_.cpu().item(), 4)))
        #######################################################################
    ###########################################################################
    if len(overall_losses) > 0:
        cost_ = sum(overall_losses) / float(len(overall_losses))
        cost_.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_description('{}'.format(round(cost_.cpu().item(), 4)))


def test_one(the_data, mode):
    pbar = tqdm(the_data)
    aucs, prerec_aucs, overall_losses, f1s = [], [], [], {}
    ################################################################
    my_model.eval()
    with torch.no_grad():
        for q_text, exact_answers, snip, _, _, g_emb in pbar:
            if args.graph:
                g_emb = []
            sent_ids = lm_tokenizer.encode(snip.lower())[1:]
            quest_ids = lm_tokenizer.encode(q_text.lower())
            #######################################################################
            lm_input = torch.tensor([quest_ids + sent_ids]).to(first_device)
            lm_out = lm_model(lm_input)[0].to(first_device)
            #######################################################################
            begin_y, end_y = utils.model_choose(g_emb, embed, lm_out, quest_ids, my_model,
                                          rest_device, method)
            #######################################################################
            target_b = torch.zeros(len(sent_ids) - 1).to(rest_device)
            target_e = torch.zeros(len(sent_ids) - 1).to(rest_device)
            #######################################################################
            for ea in exact_answers:
                if len(ea) == 0:
                    continue
                ea_ids = lm_tokenizer.encode(ea.lower())[1:-1]
                for b, e in utils.find_sub_list(ea_ids, sent_ids):
                    target_b[b] = 1
                    target_e[e] = 1
            if sum(target_b) == 0:
                # error_logger.error('error_in_target_b')
                continue
            if sum(target_e) == 0:
                # error_logger.error('error_in_target_e')
                continue
            #######################################################################
            loss_begin = my_model.loss(begin_y, target_b)
            loss_end = my_model.loss(end_y, target_e)
            overall_loss = (loss_begin + loss_end) / 2.0
            overall_losses.append(overall_loss.cpu().item())
            #######################################################################
            auc = (roc_auc_score(target_b.tolist(), begin_y.tolist()) +
                   roc_auc_score(target_e.tolist(), end_y.tolist())) / 2.0
            aucs.append(auc)
            prerec_aucs.append((utils.pre_rec_auc(target_b.tolist(), begin_y.tolist()) +
                                utils.pre_rec_auc(target_e.tolist(), end_y.tolist())) / 2.0)
            #######################################################################
            for thresh in range(1, 10):
                thr = float(thresh) / 10.0
                by = [int(tt > thr) for tt in begin_y.tolist()]
                ey = [int(tt > thr) for tt in end_y.tolist()]
                f1_1 = f1_score(target_b.tolist(), by)
                f1_2 = f1_score(target_e.tolist(), ey)
                f1 = (f1_1 + f1_2) / 2.0
                try:
                    f1s[thresh].append(f1)
                except:
                    f1s[thresh] = [f1]
    info_logger.info(
        mode + '##:'.join(
            [
                str(np.average(overall_losses)),
                str(np.average(aucs)),
                str(np.average(prerec_aucs))
            ] + [str(np.average(f1s[thr])) for thr in f1s]
        )
    )
    result_list = [np.average(overall_losses), np.average(aucs), np.average(prerec_aucs)]
    for thr in f1s:
        result_list.append(np.average(f1s[thr]))

    data_results = [float(i) for i in result_list]
    ###########################################################################
    if monitor == 'auc':
        return np.average(prerec_aucs), data_results
    else:
        return np.average(overall_losses), data_results


if __name__ == '__main__':
    __spec__ = None

    parser = parser_utils.get_parser()
    args, _ = parser.parse_known_args()

    # args.graph = False

    try:
        my_data_path = pathlib.Path('/home/gk/Documents/BioASQ/BioASQ-data/bioasq_factoid/Graph')
        save_folder = pathlib.Path('/home/gk/Documents/BioASQ/saved_models/')
        args.train_path = my_data_path.joinpath(
            'pubmed_factoid_extracted_data_train_triplets_plus_embeddings.p').resolve()
        args.dev_path = my_data_path.joinpath('pubmed_factoid_extracted_data_dev_triplets_plus_embeddings.p').resolve()
        args.test_path = my_data_path.joinpath(
            'pubmed_factoid_extracted_data_test_triplets_plus_embeddings.p').resolve()
        with open(my_data_path.joinpath('drkg_enhanced_logic_3_200_30_entity_embeddings.json'), 'r') as f:
            embed = json.load(f)
    except:
        my_data_path = pathlib.Path('/media/gkoun/BioASQ/BioASQ-data/bioasq_factoid/Graph/')
        save_folder = pathlib.Path('/media/gkoun/BioASQ/saved_models/')
        args.train_path = my_data_path.joinpath(
            'final_pubmed_squad_covid_data_dev_graph.json').resolve()
        args.dev_path = my_data_path.joinpath('final_pubmed_squad_covid_data_train_graph.json').resolve()
        args.test_path = my_data_path.joinpath(
            'final_pubmed_squad_covid_data_test_graph.json').resolve()
        with open(my_data_path.joinpath('drkg_enhanced_logic_3_200_30_entity_embeddings.json'), 'r') as f:
            embed = json.load(f)

    info_logger = utils.set_up_logger('info', 'w')
    error_logger = utils.set_up_logger('error', 'w')

    # train_data = utils.load_data(args.train_path, args.keep_only, info_logger)
    # dev_data = utils.load_data(args.dev_path, args.keep_only, info_logger)
    # test_data = utils.load_data(args.test_path, args.keep_only, info_logger)

    train_data = utils.load_data2(args.train_path, info_logger)
    dev_data = utils.load_data2(args.dev_path, info_logger)
    test_data = utils.load_data2(args.test_path, info_logger)

    monitor = args.monitor

    random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    utils.log_gpu(all_devices, info_logger)
    lm_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    info_logger.info(lm_tokenizer.special_tokens_map)

    lm_model = AutoModel.from_pretrained(args.model_name).to(first_device)
    for param in lm_model.parameters():
        param.requires_grad = False

    random.shuffle(train_data)
    num_training_steps = args.total_epochs * (len(train_data) // args.batch_size)

    args.method = 'BigAttention'
    if args.method == 'OnTopModeler':
        my_model = my_models.OnTopModeler(args.transformer_size + 200, args.hidden_dim).to(rest_device)
    elif args.method == 'CrossAttention':
        my_model = my_models.AttentionEmbeddings(args.transformer_size, args.hidden_dim).to(rest_device)
    elif args.method == 'BigAttention':
        my_model = my_models.BigAttentionEmbeddings(args.transformer_size, 200, args.hidden_dim).to(rest_device)
    elif args.method == 'PerceiverIO':
        my_model = my_models.PerceiverIO(args.transformer_size, 200, args.hidden_dim).to(rest_device)
    else:
        my_model = my_models.PerceiverIO(args.transformer_size, 200, args.hidden_dim).to(rest_device)

    optimizer = optim.AdamW(my_model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    lr_scheduler = utils.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                   num_training_steps=num_training_steps)

    ##############################################################################
    best_dev = None
    patience_ = args.patience
    epoch = -1
    state = utils.save_checkpoint(epoch, my_model, optimizer, lr_scheduler)
    filename = 'checkpoint.pth.tar'

    results = OrderedDict()
    results['DEV'] = []
    results['TEST'] = []
    results.update({"DEV": get_data('results.ods')['DEV']})
    results.update({"TEST": get_data('results.ods')['TEST']})

    results['DEV'].append([method, args.model_name, 'Graph: {}'.format(args.graph)])
    results['TEST'].append([method, args.model_name, 'Graph: {}'.format(args.graph)])

    for epoch in range(0, args.total_epochs):
        train_one(train_data)
        dev_score, data_results = test_one(dev_data, 'DEV')
        if best_dev is None or (
                (monitor == 'auc' and dev_score > best_dev)
                or
                (monitor == 'loss' and dev_score < best_dev)
        ):
            state = utils.save_checkpoint(epoch, my_model, optimizer, lr_scheduler)
            filename = save_folder.joinpath('{}_{}_MLP_{}_{}_{}_AUG.pth.tar'.format(
                args.prefix,
                args.model_name.replace(os.path.sep, '__'),
                args.hidden_dim,
                epoch,
                str(args.lr).replace('.', 'p')
                )
            )
            print('DEV Score: ', dev_score)
            best_dev = dev_score
            patience_ = args.patience
        else:
            patience_ -= 1
            if patience_ == 0:
                break
        if epoch == args.total_epochs:
            state = utils.save_checkpoint(epoch, my_model, optimizer, lr_scheduler)
            filename = save_folder.joinpath('{}_{}_MLP_{}_{}_{}_AUG.pth.tar'.format(
                args.prefix,
                args.model_name.replace(os.path.sep, '__'),
                args.hidden_dim,
                epoch,
                str(args.lr).replace('.', 'p')
                )
            )

    results['DEV'].append(data_results)
    torch.save(state, filename)
    my_model = utils.load_model_from_checkpoint(filename, my_model, optimizer, lr_scheduler)
    _, data_results = test_one(test_data, 'TEST')
    results['TEST'].append(data_results)

    save_data("results.ods", results)