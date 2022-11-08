from sklearn.metrics import auc as sklearn_auc
from sklearn.metrics import precision_recall_curve
from torch.optim.lr_scheduler import LambdaLR
import torch
# gradient descent optimization with nadam for a two-dimensional test function
import logging
import colorama
from colorama import Fore
import os

from math import sqrt
from numpy import asarray
from numpy.random import rand
from numpy.random import seed
import pickle
from nltk.corpus import stopwords

import logging
import pathlib


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        # Prepare the colored formatter
        colorama.init(autoreset=True)
        colors = {"DEBUG": Fore.BLUE, "INFO": Fore.CYAN,
                  "WARNING": Fore.YELLOW, "ERROR": Fore.RED, "CRITICAL": Fore.MAGENTA}
        msg = logging.Formatter.format(self, record)
        if record.levelname in colors:
            msg = colors[record.levelname] + msg + Fore.RESET
        return msg


def log_gpu(all_devices, logger):
    use_cuda = torch.cuda.is_available()
    for num, devices in enumerate(all_devices):
        logger.info('Device number {}: '.format(num))
        logger.info((use_cuda, devices))


def set_up_logger(level, mode):

    path = pathlib.Path(__file__).parent.parent.resolve()
    path = path.joinpath('Logs', level + '.log')
    # Create logger and assign handler
    logger = logging.getLogger(level)

    handler = logging.FileHandler(path, mode=mode)
    handler.setFormatter(ColoredFormatter("%(asctime)s|%(levelname)s|%(name)s|%(message)s"))
    logger.addHandler(handler)
    if level == 'info':
        logger.setLevel(logging.INFO)
    elif level == 'warning':
        logger.setLevel(logging.WARNING)
    elif level == 'error':
        logger.setLevel(logging.ERROR)
    elif level == 'critical':
        logger.setLevel(logging.CRITICAL)
    else:
        logger.setLevel(logging.DEBUG)

    return logger


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def pre_rec_auc(target, preds):
    # Data to plot precision - recall curve
    precision, recall, thresholds = precision_recall_curve(target, preds)
    # Use AUC function to calculate the area under the curve of precision recall curve
    auc_precision_recall = sklearn_auc(recall, precision)
    # print(auc_precision_recall)
    return auc_precision_recall


def find_sub_list(sl, l):
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind+sll] == sl:
            results.append((ind, ind+sll-1))
    return results

def save_checkpoint(epoch, model, optimizer, scheduler, filename='checkpoint.pth.tar'):
    '''
    :param state:       the stete of the pytorch mode
    :param filename:    the name of the file in which we will store the model.
    :return:            Nothing. It just saves the model.
    '''
    state = {
        'epoch':            epoch,
        'state_dict':       model.state_dict(),
        'optimizer':        optimizer.state_dict(),
        'scheduler':        scheduler.state_dict(),
    }
    torch.save(state, filename)
    return filename


def load_model_from_checkpoint(resume_from, my_model, optimizer, lr_scheduler):
    if os.path.isfile(resume_from):
        checkpoint = torch.load(resume_from, map_location=lambda storage, loc: storage)
        my_model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (epoch {})".format(resume_from, checkpoint['epoch']))
    else:
        print("=> could not find path !!! '{}'".format(resume_from))


def load_data(path, keep_only, logger):
    data = pickle.load(open(path, 'rb'))
    sws = stopwords.words('english')
    ##################################################################################################
    logger.info('All Data: {}'.format(len(data)))
    ##################################################################################################
    data_ = []
    for (qq, anss, context, type, graph_emb) in data:
        anss_ = []
        if type != keep_only:
            continue
        if len(context) > 500:
            continue
        for ans in anss:
            if len(ans.strip()) == 0:
                continue
            if len(ans.split()) > 10:
                continue
            if ans.lower() in sws:
                continue
            if ans.split()[0].lower() in ['the', 'a']:
                ans = ' '.join(ans.split()[1:])
            anss_.append(ans)
        if len(anss_) > 0:
            data_.append((qq, anss_, context, type, graph_emb["nodes_original"]))
    ##################################################################################################
    logger.info('Keep only Data: {}'.format(len(data)))
    ##################################################################################################
    return data_


def centroid_embeddings(g_emb, embed, lm_out, device):
    node_embeddings = torch.zeros(200)
    if len(g_emb) > 0:
        for emb_keys in g_emb:
            node_embeddings = torch.add(node_embeddings, torch.FloatTensor(embed[emb_keys]), out=None)
        node_embeddings = torch.div(node_embeddings, len(g_emb))
        node_embeddings = node_embeddings.view(1, 1, 200)
        node_embeddings = node_embeddings.repeat(1, lm_out.shape[1], 1).to(device)
    else:
        pass
    return torch.cat((lm_out, node_embeddings), 2)

