import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch
import numpy as np
from typing import Optional, Tuple


class OnTopModeler(nn.Module):
    def __init__(self, input_size, hidden_nodes):
        super(OnTopModeler, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_nodes, bias=True)
        self.linear2 = nn.Linear(hidden_nodes, 2, bias=True)
        self.loss = nn.BCELoss()
        self.tanh = nn.Tanh()

    def forward(self, input_xs):
        y = self.linear1(input_xs)
        y = self.tanh(y)
        y = self.linear2(y)
        return y


class AttentionEmbeddings(nn.Module):
    def __init__(self, input_size, hidden_nodes):
        super(AttentionEmbeddings, self).__init__()
        self.Kmatrix = nn.Linear(input_size, input_size, bias=False)
        self.Vmatrix = nn.Linear(input_size, input_size, bias=False)
        self.Qmatrix = nn.Linear(input_size, input_size, bias=False)
        self.sdpa = ScaledDotProductAttention(input_size)
        self.linear1 = nn.Linear(input_size, hidden_nodes, bias=True)
        self.linear2 = nn.Linear(hidden_nodes, 2, bias=True)
        self.loss = nn.BCELoss()
        self.tanh = nn.Tanh()

    def forward(self, input_xs, g_embe, len_quest_ids):
        concat = torch.cat((input_xs, g_embe), 1)
        key = self.Kmatrix(concat)
        value = self.Vmatrix(concat)
        query = self.Qmatrix(input_xs[:, len_quest_ids:-1, :])
        context, _ = self.sdpa(query, key, value)
        y = self.linear1(context)
        y = self.tanh(y)
        y = self.linear2(y)
        return y

class BigAttentionEmbeddings(nn.Module):
    def __init__(self, input_size, g_embe_size, hidden_nodes):
        super(BigAttentionEmbeddings, self).__init__()
        self.Kmatrix1 = nn.Linear(input_size, input_size, bias=False)
        self.Vmatrix1 = nn.Linear(input_size, input_size, bias=False)
        self.Qmatrix1 = nn.Linear(g_embe_size, input_size, bias=False)
        self.sdpa1 = ScaledDotProductAttention(input_size)
        self.Kmatrix2 = nn.Linear(input_size, input_size, bias=False)
        self.Vmatrix2 = nn.Linear(input_size, input_size, bias=False)
        self.Qmatrix2 = nn.Linear(input_size, input_size, bias=False)
        self.sdpa2 = ScaledDotProductAttention(input_size)
        self.linear1 = nn.Linear(input_size, hidden_nodes, bias=True)
        self.linear2 = nn.Linear(hidden_nodes, 2, bias=True)
        self.loss = nn.BCELoss()
        self.tanh = nn.Tanh()

    def forward(self, input_xs, g_embe, len_quest_ids):
        key1 = self.Kmatrix1(input_xs)
        value1 = self.Vmatrix1(input_xs)
        query1 = self.Qmatrix1(g_embe)
        context1, _ = self.sdpa1(query1, key1, value1)

        concat = torch.cat((input_xs, context1), 1)
        # concat = torch.cat((value1, context1), 1)

        key2 = self.Kmatrix2(concat)
        value2 = self.Vmatrix2(concat)

        snipet_embeddings = input_xs[:, len_quest_ids:-1, :]
        # snipet_embeddings = value1[:, len_quest_ids:-1, :]

        query2 = self.Qmatrix2(snipet_embeddings)

        context2, _ = self.sdpa2(query2, key2, value2)

        y = self.linear1(context2)
        y = self.tanh(y)
        y = self.linear2(y)
        return y


class PerceiverIO(nn.Module):
    def __init__(self, input_size, g_embe_size, hidden_nodes):
        super(PerceiverIO, self).__init__()
        self.Kmatrix1 = nn.Linear(input_size, input_size, bias=False)
        self.Vmatrix1 = nn.Linear(input_size, input_size, bias=False)
        self.Qmatrix1 = nn.Linear(g_embe_size, input_size, bias=False)
        self.sdpa1 = ScaledDotProductAttention(input_size)
        self.Kmatrix2 = nn.Linear(input_size, input_size, bias=False)
        self.Vmatrix2 = nn.Linear(input_size, input_size, bias=False)
        self.Qmatrix2 = nn.Linear(input_size, input_size, bias=False)
        self.sdpa2 = ScaledDotProductAttention(input_size)
        self.Kmatrix3 = nn.Linear(input_size, input_size, bias=False)
        self.Vmatrix3 = nn.Linear(input_size, input_size, bias=False)
        self.Qmatrix3 = nn.Linear(input_size, input_size, bias=False)
        self.sdpa3 = ScaledDotProductAttention(input_size)
        self.linear1 = nn.Linear(input_size, hidden_nodes, bias=True)
        self.linear2 = nn.Linear(hidden_nodes, 2, bias=True)
        self.loss = nn.BCELoss()
        self.tanh = nn.Tanh()

    def forward(self, input_xs, g_embe, len_quest_ids):

        #Cross-Attention 1
        key1 = self.Kmatrix1(input_xs)
        value1 = self.Vmatrix1(input_xs)
        query1 = self.Qmatrix1(g_embe)
        context1, _ = self.sdpa1(query1, key1, value1)
        ##################

        concat = torch.cat((input_xs, context1), 1)
        # concat = torch.cat((value1, context1), 1)

        #Self-Attention
        key2 = self.Kmatrix2(concat)
        value2 = self.Vmatrix2(concat)
        query2 = self.Qmatrix2(concat)
        context2, _ = self.sdpa2(query2, key2, value2)
        ###############

        snipet_embeddings = input_xs[:, len_quest_ids:-1, :]
        # snipet_embeddings = value1[:, len_quest_ids:-1, :]

        # Cross-Attention 2
        key3 = self.Kmatrix3(context2)
        value3 = self.Vmatrix3(context2)
        query3 = self.Qmatrix3(snipet_embeddings)
        context3, _ = self.sdpa3(query3, key3, value3)
        ##################

        y = self.linear1(context3)
        y = self.tanh(y)
        y = self.linear2(y)
        return y


class BigModel(nn.Module):
    def __init__(self, input_size, g_embe_size, hidden_nodes):

        super(BigModel, self).__init__()

        self.Kmatrix1 = nn.Linear(g_embe_size, input_size, bias=False)
        self.Vmatrix1 = nn.Linear(g_embe_size, input_size, bias=False)
        self.Qmatrix1 = nn.Linear(input_size, input_size, bias=False)
        self.sdpa1 = ScaledDotProductAttention(g_embe_size)

        self.linear1 = nn.Linear(input_size, input_size, bias=True)
        self.linear2 = nn.Linear(input_size, input_size, bias=True)

        self.Kmatrix2 = nn.Linear(input_size, input_size, bias=False)
        self.Vmatrix2 = nn.Linear(input_size, input_size, bias=False)
        self.Qmatrix2 = nn.Linear(input_size, input_size, bias=False)
        self.sdpa2 = ScaledDotProductAttention(input_size)

        self.Kmatrix3 = nn.Linear(input_size, input_size, bias=False)
        self.Vmatrix3 = nn.Linear(input_size, input_size, bias=False)
        self.Qmatrix3 = nn.Linear(input_size, input_size, bias=False)
        self.sdpa3 = ScaledDotProductAttention(input_size)

        self.linear3 = nn.Linear(input_size, hidden_nodes, bias=True)
        self.linear4 = nn.Linear(hidden_nodes, 2, bias=True)
        self.loss = nn.BCELoss()
        self.tanh = nn.Tanh()

    def forward(self, input_xs, g_embe, len_quest_ids):

        #Cross-Attention 1
        key1 = self.Kmatrix1(g_embe)
        value1 = self.Vmatrix1(g_embe)
        query1 = self.Qmatrix1(input_xs)
        context1, _ = self.sdpa1(query1, key1, value1)
        ##################

        graph_emb = self.linear1(context1)
        words_emb = self.linear2(input_xs)

        added_emb = graph_emb + words_emb

        #Self-Attention
        key2 = self.Kmatrix2(added_emb)
        value2 = self.Vmatrix2(added_emb)
        query2 = self.Qmatrix2(added_emb)
        context2, _ = self.sdpa2(query2, key2, value2)
        ###############

        snipet_embeddings = input_xs[:, len_quest_ids:-1, :]
        # snipet_embeddings = value1[:, len_quest_ids:-1, :]

        # Cross-Attention 2
        key3 = self.Kmatrix3(context2)
        value3 = self.Vmatrix3(context2)
        query3 = self.Qmatrix3(snipet_embeddings)
        context3, _ = self.sdpa3(query3, key3, value3)
        ##################

        y = self.linear3(context3)
        y = self.tanh(y)
        y = self.linear4(y)
        return y


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (sent_idstorch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)


    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn