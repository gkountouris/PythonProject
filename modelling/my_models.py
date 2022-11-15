import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch
import numpy as np
from typing import Optional, Tuple


class OnTopModeler(nn.Module):
    def __init__(self, input_size, hidden_nodes):
        super(OnTopModeler, self).__init__()
        self.input_size = input_size
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
    def __init__(self, input_size, hidden_nodes, g_embeddings):
        super(AttentionEmbeddings, self).__init__()
        self.input_size = input_size
        self.linear1 = nn.Linear(input_size, hidden_nodes, bias=True)
        self.linear2 = nn.Linear(hidden_nodes, 2, bias=True)
        self.loss = nn.BCELoss()
        self.tanh = nn.Tanh()

    def forward(self, input_xs):
        y = self.linear1(input_xs)
        y = self.tanh(y)
        y = self.linear2(y)
        return y


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
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
