"""Attention modules."""
import copy
import math
import torch
import torch.nn.functional as F


class MultiHeadedAttention(torch.nn.Module):
    """Copy from https://nlp.seas.harvard.edu/2018/04/03/attention.html."""

    def __init__(self, h, d_model, dropout=0.1, scaling=True):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        def _clones(module, N):
            "Produce N identical layers."
            return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

        self.linears = _clones(torch.nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)

        self.scaling = scaling
        if scaling == "learned":
            self.linear = torch.nn.Linear(d_model, 1)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            # mask: batch x 1 x 1 x len
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        delta = None
        if self.scaling == "learned":
            delta = F.relu(self.linear(query))
            delta = delta.clamp(max=math.sqrt(self.d_k)) + 1.0

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query,
            key,
            value,
            mask=mask,
            dropout=self.dropout,
            scaling=self.scaling,
            delta=delta,
        )

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


def attention(
    query, key, value, mask=None, dropout=None, scaling=True, delta=None,
):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1))

    if scaling == "yes":
        scores /= math.sqrt(d_k)
    elif scaling == "learned" and delta is not None:
        scores /= delta.squeeze().view(scores.size(0), 1, 1, -1)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class SoftAttention(torch.nn.Module):
    """Implementation of the self-attention mechanism in Rei et at. (2018)."""

    def __init__(self, hidden_size):
        super(SoftAttention, self).__init__()
        self.scorer = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, memory_bank, mask=None):
        a_tilde = self.scorer(memory_bank)
        if mask is not None:
            a_tilde = a_tilde.masked_fill(mask == 0, -1e9)
        a_norm = a_tilde / a_tilde.sum(1).unsqueeze(-1)
        out = memory_bank * a_norm
        return out
