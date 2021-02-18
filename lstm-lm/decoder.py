
"""Decoders module"""
from __future__ import division
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class LSTMDecoder(nn.Module):
    """LSTM encoder"""

    def __init__(
        self,
        hidden_size,
        num_layers,
        embeddings,
        padding_idx,
        unk_idx,
        bos_idx,
        dropout=0.0,
        z_dim=0,
        z_cat=False,
        inputless=False,
        word_dropout_rate=0,
    ):
        super(LSTMDecoder, self).__init__()
        self.embeddings = embeddings
        input_size = embeddings.embedding_dim
        if z_cat:
            input_size += z_dim

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.padding_idx = padding_idx
        self.unk_idx = unk_idx
        self.bos_idx = bos_idx
        self.z_dim = z_dim
        self.z_cat = z_cat
        self.word_dropout_rate = 1.0 if inputless else word_dropout_rate
        self.inputless = inputless

    def _word_dropout(self, sents):
        prob = torch.rand(sents.size())
        prob[(sents - self.bos_idx) * (sents - self.padding_idx) == 0] = 1
        sents = sents.clone()
        sents[prob < self.word_dropout_rate] = self.unk_idx
        return sents

    def forward(self, sents, state=None, z=None):
        if (self.word_dropout_rate > 0 and self.training) or self.inputless:
            sents = self._word_dropout(sents)

        emb = self.dropout(self.embeddings(sents))
        if z is not None and self.z_cat:
            slen, batch, _ = emb.size()
            zbatch, _ = z.size()
            assert batch == zbatch
            z = z.expand(slen, batch, z.size(1)).contiguous()
            emb = torch.cat([emb, z], 2)

        packed_emb = emb
        lengths = sents.ne(self.padding_idx).sum(0)
        lengths = lengths.view(-1).tolist()
        packed_emb = pack(emb, lengths)
        memory_bank, dec_state = self.rnn(packed_emb, state)
        memory_bank = unpack(memory_bank)[0]
        memory_bank = memory_bank.contiguous()
        return memory_bank, dec_state
