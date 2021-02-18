
"""Encoder module"""
from __future__ import division
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class LSTMEncoder(nn.Module):
    """LSTM encoder"""

    def __init__(
        self,
        hidden_size,
        num_layers,
        bidirectional,
        embeddings,
        padding_idx,
        dropout=0.0,
    ):
        super(LSTMEncoder, self).__init__()
        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings
        self.rnn = nn.LSTM(
            input_size=embeddings.embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        self.padding_idx = padding_idx
        self.bidirectional = bidirectional

    def _fix_enc_hidden(self, hidden):
        # The encoder hidden is (layers*directions) x batch x dim.
        # We need to convert it to layers x batch x (directions*dim).
        if self.bidirectional:
            hidden = torch.cat(
                [hidden[0 : hidden.size(0) : 2], hidden[1 : hidden.size(0) : 2]], 2
            )
        return hidden

    def forward(self, sents):
        emb = self.embeddings(sents)
        lengths = sents.ne(self.padding_idx).sum(0)
        lengths = lengths.view(-1).tolist()
        packed_emb = pack(emb, lengths)
        memory_bank, enc_final = self.rnn(packed_emb)
        memory_bank = unpack(memory_bank)[0]
        memory_bank = memory_bank.contiguous()
        enc_final = tuple([self._fix_enc_hidden(enc_hid) for enc_hid in enc_final])
        return memory_bank, enc_final
