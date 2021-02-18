
"""Models module"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import warnings
import utils
from encoder import LSTMEncoder
from decoder import LSTMDecoder
from stats import Statistics

warnings.simplefilter("ignore")  # To work with PyTorch 1.2


class LMModel(nn.Module):
    """Define language model class."""

    def __init__(self, fields, args):
        super(LMModel, self).__init__()
        vocab = fields["sent"].vocab
        self.vocab_size = len(vocab)
        self.unk_idx = vocab.stoi[utils.UNK_WORD]
        self.padding_idx = vocab.stoi[utils.PAD_WORD]
        self.bos_idx = vocab.stoi[utils.BOS_WORD]
        self.eos_idx = vocab.stoi[utils.EOS_WORD]
        self.device = args.device

        self.embeddings = nn.Embedding(
            self.vocab_size, args.emb_dim, padding_idx=self.padding_idx
        )

        self.encoder = None
        if args.num_z_samples > 0:
            self.encoder = LSTMEncoder(
                hidden_size=args.rnn_size,
                num_layers=args.num_enc_layers,
                bidirectional=args.bidirectional_encoder,
                embeddings=self.embeddings,
                padding_idx=self.padding_idx,
                dropout=args.dropout,
            )
            self.mu = nn.Linear(args.rnn_size, args.z_dim, bias=False)
            self.logvar = nn.Linear(args.rnn_size, args.z_dim, bias=False)
            self.z2h = nn.Linear(args.z_dim, args.rnn_size, bias=False)
            self.z_dim = args.z_dim

        self.decoder = LSTMDecoder(
            hidden_size=args.rnn_size,
            num_layers=args.num_dec_layers,
            embeddings=self.embeddings,
            padding_idx=self.padding_idx,
            unk_idx=self.unk_idx,
            bos_idx=self.bos_idx,
            dropout=args.dropout,
            z_dim=args.z_dim,
            z_cat=args.z_cat,
            inputless=args.inputless,
            word_dropout_rate=args.word_dropout_rate,
        )

        self.dropout = nn.Dropout(args.dropout)
        self.generator = nn.Linear(args.rnn_size, self.vocab_size, bias=False)
        self.num_dec_layers = args.num_dec_layers
        self.rnn_size = args.rnn_size
        self.num_z_samples = args.num_z_samples
        self.use_avg = args.use_avg
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.padding_idx, reduction="none"
        )

        self._init_params(args)

    def _init_params(self, args):
        if args.param_init != 0.0:
            for param in self.parameters():
                param.data.uniform_(-args.param_init, args.param_init)
        with torch.no_grad():
            self.embeddings.weight[self.padding_idx].fill_(0)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _correct(self, scores, tgt):
        pred = scores.max(1)[1]
        non_padding = tgt.ne(self.padding_idx)
        n_correct = pred.eq(tgt).masked_select(non_padding).sum().item()
        n_words = non_padding.sum().item()
        return n_correct, n_words

    def _build_dec_state(self, h):
        h = h.expand(self.num_dec_layers, h.size(0), h.size(1)).contiguous()
        c = h.new_zeros(h.size())
        dec_state = (h, c)
        return dec_state

    def _encode(self, sents, clip=True, eps=5.0):
        _, enc_final = self.encoder(sents)
        h, _ = enc_final
        h = h[-1]  # use last layer
        mu = self.mu(h)
        logvar = self.logvar(h)
        if clip:
            logvar = torch.clamp(logvar, min=-eps, max=eps)
        return mu, logvar

    def _decode(self, sents, mu, logvar, beta):
        src = sents[:-1]
        tgt = sents[1:]
        tgt = tgt.view(-1)
        recon_loss, n_correct, n_words = 0.0, 0, 0
        dec_outs = 0.0

        for _ in range(self.num_z_samples):
            z = self._reparameterize(mu, logvar)
            h = self.z2h(z)
            dec_state = self._build_dec_state(h)
            dec_out, _ = self.decoder(src, dec_state, z)
            dec_out = self.dropout(dec_out)
            if self.use_avg:
                dec_outs += dec_out
            else:
                logit = self.generator(dec_out.view(-1, dec_out.size(2)))
                recon_loss += self.criterion(logit, tgt).sum()
                n_correct_, n_words_ = self._correct(logit, tgt)
                n_correct += n_correct_
                n_words += n_words_

        if self.use_avg:
            dec_outs /= self.num_z_samples
            logit = self.generator(dec_outs.view(-1, dec_outs.size(2)))
            recon_loss = self.criterion(logit, tgt).sum()
            n_correct, n_words = self._correct(logit, tgt)
        else:
            recon_loss /= self.num_z_samples
            n_correct /= self.num_z_samples
            n_words /= self.num_z_samples

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + (beta * kl_loss)
        stats = Statistics(
            loss.item(),
            recon_loss.item(),
            kl_loss.item(),
            n_correct,
            n_words,
            sents.size(1),
        )
        return loss, stats

    def _forward(self, sents):
        src = sents[:-1]
        tgt = sents[1:]
        tgt = tgt.view(-1)

        dec_out, _ = self.decoder(src)
        dec_out = self.dropout(dec_out)
        logit = self.generator(dec_out.view(-1, dec_out.size(2)))
        recon_loss = self.criterion(logit, tgt).sum()
        n_correct, n_words = self._correct(logit, tgt)
        stats = Statistics(
            loss=recon_loss.item(),
            n_correct=n_correct,
            n_words=n_words,
            n_sents=sents.size(1),
        )
        return recon_loss, stats

    def forward(self, sents, beta=1.0):
        if self.encoder is None:
            return self._forward(sents)

        mu, logvar = self._encode(sents)
        return self._decode(sents, mu, logvar, beta)

    def estimate_log_prob(self, sent, n_iw_samples):
        """Estimate log probability."""
        mu, logvar = self._encode(sent)

        sents = sent.expand(sent.size(0), n_iw_samples).contiguous()
        mu = mu.expand(n_iw_samples, mu.size(1)).contiguous()
        logvar = logvar.expand(n_iw_samples, logvar.size(1)).contiguous()
        std = torch.exp(0.5 * logvar)

        src = sents[:-1]
        tgt = sents[1:]
        tgt = tgt.view(-1)

        q_z = Normal(mu, std)
        p_z = Normal(torch.zeros_like(mu), torch.ones_like(std))
        dec_outs = 0.0
        logpz_logqz = 0.0
        logw = []

        for _ in range(self.num_z_samples):
            z = self._reparameterize(mu, logvar)
            h = self.z2h(z)
            dec_state = self._build_dec_state(h)
            dec_out, _ = self.decoder(src, dec_state, z)
            dec_out = self.dropout(dec_out)
            logpz = p_z.log_prob(z).sum(-1)
            logqz = q_z.log_prob(z).sum(-1)
            if self.use_avg:
                dec_outs += dec_out
                logpz_logqz += logpz - logqz
            else:
                logit = self.generator(dec_out.view(-1, dec_out.size(2)))
                logpxz = -self.criterion(logit, tgt).view(-1, sents.size(1)).sum(0)
                logw.append(logpxz + logpz - logqz)

        if self.use_avg:
            dec_outs /= self.num_z_samples
            logpz_logqz /= self.num_z_samples
            logit = self.generator(dec_outs.view(-1, dec_outs.size(2)))
            logpxz = -self.criterion(logit, tgt).view(-1, sents.size(1)).sum(0)
            logw.append(logpxz + logpz_logqz)
        return logw

    def generate(self, batch_size, max_sent_length, temperature):
        """Generate samples from the standard normal distribution."""
        if hasattr(self, "z2h"):
            z = torch.randn(batch_size, self.z_dim, device=self.device)
            h = self.z2h(z)
        else:
            z = None
            h = torch.zeros(batch_size, self.rnn_size, device=self.device)

        dec_state = self._build_dec_state(h)
        sent_idx = torch.arange(0, batch_size, dtype=torch.long)
        running_sent_idx = torch.arange(0, batch_size, dtype=torch.long)
        non_eos_idx = torch.arange(0, batch_size, dtype=torch.long)
        sent_mask = torch.ones(batch_size).byte()
        inp = torch.empty(batch_size, dtype=torch.long, device=self.device).fill_(
            self.bos_idx
        )
        out = torch.empty(batch_size, max_sent_length, dtype=torch.long).fill_(
            self.padding_idx
        )

        for t in range(max_sent_length):
            inp = inp.unsqueeze(0)
            dec_out, dec_state = self.decoder(inp, dec_state, z)
            logit = self.generator(dec_out.view(-1, dec_out.size(2)))
            word_weights = F.softmax(logit / temperature, dim=-1)
            word_indices = torch.multinomial(word_weights, 1)
            word_indices = word_indices.squeeze(1)
            inp = word_indices

            tmp = out[non_eos_idx]
            tmp[:, t] = word_indices.cpu().data
            out[non_eos_idx] = tmp

            sent_mask[non_eos_idx] = (
                ((inp != self.eos_idx) & (inp != self.padding_idx)).byte().cpu().data
            )
            non_eos_idx = sent_idx.masked_select(sent_mask)

            non_eos = ((inp != self.eos_idx) & (inp != self.padding_idx)).cpu().data
            # non_pad = (inp != self.padding_idx).cpu().data
            running_sent_idx = running_sent_idx.masked_select(non_eos)

            if running_sent_idx.nelement() > 0:
                inp = inp[running_sent_idx]
                dec_state = (
                    dec_state[0][:, running_sent_idx],
                    dec_state[1][:, running_sent_idx],
                )
                if z is not None:
                    z = z[running_sent_idx]
                running_sent_idx = torch.arange(
                    0, len(running_sent_idx), dtype=torch.long
                )
            else:
                break
        return out
