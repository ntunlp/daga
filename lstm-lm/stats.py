
"""Statistics module"""
from __future__ import division
import math
import time
import utils


class Statistics(object):
    """Accumulator for loss statistics
       This class is modified from https://github.com/OpenNMT/OpenNMT-py.
       Reference:
       Guillaume Klein, Yoon Kim, Yuntian Deng, Jean Senellart and
       Alexander M. Rush. 2017. OpenNMT: Open-Source Toolkit for
       Neural Machine Translation. In Proceedings of ACL.
    """

    def __init__(
        self, loss=0, recon_loss=0, kl_loss=0, n_correct=0, n_words=0, n_sents=0
    ):
        self.loss = loss
        self.recon_loss = recon_loss
        self.kl_loss = kl_loss
        self.n_correct = n_correct
        self.n_words = n_words
        self.n_sents = n_sents
        self.start_time = time.time()

    def update(self, stat):
        """Update statistics."""
        self.loss += stat.loss
        self.recon_loss += stat.recon_loss
        self.kl_loss += stat.kl_loss
        self.n_correct += stat.n_correct
        self.n_words += stat.n_words
        self.n_sents += stat.n_sents

    def accuracy(self):
        """Compute accuracy."""
        return 100 * (self.n_correct / self.n_words)

    def xent(self):
        """Compute cross entropy."""
        return self.loss / self.n_words

    def ppl(self):
        """Compute perplexity."""
        return math.exp(min(self.loss / self.n_words, 100))

    def nll(self):
        """Compute NLL loss."""
        return self.loss / self.n_sents

    def kl(self):
        """Compute kl_loss loss."""
        return self.kl_loss / self.n_sents

    def elapsed_time(self):
        """Compute elapsed time."""
        return time.time() - self.start_time


class ESStatistics(object):
    """Accumulator for early stopping statistics"""

    def __init__(self, args):
        self.best_epoch = 1
        self.best_ppl = float("inf")
        self.best_nll = float("inf")
        self.best_kl = 0
        self.bad_counter = 0
        self.ppl_history = []
        self.patience = args.patience
        self.tol = args.tol

    def update_best(self, epoch, stats):
        """Update best training statistics."""
        self.best_epoch = epoch
        self.best_ppl = stats.ppl()
        self.best_nll = stats.nll()
        self.best_kl = stats.kl()
        self.bad_counter = 0

    def update_history(self, ppl):
        """Update ppl history."""
        self.ppl_history.append(ppl)
        if len(self.ppl_history) > self.patience and ppl + self.tol >= min(
            self.ppl_history[: -self.patience]
        ):
            self.bad_counter += 1
            utils.logger.info("Patience so far: %d", self.bad_counter)
