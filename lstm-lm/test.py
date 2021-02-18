
"""Test module"""
from __future__ import print_function, division
import argparse
from pprint import pformat
import math
import time
import torch
import tqdm
import utils
from utils import logger
from data import LMDataset
from train import validate
from stats import Statistics


def report_iw_nll(model, test_iter, n_iw_iter, n_iw_samples):
    """Calculate the importance-weighted estimate of NLL."""
    model.eval()
    loss, n_sents, n_words = 0.0, 0, 0
    with torch.no_grad():
        for batch in tqdm.tqdm(test_iter, total=len(test_iter)):
            sents = batch.sent
            n_sents += sents.size(1)
            for i in range(sents.size(1)):
                sent = sents[:, i]  # get one sentence
                sent = sent.masked_select(sent.ne(model.padding_idx))  # trim pad token
                n_words += sent.size(0) - 1  # skip start symbol
                sent = sent.unsqueeze(1)
                logw = []
                for _ in range(n_iw_iter):
                    logw.extend(model.estimate_log_prob(sent, n_iw_samples))
                logw = torch.cat(logw)
                logp = torch.logsumexp(logw, dim=-1) - math.log(len(logw))
                loss += logp.item()
                torch.cuda.empty_cache()

    return Statistics(loss=-loss, n_words=n_words, n_sents=n_sents)


def main():
    """Main workflow"""
    args = utils.build_test_args(argparse.ArgumentParser())

    suff = ".test"
    if args.report_iw_nll:
        if (
            args.num_iw_samples > args.iw_batch_size
            and args.num_iw_samples % args.iw_batch_size != 0
        ):
            raise RuntimeError("Expected num_iw_samples divisible by iw_batch_size")
        suff = ".test.iw" + str(args.num_iw_samples)

    utils.init_logger(args.model_file + suff)
    logger.info("Config:\n%s", pformat(vars(args)))

    assert torch.cuda.is_available()
    torch.cuda.set_device(args.gpuid)

    utils.init_random(args.seed)

    logger.info("Load parameters from '%s'", args.model_file)
    params = torch.load(args.model_file, map_location=lambda storage, loc: storage)

    utils.set_params(params["args"])

    fields = utils.load_fields_from_vocab(params["vocab"])
    logger.info("Fields: %s", fields.keys())

    model = utils.build_test_model(fields, params)
    logger.info("Model:\n%s", model)

    logger.info("Load %s", args.test_file)
    test_data = LMDataset(fields, args.test_file, args.sent_length_trunc)
    logger.info("Test sentences: %d", len(test_data))

    test_iter = utils.OrderedIterator(
        dataset=test_data,
        batch_size=args.batch_size,
        device=params["args"].device,
        train=False,
        shuffle=False,
        repeat=False,
        sort=False,
        sort_within_batch=True,
    )

    if model.encoder is None:
        args.report_iw_nll = False
        logger.info("Force report_iw_nll to False")

    start_time = time.time()
    logger.info("Start testing")
    if args.report_iw_nll:
        if args.num_iw_samples <= args.iw_batch_size:
            n_iw_iter = 1
        else:
            n_iw_iter = args.num_iw_samples // args.iw_batch_size
            args.num_iw_samples = args.iw_batch_size

        test_stats = report_iw_nll(model, test_iter, n_iw_iter, args.num_iw_samples)
        logger.info(
            "Results: test nll %.2f | test ppl %.2f", test_stats.nll(), test_stats.ppl()
        )
    else:
        test_stats = validate(model, test_iter)
        logger.info(
            "Results: test nll %.2f | test kl %.2f | test ppl %.2f",
            test_stats.nll(),
            test_stats.kl(),
            test_stats.ppl(),
        )

    logger.info("End of testing: time %.1f min", (time.time() - start_time) / 60)


if __name__ == "__main__":
    main()
