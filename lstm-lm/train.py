
"""Training module"""
from __future__ import print_function, division
import argparse
from pprint import pformat
import os
import sys
import time
import torch
import utils
from utils import logger
from data import LMDataset
from stats import Statistics, ESStatistics


def report_best(es_stats):
    """Report the best training statistics."""
    logger.info(
        "Best at epoch %d | valid nll %.2f | valid kl %.2f | " "valid ppl %.2f",
        es_stats.best_epoch,
        es_stats.best_nll,
        es_stats.best_kl,
        es_stats.best_ppl,
    )
    sys.stdout.flush()


def report_epoch(epoch, train_stats, val_stats, args):
    """Report statistics at the end of each epoch."""
    logger.info(
        "End of epoch %d | time %.0fs | "
        "train nll %.2f | train kl %.2f | train ppl %.2f | beta %.2f | "
        "valid nll %.2f | valid kl %.2f | valid ppl %.2f",
        epoch,
        time.time() - train_stats.start_time,
        train_stats.nll(),
        train_stats.kl(),
        train_stats.ppl(),
        args.beta,
        val_stats.nll(),
        val_stats.kl(),
        val_stats.ppl(),
    )
    sys.stdout.flush()


def report_batch(batch_stats, epoch, step, num_batches, args):
    """Report batch statistics."""
    if step % args.report_every == -1 % args.report_every:
        r = batch_stats
        t = r.elapsed_time()
        logger.info(
            "Epoch %3d | %4d/%4d batches | acc %5.2f | "
            "nll %8.2f | kl %6.2f | ppl %8.2f | "
            "beta %4.2f | %.0f tok/s",
            epoch,
            step + 1,
            num_batches,
            r.accuracy(),
            r.nll(),
            r.kl(),
            r.ppl(),
            args.beta,
            r.n_words / (t + 1e-5),
        )
        sys.stdout.flush()
        batch_stats = Statistics()
    return batch_stats


def validate(model, val_iter):
    """Validate with mini-batches."""
    model.eval()
    batch_stats = Statistics()
    with torch.no_grad():
        for batch in val_iter:
            sents = batch.sent
            _, stats = model(sents)
            batch_stats.update(stats)
            torch.cuda.empty_cache()
    return batch_stats


def train(model, optimizer, train_iter, epoch, args):
    """Train with mini-batches."""
    model.train()
    total_stats = Statistics()
    batch_stats = Statistics()
    num_batches = len(train_iter)
    for i, batch in enumerate(train_iter):
        if args.warmup > 0:
            args.beta = min(1, args.beta + 1.0 / (args.warmup * num_batches))

        sents = batch.sent
        loss, stats = model(sents, args.beta)
        optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm(optimizer, args)
        optimizer.step()
        total_stats.update(stats)
        batch_stats.update(stats)
        batch_stats = report_batch(batch_stats, epoch, i, num_batches, args)
        torch.cuda.empty_cache()
    return total_stats


def try_train_val(fields, model, optimizer, train_iter, val_iter, es_stats, args):
    """Try to train and validate given a number of epochs."""
    start_time = time.time()
    logger.info("Start training")
    try:
        for epoch in range(args.start_epoch, args.epochs + 1):
            if es_stats.bad_counter > args.patience:
                logger.info("Stop training early at epoch %d", epoch - 1)
                break

            train_stats = train(model, optimizer, train_iter, epoch, args)

            val_stats = validate(model, val_iter)

            logger.info("-" * 109)
            report_epoch(epoch, train_stats, val_stats, args)

            is_best = val_stats.ppl() + args.tol < es_stats.best_ppl
            if is_best:
                es_stats.update_best(epoch, val_stats)
                logger.info("New best at epoch %d", epoch)
            else:
                utils.decay_learning_rate(optimizer, args)

            es_stats.update_history(val_stats.ppl())
            utils.save_model(fields, model, optimizer, es_stats, epoch, is_best, args)
            logger.info("-" * 109)
    except KeyboardInterrupt:
        logger.info("Training interupted")

    report_best(es_stats)
    logger.info("End of training: time %.1f min", (time.time() - start_time) / 60)


def main():
    """Main workflow"""
    args = utils.build_args(argparse.ArgumentParser())

    utils.init_logger(args.model_file)

    assert torch.cuda.is_available()
    torch.cuda.set_device(args.gpuid)

    utils.init_random(args.seed)

    utils.set_params(args)
    logger.info("Config:\n%s", pformat(vars(args)))

    fields = utils.build_fields()
    logger.info("Fields: %s", fields.keys())

    logger.info("Load %s", args.train_file)
    train_data = LMDataset(fields, args.train_file, args.sent_length_trunc)
    logger.info("Training sentences: %d", len(train_data))
    logger.info("Load %s", args.valid_file)
    val_data = LMDataset(fields, args.valid_file, args.sent_length_trunc)
    logger.info("Validation sentences: %d", len(val_data))

    fields["sent"].build_vocab(train_data)

    train_iter = utils.build_dataset_iter(train_data, args)
    val_iter = utils.build_dataset_iter(val_data, args, train=False)

    if args.resume and os.path.isfile(args.checkpoint_file):
        logger.info("Resume training")
        logger.info("Load checkpoint %s", args.checkpoint_file)
        checkpoint = torch.load(
            args.checkpoint_file, map_location=lambda storage, loc: storage
        )
        es_stats = checkpoint["es_stats"]
        args = utils.set_args(args, checkpoint)
    else:
        checkpoint = None
        es_stats = ESStatistics(args)

    model = utils.build_model(fields, args, checkpoint)
    logger.info("Model:\n%s", model)

    optimizer = utils.build_optimizer(model, args, checkpoint)

    try_train_val(fields, model, optimizer, train_iter, val_iter, es_stats, args)


if __name__ == "__main__":
    main()
