
"""Generating module"""
from __future__ import print_function
import argparse
import io
import math
from pprint import pformat
import torch
import utils
from utils import logger


def save(samples, fields, out_file):
    """Save samples to file."""
    logger.info("Save to %s", out_file)
    vocab = fields["sent"].vocab
    with io.open(out_file, "w", encoding="utf-8", errors="ignore") as f:
        for sample in samples:
            tokens = []
            for word_idx in sample:
                token = vocab.itos[word_idx]
                if token == fields["sent"].eos_token:
                    break
                tokens.append(token)
            f.write(u" ".join(tokens))
            f.write(u"\n")
            f.flush()


def main():
    """Main workflow"""
    args = utils.build_gen_args(argparse.ArgumentParser())

    utils.init_logger(args.out_file)
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

    sent_idx = [i for i in range(args.num_sentences)]
    num_batches = math.ceil(float(args.num_sentences) / args.batch_size)
    samples = []
    with torch.no_grad():
        for i in range(num_batches):
            running_batch_size = len(
                sent_idx[i * args.batch_size : (i + 1) * args.batch_size]
            )
            samples.append(
                model.generate(
                    running_batch_size, args.max_sent_length, args.temperature
                )
            )
    samples = torch.cat(samples, 0)
    save(samples, fields, args.out_file)


if __name__ == "__main__":
    main()
