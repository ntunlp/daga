"""Classifier module."""
import argparse
import io
import os
import logging
import time
from pathlib import Path
import torch
import flair
import utils
from flair.data import Corpus
from flair.models import TextClassifier
from flair.datasets import FlyClassificationCorpus, ClassificationCorpus
from flair.training_utils import add_file_handler

logger = logging.getLogger("flair")


def predict(args):
    """Predict."""
    model = TextClassifier.load(os.path.join(args.model_dir, args.model_file))
    logger.info(f'Model: "{model}"')

    if args.one_per_line:
        corpus: Corpus = ClassificationCorpus(
            args.data_dir, test_file=args.test_file,
        )
    else:
        assert args.label_symbol is not None
        corpus: Corpus = FlyClassificationCorpus(
            args.data_dir,
            test_file=args.test_file,
            comment_symbol=args.comment_symbol,
            label_symbol=args.label_symbol,
        )

    fout = io.open(args.output_file, "w", encoding="utf-8", errors="ignore")
    logger.info("Saving to %s", args.output_file)

    start_time = time.time()
    for i in range(len(corpus.test)):
        sentence = corpus.test[i]
        model.predict(sentence)
        if sentence.labels:
            top = sentence.labels[0]
            fout.write(f"{top.value} {top.score:.4f}\n")
            fout.flush()

    logger.info("End of prediction: time %.1f min", (time.time() - start_time) / 60)


def main():
    """Main workflow."""
    args = utils.build_predict_args(argparse.ArgumentParser())
    log_handler = add_file_handler(logger, Path(args.output_file).with_suffix(".log"))
    utils.init_random(args.seed)

    with torch.cuda.device(args.gpuid):
        flair.device = torch.device(f"cuda:{args.gpuid}")
        predict(args)

    logger.removeHandler(log_handler)


if __name__ == "__main__":
    main()
