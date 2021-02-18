"""Tagger training module."""
import argparse
import logging
import time
from pathlib import Path
from pprint import pformat
import torch
import flair
import utils
from flair.data import Corpus
from flair.models import SequenceTagger
from flair.datasets import ColumnCorpus
from flair.trainers import ModelTrainer
from flair.training_utils import add_file_handler

logger = logging.getLogger("flair")


def train(args):
    """Train."""
    start_time = time.time()
    column_format = {i: col for i, col in enumerate(args.data_columns)}
    corpus: Corpus = ColumnCorpus(
        args.data_dir,
        column_format,
        train_file=args.train_file,
        dev_file=args.dev_file,
        comment_symbol=args.comment_symbol,
    )

    tag_type = args.data_columns[-1]
    tag_dict = corpus.make_tag_dictionary(tag_type=tag_type)
    vocab = corpus.make_vocab_dictionary().get_items()
    embeddings = utils.init_embeddings(vocab, args)

    model: SequenceTagger = SequenceTagger(
        hidden_size=args.hidden_size,
        embeddings=embeddings,
        tag_dictionary=tag_dict,
        tag_type=tag_type,
        column_format=column_format,
        use_crf=True,
        use_attn=args.use_attn,
        attn_type=args.attn_type,
        num_heads=args.num_heads,
        scaling=args.scaling,
        pooling_operation=args.pooling_operation,
        use_sent_query=args.use_sent_query,
    )

    utils.init_model(model, args)

    trainer: ModelTrainer = ModelTrainer(model, corpus, utils.optim_method(args.optim))

    trainer.train(
        args.model_dir,
        mini_batch_size=args.mini_batch_size,
        max_epochs=args.max_epochs,
        anneal_factor=args.anneal_factor,
        learning_rate=args.learning_rate,
        patience=args.patience,
        min_learning_rate=args.min_learning_rate,
        embeddings_storage_mode=args.embeddings_storage_mode,
    )

    logger.info("End of training: time %.1f min", (time.time() - start_time) / 60)


def main():
    """Main workflow."""
    args = utils.build_train_args(argparse.ArgumentParser())
    if not args.data_columns:
        raise RuntimeError("Specify data_column (e.g., text ner)")

    log_handler = add_file_handler(logger, Path(args.model_dir) / "training.log")

    logger.info(f"Args: {pformat(vars(args))}")

    utils.init_random(args.seed)

    with torch.cuda.device(args.gpuid):
        flair.device = torch.device(f"cuda:{args.gpuid}")
        train(args)

    logger.removeHandler(log_handler)


if __name__ == "__main__":
    main()
