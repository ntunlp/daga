"""Classifier training module."""
import argparse
import logging
import time
from pathlib import Path
from pprint import pformat
import torch
import flair
import utils
from flair.data import Corpus
from flair.models import TextClassifier
from flair.embeddings import DocumentRNNEmbeddings
from flair.datasets import FlyClassificationCorpus, ClassificationCorpus
from flair.trainers import ModelTrainer
from flair.training_utils import add_file_handler

logger = logging.getLogger("flair")


def train(args):
    """Train."""
    start_time = time.time()
    if args.one_per_line:
        corpus: Corpus = ClassificationCorpus(
            args.data_dir, train_file=args.train_file, dev_file=args.dev_file,
        )
    else:
        assert args.label_symbol is not None
        corpus: Corpus = FlyClassificationCorpus(
            args.data_dir,
            train_file=args.train_file,
            dev_file=args.dev_file,
            comment_symbol=args.comment_symbol,
            label_symbol=args.label_symbol,
        )

    label_dict = corpus.make_label_dictionary()
    vocab = corpus.make_vocab_dictionary().get_items()
    embeddings = utils.init_embeddings(vocab, args)

    document_embeddings = DocumentRNNEmbeddings(
        [embeddings],
        hidden_size=args.hidden_size,
        use_attn=args.use_attn,
        num_heads=args.num_heads,
        scaling=args.scaling,
        pooling_operation=args.pooling_operation,
        use_sent_query=args.use_sent_query,
    )

    model = TextClassifier(document_embeddings, label_dictionary=label_dict)

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

    log_handler = add_file_handler(logger, Path(args.model_dir) / "training.log")

    logger.info(f"Args: {pformat(vars(args))}")

    utils.init_random(args.seed)

    with torch.cuda.device(args.gpuid):
        flair.device = torch.device(f"cuda:{args.gpuid}")
        train(args)

    logger.removeHandler(log_handler)


if __name__ == "__main__":
    main()
