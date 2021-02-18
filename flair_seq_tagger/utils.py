"""Utilities module."""
import io
import logging
import random
import numpy as np
import torch
from torch.nn.init import xavier_uniform_
import tqdm
from gensim.models.keyedvectors import Word2VecKeyedVectors
from gensim.models.keyedvectors import Vocab
from flair.embeddings import CustomWordEmbeddings
from flair.models import TextClassifier

logger = logging.getLogger("flair")


def build_train_args(parser):
    """Build train arguments."""
    # Data
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--embeddings_file", type=str, default=None)
    parser.add_argument("--pretrained_file", type=str, default=None)
    parser.add_argument("--vector_size", type=int, default=300)
    parser.add_argument("--train_file", type=str, default="train.txt")
    parser.add_argument("--dev_file", type=str, default=None)
    parser.add_argument("--comment_symbol", type=str, default=None)
    parser.add_argument("--label_symbol", type=str, default=None)
    parser.add_argument("--one_per_line", action="store_true")
    # Model
    parser.add_argument("--param_init", type=float, default=0.1)
    parser.add_argument("--param_init_glorot", action="store_true")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--mini_batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--anneal_factor", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_learning_rate", type=float, default=0.0001)
    parser.add_argument(
        "--embeddings_storage_mode",
        type=str,
        default="cpu",
        choices=["none", "cpu", "gpu"],
    )
    # Attention
    parser.add_argument("--use_attn", action="store_true")
    parser.add_argument(
        "--attn_type", type=str, default="self", choices=["self", "soft"]
    )
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument(
        "--scaling", type=str, default="no", choices=["yes", "no", "learned"]
    )

    parser.add_argument(
        "--pooling_operation", type=str, default="none", choices=["mean", "max", "none"]
    )
    parser.add_argument("--use_sent_query", action="store_true")
    # Optim
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--optim", type=str, default="sgd", choices=["sgd", "adam"])
    # Misc.
    parser.add_argument("--seed", type=int, default=3435)
    parser.add_argument("--gpuid", type=int, default=0)
    # Tagger
    parser.add_argument("--data_columns", nargs="+", default=[])
    parser.add_argument(
        "--tag_type", type=str, choices=["pos", "np", "ner"], default=None
    )
    return parser.parse_args()


def build_predict_args(parser):
    """Build predict arguments."""
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_file", type=str, default="best-model.pt")
    parser.add_argument("--test_file", type=str, default="test.txt")
    parser.add_argument("--output_file", type=str, default="pred.out")
    parser.add_argument("--comment_symbol", type=str, default=None)
    parser.add_argument("--label_symbol", type=str, default=None)
    parser.add_argument("--one_per_line", action="store_true")
    parser.add_argument("--seed", type=int, default=3435)
    parser.add_argument("--gpuid", type=int, default=0)
    return parser.parse_args()


def init_random(seed):
    """Initialize random."""
    if seed > 0:
        logger.info("Set random seed to %d", seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def init_embeddings(vocab, args):
    """Initialize embeddings."""
    assert isinstance(vocab, list)
    vocab_size = len(vocab)
    vector_size = args.vector_size
    result = Word2VecKeyedVectors(vector_size)
    result.vector_size = vector_size
    for word_id, word in enumerate(vocab):
        result.vocab[word] = Vocab(index=word_id, count=vocab_size - word_id)
        result.index2word.append(word)
    if args.param_init != 0.0:
        logger.info(
            f"Initialize embeddings {vocab_size}x{vector_size} in U(-{args.param_init}, {args.param_init})"
        )
        result.vectors = np.random.uniform(
            low=-args.param_init, high=args.param_init, size=(vocab_size, vector_size)
        )

    if args.param_init_glorot:
        logger.info("Initialize model parameters with Glorot")
        np.random.uniform(
            -(np.sqrt(6.0 / (vocab_size + vector_size))),
            (np.sqrt(6.0 / (vocab_size + vector_size))),
            size=(vocab_size, vector_size),
        )

    return CustomWordEmbeddings(args.embeddings_file, result)


def optim_method(name):
    """Get optimizer."""
    method = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}
    return method[name]


def init_word_embeddings(embeddings, args):
    """Initialize word embeddings."""
    vocab_size = len(embeddings.vocab)
    logger.info(
        "Loading pre-trained embeddings from %s for %d words",
        args.embeddings_file,
        vocab_size,
    )

    # Quick load all lines from file.
    with io.open(args.embeddings_file, "r", encoding="utf-8", errors="ignore") as fin:
        lines = [line for line in fin]

    cnt = 0
    dim = None
    for line in tqdm.tqdm(lines, total=len(lines)):
        # Quick split once for checking.
        word, entries = line.rstrip().split(" ", 1)
        if word in embeddings.vocab:
            entries = entries.split(" ")
            if dim is None and len(entries) > 1:
                dim = len(entries)
                assert dim == embeddings.vector_size
            elif len(entries) == 1:
                # word2vec style
                continue
            elif dim != len(entries):
                raise RuntimeError(
                    "Vector for word '{}' has '{}' dimensions, "
                    "expected '{}' dimensions".format(word, len(entries), dim)
                )

            # Convert floats to tensor.
            weights = np.array([float(x) for x in entries], dtype=np.float32)

            # Copy to model.
            word_index = embeddings.vocab[word].index
            embeddings.vectors[word_index] = weights
            cnt += 1

    assert cnt > 0
    logger.info(
        "Found {:.2f}% ({}/{})".format(100.0 * cnt / vocab_size, cnt, vocab_size)
    )


def init_model_params(model, args):
    """Initialize model params."""
    if args.param_init != 0.0:
        logger.info(
            f"Initialize model parameters in U(-{args.param_init}, {args.param_init})"
        )
        for param in model.parameters():
            param.data.uniform_(-args.param_init, args.param_init)

    if args.param_init_glorot:
        logger.info("Initialize model parameters with Glorot")
        for param in model.parameters():
            if param.dim() > 1:
                xavier_uniform_(param)


def init_model(model, args):
    """Initialize model."""
    init_model_params(model, args)

    if args.embeddings_file is not None:
        for name, _ in model.named_modules():
            if name == "embeddings":
                init_word_embeddings(model.embeddings.precomputed_word_embeddings, args)
                break
            if name == "document_embeddings":
                init_word_embeddings(
                    model.document_embeddings.embeddings.embeddings[
                        0
                    ].precomputed_word_embeddings,
                    args,
                )
                break


def init_joint_models(model1, model2, args):
    """Initialize joint models."""
    init_model_params(model1, args)
    init_model_params(model2, args)

    # Initialize embeddings for model2 only
    if args.embeddings_file is not None:
        init_word_embeddings(
            model2.document_embeddings.embeddings.embeddings[
                0
            ].precomputed_word_embeddings,
            args,
        )

    # Initialize model2 if pretrained model is given
    if args.pretrained_file is not None:
        pretrained_model = TextClassifier.load(args.pretrained_file)
        pretrained_dict = pretrained_model.state_dict()
        model2_dict = model2.state_dict()
        for name in model2_dict.keys():
            if name in pretrained_dict:
                logger.info(f"Use pretrained {name}")
                model2_dict[name].copy_(pretrained_dict[name])

    # Santity check for sizes
    assert (
        model1.embeddings.precomputed_word_embeddings.vectors.size
        == model2.document_embeddings.embeddings.embeddings[
            0
        ].precomputed_word_embeddings.vectors.size
    )

    for (name1, param1), (name2, param2) in list(
        zip(
            model1.embedding2nn.named_parameters(),
            model2.document_embeddings.word_reprojection_map.named_parameters(),
        )
    ):
        assert name1 == name2
        assert param1.size() == param2.size()

    for (name1, param1), (name2, param2) in list(
        zip(
            model1.rnn.named_parameters(),
            model2.document_embeddings.rnn.named_parameters(),
        )
    ):
        assert name1 == name2
        assert param1.size() == param2.size()

    # Share layers
    logger.info("Set shared layers: embeddings, projection, rnn")
    model1.embeddings = model2.document_embeddings.embeddings.embeddings[0]
    model1.embedding2nn = model2.document_embeddings.word_reprojection_map
    model1.rnn = model2.document_embeddings.rnn
