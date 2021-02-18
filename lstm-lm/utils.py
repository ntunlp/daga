
"""Utilities modules"""
from __future__ import print_function
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torchtext
from model import LMModel


PAD_WORD = "<blank>"
UNK_WORD = "<unk>"
UNK = 0
BOS_WORD = "<s>"
EOS_WORD = "</s>"
logger = logging.getLogger(__name__)


def build_args(parser):
    """Build arguments."""
    # File
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_file", type=str, required=True)
    parser.add_argument("--model_file", type=str, default="model.pt")
    parser.add_argument("--resume", action="store_true")
    # Model
    parser.add_argument("--emb_dim", type=int, default=512)
    parser.add_argument("--bidirectional_encoder", action="store_true")
    parser.add_argument("--rnn_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=-1)
    parser.add_argument("--num_enc_layers", type=int, default=1)
    parser.add_argument("--num_dec_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--param_init", type=float, default=0.1)
    # VAE
    parser.add_argument("--num_z_samples", type=int, default=0)
    parser.add_argument("--z_dim", type=int, default=32)
    parser.add_argument("--z_cat", action="store_true")
    parser.add_argument("--use_avg", action="store_true")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--word_dropout_rate", type=float, default=0.0)
    parser.add_argument("--inputless", action="store_true")
    # Optimizer
    parser.add_argument("--optim", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    # Misc.
    parser.add_argument("--sent_length_trunc", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--tol", type=int, default=0.01)
    parser.add_argument("--seed", type=int, default=3435)
    parser.add_argument("--report_every", type=int, default=100)
    parser.add_argument("--gpuid", type=int, default=0)
    return parser.parse_args()


def build_test_args(parser):
    """Build test arguments."""
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sent_length_trunc", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3435)
    parser.add_argument("--report_iw_nll", action="store_true")
    parser.add_argument("--num_iw_samples", type=int, default=500)
    parser.add_argument("--iw_batch_size", type=int, default=500)
    parser.add_argument("--gpuid", type=int, default=0)
    return parser.parse_args()


def build_gen_args(parser):
    """Build gen arguments."""
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--num_sentences", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--max_sent_length", type=int, default=50)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3435)
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser.parse_args()


def init_logger(filename):
    """Initialize logger."""
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = logging.FileHandler("{}.log".format(filename), "w")
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)


def init_random(seed):
    """Initialize random."""
    if seed > 0:
        logger.info("Set random seed to %d", seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def set_params(args):
    """Set some params."""
    args.checkpoint_file = "{}.checkpoint".format(args.model_file)

    if args.num_layers != -1:
        args.num_enc_layers = args.num_layers
        args.num_dec_layers = args.num_layers
        logger.info(
            "Set number of encoder/decoder layers uniformly to %d", args.num_layers
        )

    if args.num_enc_layers < args.num_dec_layers:
        raise RuntimeError("Expected num_enc_layers >= num_dec_layers")

    if args.num_z_samples == 0:
        args.z_dim = 0
        args.z_cat = False
        args.warmup = 0

    args.beta = 1.0 if args.warmup == 0 else 0.0

    args.device = "cuda" if args.gpuid > -1 else "cpu"


def build_fields():
    """Build fields."""
    fields = {}
    fields["sent"] = torchtext.data.Field(
        init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=PAD_WORD
    )
    return fields


class OrderedIterator(torchtext.data.Iterator):
    """Define an ordered iterator class.
       This class is retrieved from https://github.com/OpenNMT/OpenNMT-py.
       Reference:
       Guillaume Klein, Yoon Kim, Yuntian Deng, Jean Senellart and
       Alexander M. Rush. 2017.  OpenNMT: Open-Source Toolkit for
       Neural Machine Translation. In Proceedings of ACL.
    """

    def create_batches(self):
        """Create batches."""
        if self.train:

            def _pool(data, random_shuffler):
                for p in torchtext.data.batch(data, self.batch_size * 100):
                    p_batch = torchtext.data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size,
                        self.batch_size_fn,
                    )
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = _pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in torchtext.data.batch(
                self.data(), self.batch_size, self.batch_size_fn
            ):
                self.batches.append(sorted(b, key=self.sort_key))


def build_dataset_iter(data, args, train=True, shuffle=True):
    """Build dataset iterator."""
    return OrderedIterator(
        dataset=data,
        batch_size=args.batch_size,
        device=args.device,
        train=train,
        shuffle=shuffle,
        repeat=False,
        sort=False,
        sort_within_batch=True,
    )


def build_model(fields, args, checkpoint=None):
    """Build model."""
    model = LMModel(fields, args)
    if checkpoint is not None:
        logger.info("Set model using saved checkpoint")
        model.load_state_dict(checkpoint["model"])
    return model.to(args.device)


def build_test_model(fields, params):
    """Build test model."""
    model = LMModel(fields, params["args"])
    model.load_state_dict(params["model"])
    model = model.to(params["args"].device)
    model.eval()
    return model


def build_optimizer(model, args, checkpoint=None):
    """Build optimizer."""
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum([p.nelement() for p in params])
    logger.info("Trainable parameters: %d", n_params)

    method = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}
    optimizer = method[args.optim](params, lr=args.lr)
    logger.info("Use %s with lr %f", args.optim, args.lr)

    if checkpoint is not None:
        logger.info("Set optimizer states using saved checkpoint")
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(args.device)
    return optimizer


def clip_grad_norm(optimizer, args):
    """Clip gradient norm."""
    for group in optimizer.param_groups:
        nn.utils.clip_grad_norm_(group["params"], args.max_grad_norm)


def decay_learning_rate(optimizer, args):
    """Decay learning rate."""
    args.lr = args.lr * args.lr_decay
    optimizer.param_groups[0]["lr"] = args.lr
    logger.info("Decay learning rate to %f", args.lr)


def load_fields_from_vocab(vocab):
    """Load Field objects from Vocab objects."""
    vocab = dict(vocab)
    fields = build_fields()
    for k, v in vocab.items():
        fields[k].vocab = v
    return fields


def save_fields_to_vocab(fields):
    """Save Vocab objects in Field objects."""
    vocab = []
    for k, f in fields.items():
        if f is not None and "vocab" in f.__dict__:
            vocab.append((k, f.vocab))
    return vocab


def get_model_state_dict(model):
    """Get model state dict."""
    return {k: v for k, v in model.state_dict().items()}


def save_model(fields, model, optimizer, es_stats, epoch, is_best, args):
    """Save model."""
    try:
        state = {
            "args": args,
            "model": get_model_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "es_stats": es_stats,
            "last_epoch": epoch + 1,
        }
        torch.save(state, args.checkpoint_file)
    except BaseException:
        logger.warning("Saving '%s' failed", args.checkpoint_file)

    if is_best:
        try:
            torch.save(
                {
                    "args": args,
                    "vocab": save_fields_to_vocab(fields),
                    "model": state["model"],
                },
                args.model_file,
            )
        except BaseException:
            logger.warning("Saving '%s' failed", args.model_file)


def set_args(args, checkpoint):
    """Set arguments from checkpoint."""
    # Prefer newer number of epochs if given
    epochs = args.epochs
    args = checkpoint["args"]
    args.epochs = epochs
    args.start_epoch = checkpoint["last_epoch"]
    if args.epochs < args.start_epoch:
        raise RuntimeError(
            "args.epochs (%d) < args.start_epoch (%d)" % (args.epochs, args.start_epoch)
        )
    logger.info(
        "Resume at epoch %d; number of epochs: %d", args.start_epoch, args.epochs
    )
    return args
