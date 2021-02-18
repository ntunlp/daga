"""Classifier evaluation module."""
import argparse
import io
from sklearn.metrics import classification_report


def parse_file(fname, is_true=True):
    """Parse file to get labels."""
    labels = []
    with io.open(fname, "r", encoding="utf-8", errors="igore") as fin:
        for line in fin:
            label = line.strip().split()[0]
            if is_true:
                assert label[:9] == "__label__"
                label = label[9:]
            labels.append(label)
    return labels


def build_args(parser):
    """Buikd arguments."""
    parser.add_argument("--true", type=str, required=True)
    parser.add_argument("--pred", type=str, required=True)
    return parser.parse_args()


def main():
    """Main workflow."""
    args = build_args(argparse.ArgumentParser())
    y_true = parse_file(args.true)
    y_pred = parse_file(args.pred, is_true=False)
    assert len(y_true) == len(y_pred)
    print(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    main()
