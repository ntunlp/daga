
"""Language modeling dataset"""
import io
import torchtext


class LMDataset(torchtext.data.Dataset):
    """Define a dataset class."""

    def __init__(self, fields, filename, truncate=0):
        sents = []
        with io.open(filename, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip().split(" ")
                if truncate:
                    line = line[:truncate]
                sents += [line]
        fields = [(k, fields[k]) for k in fields]
        examples = [torchtext.data.Example.fromlist([sent], fields) for sent in sents]
        super(LMDataset, self).__init__(examples, fields)

    def sort_key(self, ex):
        """Sort by sentence length."""
        return len(ex.sent)
