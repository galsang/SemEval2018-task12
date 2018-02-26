import torch

from torchtext import data
from torchtext.vocab import GloVe

from nltk import word_tokenize


class BASE():
    def __init__(self, args):
        self.args = args

        self.RAW = data.RawField()
        self.TEXT = data.Field(batch_first=True, tokenize=word_tokenize, include_lengths=True, lower=True)

class ARC(BASE):
    def __init__(self, args):
        super(ARC, self).__init__(args)
        self.LABEL = data.Field(sequential=False, unk_token=None, tensor_type=torch.FloatTensor)

        self.train, self.dev, self.test = data.TabularDataset.splits(
            path='.data/arc/preprocessed/single',
            train='train.txt',
            validation='dev.txt',
            test='test.txt',
            format='tsv',
            skip_header=True,
            fields=[('id', self.RAW),
                    ('warrant', self.TEXT),
                    ('label', self.LABEL),
                    ('reason', self.TEXT),
                    ('claim', self.TEXT),
                    ('debateTitle', self.TEXT),
                    ('debateInfo', self.TEXT)])

        self.TEXT.build_vocab(self.train, self.dev, self.test, vectors=GloVe(name='840B', dim=300))
        self.LABEL.build_vocab(self.train)

        self.sort_key = lambda x: len(x.warrant) + len(x.reason) + len(x.claim)
        self.train_iter, self.dev_iter, self.test_iter = data.Iterator.splits(
            (self.train, self.dev, self.test),
            batch_sizes=[self.args.batch_size, 256, 256],
            device=self.args.gpu,
            sort_key=self.sort_key)

        self.dev_iter.sort = False
        self.dev_iter.sort_within_batch = False
        self.test_iter.sort = False
        self.test_iter.sort_within_batch = False

