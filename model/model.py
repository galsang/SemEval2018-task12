import torch
import torch.nn as nn
import torch.nn.functional as F

from third_party.cove.cove import MTLSTM


class BASE(nn.Module):
    def __init__(self, args, data):
        super(BASE, self).__init__()

        self.args = args

        if args.model == 'cove':
            self.cove = MTLSTM(n_vocab=args.word_vocab_size, vectors=data.TEXT.vocab.vectors)
        else:
            self.word_emb = nn.Embedding(args.word_vocab_size, args.word_dim)
            self.word_emb.weight.data.copy_(data.TEXT.vocab.vectors)
            # <unk> vectors is randomly initialized
            nn.init.uniform(self.word_emb.weight.data[0], -0.05, 0.05)
            # self.word_emb.weight.requires_grad = False

            if args.model == 'lstm':
                self.rnn = nn.LSTM(300, 300, num_layers=2, bidirectional=True, batch_first=True)
            elif args.model == 'bow':
                pass
            else:
                raise (NotImplementedError('!'))

        in_dim = 300 * (1 + int(args.model == 'lstm' or args.model == 'cove'))
        self.enc_w = self.customizedLinear(in_dim, args.hidden_size)
        self.enc_r = self.customizedLinear(in_dim, args.hidden_size)
        self.enc_c = self.customizedLinear(in_dim, args.hidden_size)

    def customizedLinear(self, in_dim, out_dim, activation=nn.Tanh()):
        cl = nn.Sequential(nn.Linear(in_dim, out_dim))
        nn.init.uniform(cl[0].weight, -0.005, 0.005)
        nn.init.constant(cl[0].bias, 0)

        if activation is not None:
            cl.add_module(str(len(cl)), activation)

        return cl

    def dropout(self, x):
        return F.dropout(x, p=self.args.dropout, training=self.training)

    def last_pooling(self, x):
        """
        :param x: (2, batch, hidden_size)
        :return: (batch, hidden_size * 2)
        """
        return torch.cat([x[:, -1, :(x.size(2) // 2)], x[:, 0, (x.size(2) // 2):]], dim=1)

    def max_pooling(self, v):
        """
        :param v: (batch, seq_len, hidden_size)
        :return: (batch, hidden_size)
        """
        return v.max(dim=1)[0]

    def min_pooling(self, v):
        return v.min(dim=1)[0]

    def average_pooling(self, v):
        """
        :param v: (batch, seq_len, hidden_size)
        :return: (batch, hidden_size)
        """
        return v.mean(dim=1)


class SECOVARC(BASE):
    def __init__(self, args, data):
        super(SECOVARC, self).__init__(args, data)
        self.fc = self.customizedLinear(args.hidden_size * (3 + int(self.args.heuristics) * 2), 1, activation=None)

    def forward(self, batch):
        dropout = self.dropout
        pooling = getattr(self, f'{self.args.pooling}_pooling')

        if self.args.model == 'cove':
            w = dropout(pooling(self.cove(*batch.warrant)))
            c = dropout(pooling(self.cove(*batch.claim)))
            r = dropout(pooling(self.cove(*batch.reason)))
        else:
            w = self.word_emb(batch.warrant[0])
            c = self.word_emb(batch.claim[0])
            r = self.word_emb(batch.reason[0])
            if self.args.model == 'lstm':
                w = dropout(pooling(self.rnn(w)[0]))
                c = dropout(pooling(self.rnn(c)[0]))
                r = dropout(pooling(self.rnn(r)[0]))
            elif self.args.model == 'bow':
                w = dropout(w.mean(dim=1))
                c = dropout(c.mean(dim=1))
                r = dropout(r.mean(dim=1))
            else:
                raise (NotImplementedError('!'))

        w_final = self.enc_w(w)
        c_final = self.enc_c(c)
        r_final = self.enc_r(r)

        x = torch.cat([w_final, r_final, c_final], dim=1)
        if self.args.heuristics:
            x = torch.cat([x, torch.abs(w_final - r_final - c_final), w_final * r_final * c_final], dim=1)
        x = dropout(x)
        x = self.fc(x)
        return x
