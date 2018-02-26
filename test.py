import argparse

import torch
from torch import nn

from model.model import SECOVARC
from model.data import ARC


def test(model, data, mode='test'):
    if mode == 'dev':
        iterator = iter(data.dev_iter)
    else:
        iterator = iter(data.test_iter)

    criterion = nn.BCEWithLogitsLoss()

    model.eval()
    acc, loss, size = 0, 0, 0
    single_acc = 0
    for batch in iterator:
        pred = model(batch).squeeze()
        batch_loss = criterion(pred, batch.label)
        loss += batch_loss.data[0]

        single_acc += ((pred > 0).long() == batch.label.long()).sum().float()
        acc += (pred[::2] > pred[1::2]).sum().float()
        size += len(pred) // 2

    acc /= size
    acc = acc.cpu().data[0]
    single_acc /= (size * 2)
    single_acc = single_acc.cpu().data[0]

    return loss, acc, single_acc