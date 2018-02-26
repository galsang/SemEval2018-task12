import argparse
import copy

from torch import nn, optim

from tensorboardX import SummaryWriter
from time import gmtime, strftime

from model.data import ARC
from model.model import SECOVARC
from test import test


def train(args, data):
    model = SECOVARC(args, data)
    if args.gpu > -1:
        model.cuda(args.gpu)

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = getattr(optim, args.optim)(parameters, lr=args.learning_rate, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    writer = SummaryWriter(log_dir='runs/' + args.model_time)

    model.train()
    loss, last_epoch = 0, -1
    max_dev_acc, max_test_acc = 0, 0

    iterator = data.train_iter
    for i, batch in enumerate(iterator):
        present_epoch = int(iterator.epoch)
        if present_epoch == args.epoch:
            break
        if present_epoch > last_epoch:
            print('epoch:', present_epoch + 1)
        last_epoch = present_epoch

        pred = model(batch).squeeze()

        optimizer.zero_grad()
        batch_loss = criterion(pred, batch.label)
        loss += batch_loss.data[0]
        batch_loss.backward()
        optimizer.step()

        if (i + 1) % args.print_freq == 0:
            dev_loss, dev_acc, single_dev_acc = test(model, data, mode='dev')
            test_loss, test_acc, single_test_acc = test(model, data, mode='test')
            c = (i + 1) // args.print_freq

            writer.add_scalar('loss/train', loss, c)
            writer.add_scalar('loss/dev', dev_loss, c)
            writer.add_scalar('loss/test', test_loss, c)
            writer.add_scalar('acc/dev', dev_acc, c)
            writer.add_scalar('acc/test', test_acc, c)
            writer.add_scalar('single_acc/dev', single_dev_acc, c)
            writer.add_scalar('single_acc/test', single_test_acc, c)

            print(f'train loss: {loss:.3f} / dev loss: {dev_loss:.3f} / test loss: {test_loss:.3f} / '
                  f'dev acc: {dev_acc:.3f} / test acc: {test_acc:.3f} /\n'
                  f'single dev acc: {single_dev_acc:.3f} / single test acc: {single_test_acc:.3f}')

            if dev_acc > max_dev_acc:
                max_dev_acc = dev_acc
                max_test_acc = test_acc
                best_model = copy.deepcopy(model)

            loss = 0
            model.train()

    writer.close()
    print(f'max dev acc: {max_dev_acc:.3f} / max test acc: {max_test_acc:.3f}')
    return best_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-size', default=300, type=int)
    parser.add_argument('--heuristics', default=False, action='store_true')
    parser.add_argument('--learning-rate', default=0.001, type=float)
    parser.add_argument('--model', default='cove',
                        help='available: bow, lstm, cove')
    parser.add_argument('--optim', default='Adam')
    parser.add_argument('--print-freq', default=1, type=int)
    parser.add_argument('--pooling', default='max',
                        help='available: max, last, average, min')
    parser.add_argument('--word_dim', default=300, type=int)

    args = parser.parse_args()
    setattr(args, 'model_time', strftime('%H:%M:%S', gmtime()))

    print('loading data...')
    data = ARC(args)

    setattr(args, 'word_vocab_size', len(data.TEXT.vocab))

    print('training started...')
    best_model = train(args, data)
    print('training finished!')

if __name__ == '__main__':
    main()
