import os
import shutil


def preprocess():
    """
    prerequisites:
        - train data: train-full.txt
        - dev data: dev-full.txt
        - test data: test-only-data.txt, truth.txt
    :return: preprocessed/{train.txt, dev.txt, test.txt}
    """

    if not os.path.exists('.data/arc/preprocessed'):
        os.makedirs('.data/arc/preprocessed')
    if not os.path.exists('.data/arc/preprocessed/single'):
        os.makedirs('.data/arc/preprocessed/single')
    if not os.path.exists('.data/arc/preprocessed/double'):
        os.makedirs('.data/arc/preprocessed/double')

    manipulate_data('train')
    manipulate_data('dev')
    manipulate_data('test')


def manipulate_data(mode):
    prefix = '.data/arc'
    if mode == 'test':
        with open(f'{prefix}/test-only-data.txt', 'r', encoding='utf-8') as r1:
            with open(f'{prefix}/truth.txt', 'r', encoding='utf-8') as r2:
                with open(f'{prefix}/preprocessed/single/test.txt', 'w', encoding='utf-8') as w:
                    with open(f'{prefix}/preprocessed/double/test.txt', 'w', encoding='utf-8') as w2:
                        r1.readline()
                        r2.readline()
                        print('#id\twarrant\tlabel\treason\tclaim\tdebateTitle\tdebateInfo', file=w)
                        print('#id\twarrant0\twarrant1\tcorrectLabelW0orW1\treason\tclaim\tdebateTitle\tdebateInfo',
                              file=w2)

                        data = r1.readlines()
                        labels = r2.readlines()

                        for i in range(len(data)):
                            id, w0, w1, reason, claim, debateTitle, debateInfo = data[i].strip().split('\t')
                            id, label = labels[i].strip().split('\t')

                            if int(label) == 0:
                                tw, fw = w0, w1
                            else:
                                tw, fw = w1, w0

                            print(f'{id}_1\t{tw}\t1\t{reason}\t{claim}\t{debateTitle}\t{debateInfo}', file=w)
                            print(f'{id}_0\t{fw}\t0\t{reason}\t{claim}\t{debateTitle}\t{debateInfo}', file=w)
                            print(f'{id}\t{w0}\t{w1}\t{label}\t{reason}\t{claim}\t{debateTitle}\t{debateInfo}', file=w2)
    else:
        with open(f'{prefix}/{mode}-full.txt', 'r', encoding='utf-8') as r:
            with open(f'{prefix}/preprocessed/single/{mode}.txt', 'w', encoding='utf-8') as w:
                r.readline()
                print('#id\twarrant\tlabel\treason\tclaim\tdebateTitle\tdebateInfo', file=w)

                for line in r:
                    id, w0, w1, label, reason, claim, debateTitle, debateInfo = line.strip().split('\t')

                    if int(label) == 0:
                        tw, fw = w0, w1
                    else:
                        tw, fw = w1, w0

                    print(f'{id}_1\t{tw}\t1\t{reason}\t{claim}\t{debateTitle}\t{debateInfo}', file=w)
                    print(f'{id}_0\t{fw}\t0\t{reason}\t{claim}\t{debateTitle}\t{debateInfo}', file=w)

        shutil.copy(f'{prefix}/{mode}-full.txt', f'{prefix}/preprocessed/double/{mode}.txt')


if __name__ == '__main__':
    preprocess()
