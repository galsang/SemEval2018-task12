# SemEval2018-task12

This is the implementation of **SECOVARC** (The Sentence Encoder with COntextualized Vectors for Argument Reasoning Comprehension), 
for [SemEval-2018 Task 12 - The Argument Reasoning Comprehension Task](https://competitions.codalab.org/competitions/17327).

## Results

| Model        | Dev Acc(%) | Test Acc(%)| 
|--------------|:----------:|:----------:|
| Intra-attention [(Harbenal et al., 2018)](https://arxiv.org/abs/1708.01425)            | 63.8 | 55.6 |
| Intra-attention w/context [(Harbenal et al., 2018)](https://arxiv.org/abs/1708.01425)  | 63.7 | 56.0 |
| **SECOARC-last (w/o heruistics)**  | 70.1 | 55.9 |
| **SECOARC-last (w/ heruistics)**  | **70.6** | 55.4 |
| **SECOARC-max (w/o heruistics)**  | 68.0 | 59.1 |
| **SECOARC-max (w/ heuristics)**   | 68.4 | **59.2** |

## Development Environment
- OS: Ubuntu 16.04 LTS (64bit)
- Language: Python 3.6.2
- Pytorch: 0.3.0

## Requirements

Please install the following library requirements specified in the **requirements.txt** first.

    nltk==3.2.4
    tensorboardX==1.0
    torch==0.3.0
    torchtext==0.2.1
    
## Preprocessing

As SECOVARC only accpets one warrant at a time, data manipulation is inevitable.
By executing the **preprocess.py**, you can build a modified version of data (**located in .data/arc/preprocessed**).

> python preprocces.py 

## Training

> python train.py --help

	usage: train.py [-h] [--batch-size BATCH_SIZE] [--dropout DROPOUT]
                [--epoch EPOCH] [--gpu GPU] [--hidden-size HIDDEN_SIZE]
                [--heuristics] [--learning-rate LEARNING_RATE] [--model MODEL]
                [--optim OPTIM] [--print-freq PRINT_FREQ] [--pooling POOLING]
                [--word_dim WORD_DIM]

    optional arguments:
      -h, --help            show this help message and exit
      --batch-size BATCH_SIZE
      --dropout DROPOUT
      --epoch EPOCH
      --gpu GPU
      --hidden-size HIDDEN_SIZE
      --heuristics
      --learning-rate LEARNING_RATE
      --model MODEL         available: bow, lstm, cove
      --optim OPTIM
      --print-freq PRINT_FREQ
      --pooling POOLING     available: max, last, average, min
      --word_dim WORD_DIM
 
