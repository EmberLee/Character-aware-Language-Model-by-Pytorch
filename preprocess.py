import numpy as np
import chainer
import torch
import torch.nn as nn
import os

os.chdir('C:/Users/ingulbull/Desktop/2019-1/Repro_study_2019_1')
train, val, test = chainer.datasets.get_ptb_words()
# print('train shape: ', train.shape, train)
# print('val   shape: ', val.shape, val)
# print('test  shape: ', test.shape, test)
# train shape:  (929589,) [ 0  1  2 ... 39 26 24]
# val   shape:  (73760,) [2211  396 1129 ...  108   27   24]
# test  shape:  (82430,) [142  78  54 ...  87 214  24]

## Word ID and word correspondence
word_vocab = chainer.datasets.get_ptb_words_vocabulary()
# print('Number of vocabulary', len(ptb_dict))
# print('ptb_dict', ptb_dict)
# Number of vocabulary 10000
# ptb_dict {'aer': 0, 'banknote': 1, 'berlitz': 2, 'calloway': 3,
""" Convert to word sequences """
idx2word = dict((v, k) for k,v in word_vocab.items())
train_list = [idx2word[i] for i in train]
val_list = [idx2word[i] for i in val]
test_list = [idx2word[i] for i in test]
train_seq = ' '.join(train_list).split('<eos>')
val_seq = ' '.join(val_list).split('<eos>')
test_seq = ' '.join(test_list).split('<eos>')

def make_vocab(seqs, char_vocab, max_len, append_token = False):
    """ Return the dictionaries of word and char with indexes. max_len is the max length of word in the data """
    word_count = len(word_vocab)
    char_count = len(char_vocab) + 1
    for seq in seqs:
        words = seq.split()
        for word in words:
            for char in word:
                if char not in char_vocab:
                    char_vocab[char] = char_count
                    char_count += 1

            if append_token == True:
                char_vocab['<pad>'] = 0
                char_vocab['<bow>'] = char_count
                char_vocab['<eow>'] = char_count + 1

            if max_len < len(word):
                max_len = len(word)

    return (char_vocab, max_len)

char_vocab = {}
max_len = 0
char_vocab, max_len = make_vocab(train_seq, char_vocab, max_len)
char_vocab, max_len2 = make_vocab(val_seq, char_vocab, max_len)
char_vocab, max_len3 = make_vocab(test_seq, char_vocab, max_len2, append_token = True)

def word2char_idx(words, char_vocab, max_len, time_steps = 35):
    """ words : a list of words (train_list)
        Return a 3d array of idx of char """
    result = []
    for i, word in enumerate(words):
        char_per_word = [char_vocab[char] for char in word]
        char_per_word = [char_vocab['<bow>']] + char_per_word + [char_vocab['<eow>']]

        for k in range(0, max_len + 2 - len(char_per_word)):
            """Zero padding to compare with (max word length+2) because of bos and eos tokens"""
            char_per_word.append(0)
        result.append(char_per_word)

        if (i+1) == len(words) and (i+1) % time_steps != 0:
            zero_insert = [0 for _ in range(len(char_per_word))]
            for j in range(0, time_steps - (i+1) % time_steps):
                result.append(zero_insert)

    result = np.array(result).reshape(len(result) // time_steps, time_steps, -1)

    return torch.LongTensor(result)

train_char_idx = word2char_idx(train_list, char_vocab, max_len) ## 26560 x (time_steps) x (maxLen)
# val_char_idx = word2char_idx(val_list, char_vocab, max_len2)
# test_char_idx = word2char_idx(test_list, char_vocab, max_len3)
