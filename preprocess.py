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
word2idx = chainer.datasets.get_ptb_words_vocabulary()
# print('Number of vocabulary', len(ptb_dict))
# print('ptb_dict', ptb_dict)
# Number of vocabulary 10000
# ptb_dict {'aer': 0, 'banknote': 1, 'berlitz': 2, 'calloway': 3,
""" Convert to word sequences """
idx2word = dict((v, k) for k,v in word2idx.items())
train_list = [idx2word[i] for i in train]
val_list = [idx2word[i] for i in val]
test_list = [idx2word[i] for i in test]
train_seq = ' '.join(train_list).split('<eos>')
val_seq = ' '.join(val_list).split('<eos>')
test_seq = ' '.join(test_list).split('<eos>')

def make_vocab(seqs, word_vocab, char_vocab, max_len, append_token = False):
    """ Return the dictionaries of word and char with indexes. max_len is the max length of word in the data """
    word_count = len(word_vocab)
    char_count = len(char_vocab) + 1
    for seq in seqs:
        words = seq.split()
        for word in words:
            if word not in word_vocab:
                word_vocab[word] = word_count
                word_count += 1

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

    return (word_vocab, char_vocab, max_len)

word_vocab = {}
char_vocab = {}
max_len = 0
word_vocab, char_vocab, max_len = make_vocab(train_seq, word_vocab, char_vocab, max_len)
word_vocab, char_vocab, max_len2 = make_vocab(val_seq, word_vocab, char_vocab, max_len)
word_vocab, char_vocab, max_len3 = make_vocab(test_seq, word_vocab, char_vocab, max_len2, append_token = True)

def word2char_idx(words, char_vocab, max_len):
    """ words : a list of words (train_list)
        Return a 2d array of idx of char """
    result = []
    for word in words:
        result_per_word = [char_vocab.get(char) for char in word]
        result_per_word = [char_vocab.get('<bow>')] + result_per_word + [char_vocab.get('<eow>')]
        if len(result_per_word) < max_len + 2:
            """Zero padding to compare with (max word length+2) because of bos and eos tokens"""
            result_per_word += [0 for _ in range(max_len + 2 - len(result_per_word))]
        result.append(result_per_word)
    return torch.LongTensor(result)

train_char_idx = word2char_idx(train_list, char_vocab, max_len)
val_char_idx = word2char_idx(val_list, char_vocab, max_len2)
test_char_idx = word2char_idx(test_list, char_vocab, max_len3)
