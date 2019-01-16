import numpy as np
import chainer
import os; os.chdir('C:/Users/ingulbull/Desktop/2019-1/Repro_study_2019_1')
import preprocess as prp
import loader

''' In the prp module :
    train, val, test,
    word2idx, idx2word,
    train_list, val_list, test_list,
    train_seq, val_seq, test_seq,
    word_vocab, char_vocab, max_len,
    function - make_vocab, word2char_idx, make_lookup'''
# prp.train_char_idx

## Sequence to each words and further each characters
## Each seq ends with <eos> token.
import torch
def Seq2CharIdx(seq, char2idx):
    split_seq = seq.split('<eos>')
    idx_of_char_per_sentence = []
    for sentence in split_seq:
        words = sentence.split(' ')
        idx_per_each_sentence = []
        for word in words:
            chars = list(word)
            for char in chars:
                idx_per_each_sentence.append(char2idx.get(char))
        idx_of_char_per_sentence.append(idx_per_each_sentence)

    return idx_of_char_per_sentence
#####


## Module
import torch
import torch.nn as nn
import torch.nn.functional as F

src = {'char_vocab':prp.char_vocab,
       'word_vocab':prp.word_vocab,
       'maxLen':prp.max_len + 2,
       'embed_size_char':15,
       'embed_size_word':300,
       'num_filter_per_width':25,
       'widths':[1,2,3,4,5,6],
       'hidden_size':300,
       'num_layer':2
       }



class Highway(nn.Module):
    def __init__(self, y_k_size):
        super(Highway, self).__init__()
        self.trf_fc = nn.Linear(y_k_size, y_k_size, bias=True)
        self.fc = nn.Linear(y_k_size, y_k_size, bias=True)

    def forward(self, y_k):
        trf_gate = torch.sigmoid(self.trf_fc(y_k)) ## y_k_size x 1
        carry_gate = 1 - trf_gate
        return torch.mul(trf_gate, F.relu(self.fc(y_k))) + torch.mul(carry_gate, y_k)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

class CharAwareLM(nn.Module):
    def __init__(self, src):
        super(CharAwareLM, self).__init__()
        self.char_vocab = src['char_vocab']
        self.word_vocab = src['word_vocab']
        self.char_dim = src['embed_size_char']
        self.word_dim = src['embed_size_word']
        ## embedding layer => Q.shape : (|C|, d)
        self.char_embed = nn.Embedding(len(src['char_vocab']), self.char_dim, padding_idx=0)
        ## num_filter_per_width = 25(multiplied by each filter width), widths = [1,2,3,4,5,6]
        ## filter_width_list = [(25, 1), (50, 2), (75, 3), (100, 4), (125, 5), (150, 6)]
        self.num_filter = src['num_filter_per_width']
        self.filter_width_list = list(zip([self.num_filter * src['widths'][i] for i in range(len(src['widths']))],
                                           src['widths']))

        ## CNN layers packed with nn.ModuleList
        self.sequential_cnn = []
        self.maxpool = []

        for out_channel, filter_size in self.filter_width_list:
            self.sequential_cnn.append(nn.Conv1d(
                                       in_channels=self.char_dim,
                                       out_channels=out_channel,
                                       kernel_size=filter_size)
                                       )
            self.maxpool.append(nn.MaxPool1d(
                                kernel_size=src['maxLen'] - filter_size + 1)
                                )
        self.sequential_cnn = nn.ModuleList(self.sequential_cnn)

        ## Highway layer
        ## y_k_size : sum of number of filters = h
        self.y_k_size = sum([out_channel for out_channel, filter_size in self.filter_width_list])
        self.highway = Highway(self.y_k_size)
        # self.highway2 = Highway(self.y_k_size)

        ## LSTM layer
        self.lstm = nn.LSTM(input_size=self.y_k_size, hidden_size=src['hidden_size'],
                            num_layers=src['num_layer'], batch_first=True, dropout=0.5)

        ## Output softmax layer
        self.output = nn.Sequential(
                            nn.Dropout(0.5),
                            nn.Linear(self.y_k_size, len(src['word_vocab'])),
                            nn.Softmax(dim=-1))

    def forward(self, data_char_idx, h0):
        ''' data_char_idx : (batch) x (max_len+2 = maxLen)
            h0 : '''
        x = self.char_embed(data_char_idx) ## (batch) x (maxLen) x (char_embedding_size)
        x.permute(0, 2, 1) ## (batch) x (char_embedding_size) x (maxLen)

        y_k = [ith_cnn(x) for ith_cnn in self.sequential_cnn] ## (batch) x (num_filter) x (maxLen-width+1)
        for i in range(len(self.maxpool)):
            y_k[i] = self.maxpool[i](y_k[i]) ## (batch) x (num_filter) x 1
        y_k = torch.cat(y_k, 1) ## (batch) x (y_k_size) x 1
        y_k = y_k.permute(0, 2, 1) ## (batch) x 1 x (y_k_size)

        z = self.highway(y_k)

        out, h_c = self.lstm(z, h0)
        out = self.output(out)

        return out

model = CharAwareLM(src)
train2 = prp.word2char_idx(prp.train_list, prp.char_vocab, prp.max_len)
train2.shape
h0 = torch.randn(64, )
model(train2, )

train2
