import numpy as np
# import os; os.chdir('C:/Users/ingulbull/Desktop/2019-1/Repro_study_2019_1')
import preprocess as prp
import loader
from config import src

''' In the prp module :
    train, val, test,
    word2idx, idx2word,
    train_list, val_list, test_list,
    train_seq, val_seq, test_seq,
    word_vocab, char_vocab, max_len,
    function - make_vocab, word2char_idx, make_lookup'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    def __init__(self, y_k_size):
        super(Highway, self).__init__()
        self.trf_fc = nn.Linear(y_k_size, y_k_size, bias=True)
        self.fc = nn.Linear(y_k_size, y_k_size, bias=True)

    def forward(self, y_k):
        trf_gate = torch.sigmoid(self.trf_fc(y_k)) ## y_k_size x 1
        carry_gate = 1 - trf_gate
        return torch.mul(trf_gate, F.relu(self.fc(y_k))) + torch.mul(carry_gate, y_k)

class CharAwareLM(nn.Module):
    def __init__(self, src):
        super(CharAwareLM, self).__init__()
        self.char_vocab = src['char_vocab']
        self.word_vocab = src['word_vocab']
        self.time_steps = src['time_steps']
        self.char_dim = src['embed_size_char']
        self.word_dim = src['embed_size_word']
        self.hidden_size = src['hidden_size']
        ## embedding layer => Q.shape : (|C|, d)
        self.char_embed = nn.Embedding(len(src['char_vocab']), self.char_dim, padding_idx=0)
        ## num_filter_per_width = 25(multiplied by each filter width), widths = [1,2,3,4,5,6]
        ## filter_width_list = [(25, 1), (50, 2), (75, 3), (100, 4), (125, 5), (150, 6)]
        self.num_filter = src['num_filter_per_width']
        self.filter_width_list = list(zip([self.num_filter * src['widths'][i] for i in range(len(src['widths']))],
                                           src['widths']))
        self.num_layer = src['num_layer']
        self.batch_size = src['batch_size']

        ## CNN layers packed with nn.ModuleList
        self.sequential_cnn = []

        for out_channel, filter_size in self.filter_width_list:
            self.sequential_cnn.append(nn.Sequential(
                                           nn.Conv2d(
                                           in_channels=1,
                                           out_channels=out_channel,
                                           kernel_size=(filter_size, self.char_dim)),
                                           nn.Tanh(),
                                           nn.MaxPool2d(
                                           kernel_size=(src['maxLen'] - filter_size + 1, 1)
                                       )))

        self.sequential_cnn = nn.ModuleList(self.sequential_cnn)

        ## Highway layer
        ## y_k_size : sum of number of filters = h
        self.y_k_size = sum([out_channel for out_channel, filter_size in self.filter_width_list])
        self.highway = Highway(self.y_k_size)
        # self.highway2 = Highway(self.y_k_size)

        ## LSTM layer
        self.lstm = nn.LSTM(input_size=self.y_k_size, hidden_size=self.hidden_size,
                            num_layers=src['num_layer'], batch_first=True, dropout=0.5)

        ## Output softmax layer
        self.output = nn.Sequential(
                            nn.Dropout(0.5),
                            nn.Linear(self.hidden_size, len(src['word_vocab'])),
                            )

    def init_weight(self):
        for ith_cnn in self.sequential_cnn:
            ith_cnn[0].weight.data.uniform_(-0.05, 0.05)
        self.lstm.weights_hh_l0.data.uniform_(-0.05, 0.05)
        self.lstm.weights_hh_l1.data.uniform_(-0.05, 0.05)
        self.lstm.weights_ih_l0.data.uniform_(-0.05, 0.05)
        self.lstm.weights_ih_l1.data.uniform_(-0.05, 0.05)
        self.output[1].weight.data.uniform_(-0.05, 0.05)

    def forward(self, data_char_idx, h0_with_c0):
        ''' data_char_idx : (batch) x (time_steps) x (max_len+2 = maxLen)
            h0 : '''
        x = self.char_embed(data_char_idx) ## (batch) x (time_steps) x (maxLen) x (char_embedding_size)
        x = x.view(-1, 1, x.shape[2], x.shape[3]) ## (batch * time_steps) x 1 x (maxLen) x (char_embedding_size)

        y_k = [ith_cnn(x) for ith_cnn in self.sequential_cnn] ## (batch * time_steps) x (num_filter) x 1 x 1
        y_k = torch.cat(y_k, 1) ## (batch * time_steps) x (y_k_size) x 1 x 1
        y_k = y_k.squeeze(3) ## (batch * time_steps) x (y_k_size) x 1
        y_k = y_k.permute(0, 2, 1) ## (batch * time_steps) x 1 x (y_k_size)

        z = self.highway(y_k)
        z = z.view(self.batch_size, self.time_steps, -1)

        out, h_with_c = self.lstm(z, h0_with_c0) ## out : (batch) x (time_steps) x (hidden_size)
                              ## h_with_c : (h_n, c_n) -> (num_layer) x (batch) x (hidden_size)
        out = self.output(out) ## (batch) x (time_steps) x (word_vocab_size)
        out = out.contiguous().view(self.batch_size * self.time_steps, -1)

        return out, h_with_c
