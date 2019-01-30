import numpy as np
import os; os.chdir('C:/Users/ingulbull/Desktop/2019-1/Repro_study_2019_1')
import torch
import torch.nn as nn
from loader import train_loader, val_loader
from model import src, Highway, CharAwareLM


## Hyper Parameters
learning_rate = 1.0
num_epochs = 25
src.keys()

class Trainer():
    def __init__(self, src, learning_rate, num_epochs, train_loader, val_loader):
        self.char_vocab = src['char_vocab']
        self.word_vocab = src['word_vocab']
        self.max_len = src['maxLen']
        self.time_steps = src['time_steps']
        self.embed_size_char = src['embed_size_char']
        self.hidden_size = src['hidden_size']
        self.num_layer = src['num_layer']
        self.lr = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = src['batch_size']
        self.model = CharAwareLM(src)
        self.model.init_uniform()
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train(self):
        best_ppl = 10000

        if torch.cuda.is_available():
            self.model.cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(filter(
                                    lambda p: p.requires_grad, self.model.parameters()),
                                    lr = self.lr
                                    )

        for epoch in range(self.num_epochs):
            h0_with_c0 = torch.zeros(self.num_layer, self.batch_size, self.hidden_size), torch.zeros(self.num_layer, self.batch_size, self.hidden_size)
            self.model.train(True)

            for i, (data_char, target) in enumerate(self.train_loader):
                self.model.zero_grad()

                h0_with_c0 = [state.detach() for state in h0_with_c0]
                out, h_with_c = self.model(data_char, h0_with_c0)

                loss = criterion(out, target.view(-1))
                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                optimizer.step()

                step = (i+1) // self.time_steps
                if step % 100 == 0:
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.2f' %
                         (epoch+1, self.num_epochs, step, len(self.train_loader) // self.time_steps,
                          loss.item(), np.exp(loss.item())))

            self.model.eval()
            val_loss = self._validate(self.model, h_with_c, criterion)
            val_ppl = np.exp(val_loss)

            if best_ppl - val_ppl < 1:
                if self.lr > 0.03:
                    self.lr = self.lr * 0.5
                    print('Adjusted learning_rate: %.5f' % self.lr)
                    optimizer = torch.optim.SGD(filter(
                                                lambda p: p.requires_grad, self.model.parameters()),
                                                lr = self.lr
                                                )
                else:
                    pass

            if val_ppl < best_ppl:
                print('Current best Val Loss: ', val_loss)
                best_ppl = val_ppl


    def _validate(self, model, hidden, criterion):
        val_loss = 0
        step = 0
        for i, (data_char, target) in enumerate(self.val_loader):
            out_val, _ = model(data_char, hidden)
            loss = criterion(out_val, target)
            val_loss += loss.item()
            step += 1

            model.zero_grad()

        print('Val Loss: %.4f, Perplexity: %5.2f' % (val_loss / step, np.exp(val_loss / step)))

        return val_loss / step
    # def _validate(self, valid_loader):
    #

trainer = Trainer(src, learning_rate, num_epochs, train_loader, val_loader)
trainer.train()
