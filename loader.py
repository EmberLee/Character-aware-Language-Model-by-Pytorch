import torch
import torch.nn as nn
from torch.utils import data
# import os; os.chdir('C:/Users/ingulbull/Desktop/2019-1/Repro_study_2019_1')
import preprocess as prp

''' Things to sum up :
    batch_size for data-S : 20,
    init_lr : 10 => halved if the perplexity does not decrease by more than 1.0 on the valid set after an epoch.
    target :  '''
class OwnDataset(data.Dataset):
    def __init__(self, data_char_idx, target):
        self.data_char_idx = data_char_idx
        self.target = torch.LongTensor(target)

    def __getitem__(self, index):
        return torch.tensor(self.data_char_idx[index]), torch.tensor(self.target[index])

    def __len__(self):
        return len(self.data_char_idx)

batch = 20

trainset = OwnDataset(prp.train_char_idx, prp.target_train)
train_loader = data.DataLoader(dataset=trainset,
                               batch_size=batch,
                               shuffle=False,
                               drop_last=True)

valset = OwnDataset(prp.val_char_idx, prp.target_val)
val_loader = data.DataLoader(dataset=valset,
                               batch_size=batch,
                               shuffle=False,
                               drop_last=True)

testset = OwnDataset(prp.test_char_idx, prp.target_test)
test_loader = data.DataLoader(dataset=testset,
                               batch_size=batch,
                               shuffle=False,
                               drop_last=True)
