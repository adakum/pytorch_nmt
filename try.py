import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np
import itertools

def flatten(l):
	return list(itertools.chain.from_iterable(l))

seqs = ['ghatmasala','nicela','chutpakodas']

# make <pad> idx 0
vocab = ['<pad>'] + sorted(list(set(flatten(seqs))))
print(vocab)
# make model
embed = nn.Embedding(len(vocab), 10)
lstm = nn.LSTM(10, 5)

a=torch.FloatTensor([1,2,3,4,5])

drop = nn.Dropout(0.4)
print(a)
print(drop(a))
print("again")
print(drop(a))