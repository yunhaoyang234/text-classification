import pandas as pd
import os
from io import open
import glob
import unicodedata
import string
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nltk.tokenize import sent_tokenize, word_tokenize

from utils import *
from models import *

print('read data')
data = pd.read_excel('2018 payroll.xlsx', header=None)
data2 = pd.read_excel('2018_tagged_data.xlsx')
data.columns = data2.columns

col1 = data['Unnamed: 31'][:5000]
col2 = data['Unnamed: 32'][:5000]

labels = get_labels(data[:][:5000])

train_exs = get_train_exs(col1, col2, labels)

print('load embedding')
word_map, to_ix = load_embeddings()

print('train model')
param = {'size': 10, 'window': 5, 'lr': 0.1,'epoch': 12, 'hidden_dim': 25, 'layer_size': 1}
cross_val(3, param, train_exs, 17, metric=[3,5,7,14], word_map=word_map, word_to_ix=to_ix)

word_vectors=list(word_map.values())
model = train_model(train_exs, word_vectors, lr=param['lr'], epochs=param['epoch'],
                    HIDDEN_DIM=param['hidden_dim'], LAYER_SIZE=param['layer_size'], 
                    word_to_ix=to_ix)
print('predict')
make_prediction(model, '2018 payroll.xlsx', to_ix, '2018 result.xlsx')