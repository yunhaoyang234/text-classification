# models.py
import glob
import unicodedata
import string
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nltk.tokenize import sent_tokenize, word_tokenize
#import gensim 
#from gensim.models import Word2Vec
from utils import *

torch.manual_seed(1)

'''
word to vector
'''
def one_hot_encoding(train_examples):
    word_to_ix = {}
    sentences = []
    for exs in train_examples:
        sent = exs.words.split()
        sentences.append(sent)
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix, sentences

def train_vector(sentences, words, min_count = 1, size = 10, window = 5):
    model = gensim.models.Word2Vec(sentences, min_count = min_count, size = size, window = window, sg=1)
    vectors = []
    for w in words:
        vectors.append(model[w])
    return vectors, model

def embed_word(to_ix, sent):
    embed = []
    context = sent.split()
    for c in context:
        if c in to_ix.keys():
            embed.append(to_ix[c])
    inp = torch.tensor(embed, dtype=torch.long).unsqueeze(0)
    return inp

def load_embeddings(path='chinese_words.txt'):
    embedding_map = {}
    to_ix = {}
    idx = 0
    with open(path) as f:
        next(f)
        for line in f:
            try:
                pieces = line.rstrip().split()
                embedding_map[pieces[0]] = [float(weight) for weight in pieces[1:]]
                to_ix[pieces[0]] = idx
                idx += 1
            except:
                pass
    return embedding_map, to_ix

'''
Bi-LSTM
'''
class LSTM(nn.Module):
    def __init__(self, vectors, embedding_dim, hidden_size, vocab_size, tagset_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.word_embeddings = nn.Embedding.from_pretrained(torch.tensor(vectors))
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, bidirectional=True)
        self.hidden2tag = nn.Linear(2 * hidden_size, tagset_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.lstm.weight_hh_l0, a=-1.0 / np.sqrt(self.hidden_size),
                         b=1.0 / np.sqrt(self.hidden_size))
        nn.init.uniform_(self.lstm.weight_ih_l0, a=-1.0 / np.sqrt(self.hidden_size),
                         b=1.0 / np.sqrt(self.hidden_size))
        nn.init.uniform_(self.lstm.bias_hh_l0, a=-1.0 / np.sqrt(self.hidden_size),
                         b=1.0 / np.sqrt(self.hidden_size))
        nn.init.uniform_(self.lstm.bias_ih_l0, a=-1.0 / np.sqrt(self.hidden_size),
                         b=1.0 / np.sqrt(self.hidden_size))

    def forward(self, inpu):
        embedded_input = self.word_embeddings(inpu)
        state = (torch.zeros(self.num_layers * 2, 1, self.hidden_size),
                 torch.zeros(self.num_layers * 2, 1, self.hidden_size))
        output = None
        for inp in embedded_input[0]:
            inp = torch.tensor([[inp.tolist()]])
            output, state = self.lstm.forward(inp, state)
        output = self.hidden2tag(output)
        return output
    
    def predict(self, inpu):
        embedded_input = self.word_embeddings(inpu)
        state = (torch.zeros(self.num_layers * 2, 1, self.hidden_size),
                 torch.zeros(self.num_layers * 2, 1, self.hidden_size))
        output = None
        for inp in embedded_input[0]:
            inp = torch.tensor([[inp.tolist()]])
            output, state = self.lstm.forward(inp, state)
        output = self.softmax(self.hidden2tag(output)[0])
        return np.argmax(output[0].tolist())

'''
training and prediction
'''
def train_model(training_data, vectors, lr=0.1, epochs=30, EMBEDDING_DIM=32, HIDDEN_DIM=32, LAYER_SIZE=3, OUTPUT_SIZE=17, word_to_ix=None):
    if len(vectors) > 0:
        EMBEDDING_DIM = len(vectors[0])
    model = LSTM(vectors, EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), OUTPUT_SIZE, LAYER_SIZE)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for example in training_data:
            model.zero_grad()

            sentence_in = embed_word(word_to_ix, example.words)
            output = model(sentence_in)[0]
            label = torch.tensor([example.label], dtype=torch.long)

            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()
    return model

def predict(model, test_exs, word_to_ix):
    pred = []
    for example in test_exs:
        words = embed_word(word_to_ix, example.words)
        pred.append(model.predict(words))
    return pred

'''
Tuning Parameters through Cross Validation
'''
'''
input
num: number of cross validation, int
param: parameters, dict {}
train_exs: training set, list [Example]
output_size: number of targets, int
metric: categories wants to evaluate on, list of int []
'''
def cross_val(num, param, train_exs, output_size, metric, word_map=None, word_to_ix=None):
    acc = 0
    prec = 0
    reca = 0
    f1 = 0
    length = len(train_exs)
    word_vectors=None
    if word_map == None:
        print("generate vectors")
        word_to_ix, sentences = one_hot_encoding(train_exs)
        word_vectors, w2v_model = train_vector(sentences, word_to_ix.keys(), 
                                               1, param['size'], param['window'])
    else:
        word_vectors=list(word_map.values())
        
    for i in range(num):
        test_set = train_exs[i * length//num: (i+1) * length//num]
        train_set = train_exs[:i * length//num] + train_exs[(i+1) * length//num:]

        model = train_model(train_set, word_vectors, lr=param['lr'], epochs=param['epoch'],
                            HIDDEN_DIM=param['hidden_dim'], LAYER_SIZE=param['layer_size'], 
                            word_to_ix=word_to_ix, OUTPUT_SIZE=output_size)
        pred = predict(model, test_set, word_to_ix)
        test_label = []
        for exs in test_set:
            test_label.append(exs.label)
            
        # evaluation
        acc += calc_acc(pred, test_label)
        p = 0
        r = 0
        f = 0
        print_eval_1_layer(pred, test_label, metric)
        for clas in metric:
            p += precision(pred, test_label, clas)
            r += recall(pred, test_label, clas)
            f += f1_score(pred, test_label, clas)
        prec += p/len(metric)
        reca += r/len(metric)
        f1 += f/len(metric)
        print(calc_acc(pred, test_label),f/len(metric))
    return acc/num, prec/num, reca/num, f1/num

'''
input
cv: number of cross validation, int
train_exs: training set, list [Example]
output_size: number of targets, int
metric: categories wants to evaluate on, list of int []
'''
def tune_param(train_exs, cv=3, output_size=2, metric=[1], word_map=None, word_to_ix=None):
    for size in range(10, 20, 5):
        for window in range(5, 15, 5):
            for lr in [0.1, 0.3, 0.6]:
                for epoch in range(10, 20, 8):
                    for hidden in range(10, 100, 30):
                        for layer in range(1, 3, 2):
                            param = {'size': size, 'window': window, 'epoch': epoch, 
                                     'lr': lr,'hidden_dim': hidden, 'layer_size': layer}
                            print(param)
                            acc = cross_val(cv, param, train_exs, output_size, metric, word_map, word_to_ix)
                            print('accuracy: ', acc[0], 'precision: ', acc[1], 'recall: ', acc[2], 'f1: ', acc[3])
                            print()
                            print()