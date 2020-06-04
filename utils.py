# utils.py
import pandas as pd
import glob
import unicodedata
import string
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nltk.tokenize import sent_tokenize, word_tokenize


label = ['配置客户端', '配置客户端(PC)', '账号',
             '网络服务中断', #3
             'GPS本地(调试)',
             'GPS第三方问题', # 5
             '其他软件兼容',
             '汇报Bug(严重)', # 7
             '汇报bug(暂时性)', 'Bug缺少模块', '下载/传文件', '操作问题', '操作问题(gps)',
             '抱怨/退费',
             '要求/改进功能', #14
             '商务', '无响应']

important_label = ['网络服务中断','GPS第三方问题','汇报Bug(严重)','要求/改进功能']
important_label_index = [3,5,7,14]
other_label_index = [0,1,2,4,6,8,9,10,11,12,13,15,16]





class Example:
    def __init__(self, words, label):
        self.words = words
        self.label = label

    def __repr__(self):
        return repr(self.words) + "; label=" + repr(self.label)

    def __str__(self):
        return self.__repr__()

# get the labels of data records
# return list of labels
def get_labels(data):
    labels = []
    for index, record in data.iterrows():
        idx = 0
        for lab in label:
            if record[lab] == 1:
                break
            idx += 1
        idx = 16 if idx > 16 else idx
        labels.append(idx)
    return labels

def get_train_exs(col1, col2, labels):
    train_exs = []
    for i in range(0, len(col1)):
        if not (labels[i] == 6):
            ex = str(col1[i]) + ' ' + str(col2[i])
            train_exs.append(Example(ex, labels[i]))
    return train_exs

# split important and unimportant data records
# return two lists of Examples [Example]
def split_data(col1, col2, labels):
    imp_data = []
    uni_data = []
    for i in range(len(labels)):
        ex = Example(str(col1[i]) + ' ' + str(col2[i]), labels[i])
        if labels[i] in important_label_index:
            imp_data.append(ex)
        else:
            uni_data.append(ex)
    return imp_data, uni_data

'''
first layer- binary classification
second layer- classify important/unimportant labels
by yyh
'''
def get_labels_binary(data):
    labels = get_labels(data)
    label_first = []
    for lab in labels:
        if lab in important_label_index:
            label_first.append(1)
        else:
            label_first.append(0)
    return label_first

def get_labels_important(train_exs, important):
    exs = []
    for tr in train_exs:
        if important:
            ex = Example(tr.words, important_label_index.index(tr.label))
            exs.append(ex)
        else:
            ex = Example(tr.words, other_label_index.index(tr.label))
            exs.append(ex)
    return exs

'''
evaluation
'''
def calc_acc(pred, label):
    correct = 0
    for i in range(0, len(pred)):
        if pred[i] == label[i]:
            correct += 1
    return correct/len(pred)

def precision(pred, actu, clas):
    total = 0
    correct = 0
    for i in range(len(pred)):
        if pred[i] == clas:
            total += 1
            if pred[i] == actu[i]:
                correct += 1
    if total == 0:
        return 1
    return correct/total

def recall(pred, actu, clas):
    total = 0
    correct = 0
    for i in range(len(pred)):
        if actu[i] == clas:
            total += 1
            if pred[i] == actu[i]:
                correct += 1
    if total == 0:
        return 0
    return correct/total

def f1_score(pred, actu, clas):
    prec = precision(pred, actu, clas)
    reca = recall(pred, actu, clas)
    if prec + reca == 0:
        return 0
    return 2 * prec * reca / (prec + reca)

def print_eval_1_layer(pred, test_label, metric):
    for clas in metric:
        print(label[clas])
        print('precision', precision(pred, test_label, clas))
        print('recall', recall(pred, test_label, clas))
        print('f1: ', f1_score(pred, test_label, clas))
        
        
'''
Make Prediction
input: file path, prefer excel file, trained LSTM model, dict of word to index
output: None, write new excel to current working directory
'''
LABEL_START = 14
LABEL_END = 30
def make_prediction(model, file_path, word_to_ix, output_file):
    df, data = read_data(file_path)
    results = {}
    for i in range(17):
        results[i] = [0] * len(data[LABEL_START])
    prediction = predict(model, data, word_to_ix)
    idx = 0
    for pred in prediction:
        results[pred][idx] = 1
        idx += 1
    
    for i in range(LABEL_START, LABEL_END+1):
        data[i] = results[i]
    columns = range(0, 33)
    for c in range(17):
        columns[LABEL_START+c] = label[c]
    data.columns = columns
    data.to_excel(output_file, index=False)

def read_data(file_path):
    df = pd.read_excel(file_path, header=None)
    data = get_train_exs(df[31], df[32], [0]*len(df[31]))
    return df, data