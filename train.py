'''
train model
'''

import torch
from torch import optim
import torch.nn as nn
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seq2seq_model as model
import word
import dataset
import torch.utils.data as data
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    '--hidden_size',
    dest='hidden_size',
    action='store',
    required=True,
    help='lstm hidden size')

parser.add_argument(
    '--learning_rate',
    dest='hidden_size',
    action='store',
    required=True,
    help='learning_rate')

parser.add_argument(
    '--batch_size',
    dest='batch_size',
    action='store',
    required=True,
    help='batch size')

parser.add_argument(
    '--num_layers',
    dest='num_layers',
    action='store',
    required=True,
    help='lstm 层数')

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    w = word.Word(
        'data/dgk_shooter.conv',
        'data/dgk_segment.conv',
        'data/dgk_segment.conv',
        'model/dgk_gensim_model'
    )
    q, a = w.QA()
    generate = w.generate_vector(q, a)
    i, q_v, a_v = next(generate)    # 生成数据, i用于判断结束
    ds = dataset.VecDataSet(q_v, a_v)
    train_loader = data.DataLoader(ds, batch_size=64, shuffle=True)
    encoder = model.EncoderRNN(q_v.shape[2], args.hidden_size, q_v.shape[2], args.num_layers, args.batch_size).to(device)
    decoder = model.DecoderRNN(q_v.shape[2], args.hidden_size, q_v.shape[2], args.num_layers, args.batch_size).to(device)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.learning_rate)
    loss_func = nn.CrossEntropyLoss()
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        h0 = encoder.initHidden()
        output, hidden = encoder(x, (h0, h0))
        # 把encoder的输出作为decoder的第一个输入
        y = y[:, :-1, :]    # 去掉y最后一个词，也可以在生成数据时把y少填充一个
        y = torch.cat((output, y), 1)
        h0 = decoder.initHidden()
        output, hidden = decoder(y, (h0, h0))
        loss = loss_func(output, y)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

def test():
    print('test')

if __name__ == '__main__':
    train()

