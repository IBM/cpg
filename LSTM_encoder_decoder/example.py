# Author: Laura Kulowski

'''

Example of using a LSTM encoder-decoder to model a synthetic time series 

'''
import argparse
import logging

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from importlib import reload
import sys

import torch

import generate_dataset
import lstm_encoder_decoder
import plotting
import data
from data import MyDataLoader


matplotlib.rcParams.update({'font.size': 17})

# # ----------------------------------------------------------------------------------------------------------------
# # generate dataset for LSTM
# t, y = generate_dataset.synthetic_data()
# t_train, y_train, t_test, y_test = generate_dataset.train_test_split(t, y, split=0.8)
#
# # plot time series
# plt.figure(figsize=(18, 6))
# plt.plot(t, y, color='k', linewidth=2)
# plt.xlim([t[0], t[-1]])
# plt.xlabel('$t$')
# plt.ylabel('$y$')
# plt.title('Synthetic Time Series')
# plt.savefig('plots/synthetic_time_series.png')
#
# # plot time series with train/test split
# plt.figure(figsize=(18, 6))
# plt.plot(t_train, y_train, color='0.4', linewidth=2, label='Train')
# plt.plot(np.concatenate([[t_train[-1]], t_test]), np.concatenate([[y_train[-1]], y_test]),
#          color=(0.74, 0.37, 0.22), linewidth=2, label='Test')
# plt.xlim([t[0], t[-1]])
# plt.xlabel(r'$t$')
# plt.ylabel(r'$y$')
# plt.title('Time Series Split into Train and Test Sets')
# plt.legend(bbox_to_anchor=(1, 1))
# plt.tight_layout
# plt.savefig('plots/train_test_split.png')
#
# # ----------------------------------------------------------------------------------------------------------------
# # window dataset
#
# # set size of input/output windows
# iw = 80
# ow = 20
# s = 5
#
# # generate windowed training/test datasets
# Xtrain, Ytrain = generate_dataset.windowed_dataset(y_train, input_window=iw, output_window=ow, stride=s)
# Xtest, Ytest = generate_dataset.windowed_dataset(y_test, input_window=iw, output_window=ow, stride=s)
#
# # plot example of windowed data
# plt.figure(figsize=(10, 6))
# plt.plot(np.arange(0, iw), Xtrain[:, 0, 0], 'k', linewidth=2.2, label='Input')
# plt.plot(np.arange(iw - 1, iw + ow), np.concatenate([[Xtrain[-1, 0, 0]], Ytrain[:, 0, 0]]),
#          color=(0.2, 0.42, 0.72), linewidth=2.2, label='Target')
# plt.xlim([0, iw + ow - 1])
# plt.xlabel(r'$t$')
# plt.ylabel(r'$y$')
# plt.title('Example of Windowed Training Data')
# plt.legend(bbox_to_anchor=(1.3, 1))
# plt.tight_layout()
# plt.savefig('plots/windowed_data.png')

# ----------------------------------------------------------------------------------------------------------------
# LSTM encoder-decoder

# convert windowed data from np.array to PyTorch tensor
#X_train, Y_train, X_test, Y_test = generate_dataset.numpy_to_torch(Xtrain, Ytrain, Xtest, Ytest)


train_data, test_data = data.load_SCAN_length()
training_size = int(len(train_data) * 1.0)
train_data = train_data[:training_size]
print(f"Train data set size: {len(train_data)}")
print(f"Train data sample:\n\tx: {train_data[0][0]}\n\ty: {train_data[0][1]}")
print(f"Test data set size: {len(test_data)}")
print(f"Test data sample:\n\tx: {test_data[0][0]}\n\ty: {test_data[0][1]}")
x_vocab = data.build_vocab([x for x, _ in train_data],
                           base_tokens=['<PAD>', '<UNK>'])
y_vocab = data.build_vocab([y for _, y in train_data],
                           base_tokens=['<PAD>', '<SOS>', '<EOS>', '<UNK>'])
max_x_seq_len = max(len(x) for x, _ in train_data + test_data)
max_y_seq_len = max(len(y) for _, y in train_data + test_data) + 1  # +1 for <EOS>
print(f"X Vocab size: {len(x_vocab)}")
print(f"Y Vocab size: {len(y_vocab)}")
print(f"X Max length: {max_x_seq_len}")
print(f"Y Max length: {max_y_seq_len}")

preprocessed_train_data = data.preprocess(train_data, x_vocab, y_vocab)
preprocessed_test_data = data.preprocess(test_data, x_vocab, y_vocab)
train_loader = MyDataLoader(preprocessed_train_data,
                            batch_size=100,
                            shuffle=False,
                            x_pad_idx=x_vocab.token_to_idx('<PAD>'),
                            y_pad_idx=y_vocab.token_to_idx('<PAD>'),
                            max_x_seq_len=max_x_seq_len,
                            max_y_seq_len=max_y_seq_len)

preprocessed_valid_data = data.preprocess(test_data, x_vocab, y_vocab)  # from scan length test_data
valid_loader = MyDataLoader(preprocessed_valid_data,
                            batch_size=100,
                            shuffle=False,
                            x_pad_idx=x_vocab.token_to_idx('<PAD>'),
                            y_pad_idx=y_vocab.token_to_idx('<PAD>'),
                            max_x_seq_len=max_x_seq_len,
                            max_y_seq_len=max_y_seq_len)
# specify model parameters and train
X_train, Y_train = train_loader.get_full_data(len(x_vocab), len(y_vocab))
X_test, Y_test = valid_loader.get_full_data(len(x_vocab), len(y_vocab))

model = lstm_encoder_decoder.lstm_seq2seq(encoder_input_size=X_train.shape[2],
                                          decoder_input_size=Y_train.shape[2],
                                          hidden_size=32,
                                          y_vocab=y_vocab)
loss = model.train_model(X_train.transpose(0, 1),
                         Y_train.transpose(0, 1),
                         n_epochs=50,
                         target_len=max_y_seq_len,
                         batch_size=32,
                         training_prediction='mixed_teacher_forcing',
                         teacher_forcing_ratio=0.6,
                         learning_rate=0.01,
                         dynamic_tf=False)

# plot predictions on train/test data
plotting.plot_train_test_results(model, X_train, Y_train, X_test, Y_test)

plt.close('all')