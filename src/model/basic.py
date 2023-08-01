"""Basic or helper implementation."""

import os
import urllib
from dataclasses import dataclass, field
from typing import Dict
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_


class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            self.layers.append(nn.Linear(in_d, out_d))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = F.relu(x)
        return x


def int_to_one_hot(x, size):
    result = torch.tensor([0 for _ in range(size)])
    result[x] = 1
    return result


def download_file(url, filepath, verbose=False):
    if verbose:
        print(f"Downloading \"{url}\" to \"{filepath}\"...")

    # create directory structure if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # create file to download to if it doesn't exist
    try:
        open(filepath, 'a').close()
    except OSError:
        print(f"Cannot download \"{url}\" to \"{filepath}\" because the target file cannot be opened or created")
    urllib.request.urlretrieve(url, filepath)

@dataclass
class Vocabulary:
    _token_to_idx: Dict[str, int] = field(default_factory=dict)
    _idx_to_token: Dict[str, int] = field(default_factory=dict)
    _token_to_tensor: Dict[str, int] = field(default_factory=dict)

    def add_token(self, token):
        if token not in self._token_to_idx:
            idx = len(self._token_to_idx)
            self._token_to_idx[token] = idx
            self._idx_to_token[idx] = token
            self._token_to_tensor[token] = torch.tensor(idx)

    def contains_token(self, token):
        return token in self._token_to_idx

    def add_tokens(self, tokens):
        for token in tokens:
            self.add_token(token)

    def __len__(self):
        return self.size()

    def size(self):
        return len(self._token_to_idx)

    def token_to_idx(self, token):
        return self._token_to_idx.get(token, self._token_to_idx['<UNK>'])

    def idx_to_token(self, idx):
        return self._idx_to_token[idx]

    def token_to_tensor(self, token):
        return self._token_to_tensor.get(token, self._token_to_tensor['<UNK>'])

    def token_to_ohe(self, token):
        return torch.nn.functional.one_hot(self._token_to_tensor[token], len(self)).long()

    def encode(self, x):
        return [self.token_to_idx(token) for token in x]

    def decode(self, x):
        return [self.idx_to_token(idx) for idx in x]

    def decode_batch(self, X: np.ndarray, mask: np.ndarray = None):
        if mask is None:
            mask = np.ones_like(X)
        return [self.decode(x[:m]) for x, m in zip(X, mask.sum(axis=1))]


def build_vocab(data, base_tokens=[]):
    vocab = Vocabulary()
    vocab.add_tokens(base_tokens)

    for sequence in data:
        for token in sequence:
            vocab.add_token(token)

    return vocab


class MyDataLoader:
    def __init__(self, data, batch_size, shuffle=True, sort_by_len=False, x_pad_idx=0, y_pad_idx=0,
                 max_x_seq_len=128, max_y_seq_len=128, max_sentence_len=torch.inf):
        self.data = data
        self.batch_size = batch_size
        self.num_batches = 0
        self.iterations = 0
        self.max_sentence_len = max_sentence_len

        if shuffle:
            np.random.shuffle(self.data)
        if sort_by_len:
            self.data.sort(key=lambda x: len(x[0]))
        if self.max_sentence_len != torch.inf:
            self.data = list(filter(lambda x: len(x[0]) <= self.max_sentence_len and len(x[1]) <= self.max_sentence_len,
                               self.data))
        self.batches = list(self.make_batches(self.data, batch_size, x_pad_idx, y_pad_idx, max_x_seq_len, max_y_seq_len))

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def make_batches(self, data, batch_size, x_pad_idx, y_pad_idx, max_x_seq_len, max_y_seq_len, drop_last=False):
        num_batches = int(len(data) / batch_size) + int(~drop_last and len(data) % batch_size != 0)
        self.num_batches = num_batches
        for i in range(num_batches):
            batch = data[i * batch_size: (i + 1) * batch_size]
            X, Y = list(zip(*batch))
            X_tensor = self.make_batch(X, max_x_seq_len, x_pad_idx)
            Y_tensor = self.make_batch(Y, max_y_seq_len, y_pad_idx)
            yield X_tensor, Y_tensor

    def make_batch(self, data, max_seq_len, pad_idx):
        seq_len = max(max_seq_len, max(len(x) for x in data))
        batch = torch.full((len(data), seq_len), pad_idx, dtype=torch.long)
        for i, x in enumerate(data):
            batch[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        return batch

    def get_full_data(self, max_x_vocab, max_y_vocab):
        X = torch.concat([torch.nn.functional.one_hot(i[0], num_classes=max_x_vocab).float() for i in self.batches],
                         dim=0)
        Y = torch.concat([torch.nn.functional.one_hot(i[1], num_classes=max_y_vocab).float() for i in self.batches],
                         dim=0)
        return X, Y
    
def run_iter(model, batch, params=None, optimizer=None, is_training=False, verbose=False, print_error=False):
    batch_x, batch_y = batch
    B = batch_x.size(0)
    x_vocab = model.dataset.x_vocab
    y_vocab = model.dataset.y_vocab

    x_tokens = x_vocab.decode_batch(batch_x.numpy(), batch_x != x_vocab.token_to_idx('<PAD>'))
    for i in range(len(x_tokens)):
        x_tokens[i] = " ".join(x_tokens[i])
    
    positions = []
    types = []
    spans = []
    for i in range(B):
        p, t, s = model.dataset.parse(x_tokens[i])
        positions.append(p)
        types.append(t)
        spans.append(s)
    
    decoding = model(batch_x, positions, types, spans)
    decoding_idx = decoding.argmax(-1)
    
    N = decoding.size(1)
    M = batch_y.size(1)
    if N >= M:
        # pad expected output
        expected_padded = torch.full((B, N), float(model.dataset.y_vocab.token_to_idx('<PAD>')))
        expected_padded[:, :M] = batch_y
        outputs_padded = decoding_idx
    else:
        # pad decoded output
        expected_padded = batch_y
        outputs_padded = torch.full((B, M), float(y_vocab.token_to_idx('<PAD>')))
        outputs_padded[:, :N] = decoding_idx

    match = torch.eq(expected_padded.float(), outputs_padded.float())
    match = [(match[i].sum() == match.size(1)).float() for i in range(match.size(0))]
    loss = F.cross_entropy(torch.clamp(torch.flatten(decoding, start_dim=0, end_dim=1), min=1.0e-10, max=1.0),
                            expected_padded.flatten().long())
    accuracy = torch.tensor(match).mean()

    if verbose or (print_error and accuracy != 1.):
        input = x_vocab.decode_batch(batch_x.numpy(), batch_x != x_vocab.token_to_idx('<PAD>'))
        expected = y_vocab.decode_batch(batch_y.numpy(), batch_y != y_vocab.token_to_idx('<PAD>'))
        decoded = y_vocab.decode_batch(decoding_idx.numpy(), decoding_idx != y_vocab.token_to_idx('<PAD>'))
        print("--------------------------------")
        for i in range(B):
            print("input: ", " ".join(input[i]), "\n")
            print("expected: ", " ".join(expected[i]), "\n")
            print("decoded: ", " ".join(decoded[i]))
            print("--------------------------------")

    if is_training:
        # zero gradients
        for param in model.parameters():
            param.grad = None
        loss.backward()
        clip_grad_norm_(parameters=params, max_norm=5)
        optimizer.step()
    return loss, accuracy


def quote(l):
    return map(lambda x: "\"" + x + "\"", l)