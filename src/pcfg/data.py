from enum import IntEnum

import torch
import numpy as np
import re
import urllib.request
import os
from dataclasses import dataclass, field
from typing import Dict
from random import shuffle


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

def load_PCFG_split(split):
    if split == 'all':
        return load_PCFG_all()
    elif split == 'localism':
        return load_PCFG_localism()
    elif split == 'systematicity':
        return load_PCFG_systematicity()
    elif split == 'substitutivity':
        return load_PCFG_substitutivity()
    elif split == 'productivity':
        return load_PCFG_productivity()
    elif split.startswith('overgeneralization'):
        ratio = float(split.split('-')[-1])
        return load_PCFG_overgeneralization(ratio)


def load_PCFG(src_fp, tgt_fp):
    src_data = load_PCFG_file(src_fp)
    tgt_data = load_PCFG_file(tgt_fp)
    return list(zip(src_data, tgt_data))


def load_PCFG_file(filepath):
    with open(filepath, "rt") as PCFG_f:
        return [line.split() for line in PCFG_f if line]

PCFG_ALL_TRAIN_SRC_URL = "https://github.com/i-machine-think/am-i-compositional/raw/master/data/pcfgset/pcfgset/train.src"
PCFG_ALL_TRAIN_TGT_URL = "https://github.com/i-machine-think/am-i-compositional/raw/master/data/pcfgset/pcfgset/train.tgt"
PCFG_ALL_TEST_SRC_URL = "https://github.com/i-machine-think/am-i-compositional/raw/master/data/pcfgset/pcfgset/dev.src"
PCFG_ALL_TEST_TGT_URL = "https://github.com/i-machine-think/am-i-compositional/raw/master/data/pcfgset/pcfgset/dev.tgt"
PCFG_ALL_TRAIN_SRC_FILEPATH = "./data/PCFG_all_train_src.txt"
PCFG_ALL_TRAIN_TGT_FILEPATH = "./data/PCFG_all_train_tgt.txt"
PCFG_ALL_TEST_SRC_FILEPATH = "./data/PCFG_all_test_src.txt"
PCFG_ALL_TEST_TGT_FILEPATH = "./data/PCFG_all_test_tgt.txt"
def load_PCFG_all():
    if not os.path.exists(PCFG_ALL_TRAIN_SRC_FILEPATH):
        download_file(PCFG_ALL_TRAIN_SRC_URL, PCFG_ALL_TRAIN_SRC_FILEPATH, verbose=True)
    if not os.path.exists(PCFG_ALL_TRAIN_TGT_FILEPATH):
        download_file(PCFG_ALL_TRAIN_TGT_URL, PCFG_ALL_TRAIN_TGT_FILEPATH, verbose=True)
    if not os.path.exists(PCFG_ALL_TEST_SRC_FILEPATH):
        download_file(PCFG_ALL_TEST_SRC_URL, PCFG_ALL_TEST_SRC_FILEPATH, verbose=True)
    if not os.path.exists(PCFG_ALL_TEST_TGT_FILEPATH):
        download_file(PCFG_ALL_TEST_TGT_URL, PCFG_ALL_TEST_TGT_FILEPATH, verbose=True)
    train_data = load_PCFG(PCFG_ALL_TRAIN_SRC_FILEPATH, PCFG_ALL_TRAIN_TGT_FILEPATH)
    test_data = load_PCFG(PCFG_ALL_TEST_SRC_FILEPATH, PCFG_ALL_TEST_TGT_FILEPATH)
    return train_data, test_data

PCFG_LOCALISM_SRC_URL = "https://raw.githubusercontent.com/i-machine-think/am-i-compositional/master/data/pcfgset/localism/increasing_string_length/increasing_length.src"
PCFG_LOCALISM_TGT_URL = "https://raw.githubusercontent.com/i-machine-think/am-i-compositional/master/data/pcfgset/localism/increasing_string_length/increasing_length.tgt"
PCFG_LOCALISM_SRC_FILEPATH = "./data/PCFG_localism_src.txt"
PCFG_LOCALISM_TGT_FILEPATH = "./data/PCFG_localism_tgt.txt"
def load_PCFG_localism(train_pctg=0.75):
    if not os.path.exists(PCFG_LOCALISM_SRC_FILEPATH):
        download_file(PCFG_LOCALISM_SRC_URL, PCFG_LOCALISM_SRC_FILEPATH, verbose=True)
    if not os.path.exists(PCFG_LOCALISM_TGT_FILEPATH):
        download_file(PCFG_LOCALISM_TGT_URL, PCFG_LOCALISM_TGT_FILEPATH, verbose=True)
    data = load_PCFG(PCFG_LOCALISM_SRC_FILEPATH, PCFG_LOCALISM_TGT_FILEPATH)
    shuffle(data)
    train_data = data[:int(len(data)*train_pctg)]
    test_data = data[int(len(data)*train_pctg):]
    return train_data, test_data

PCFG_SYSTEMATICITY_TRAIN_SRC_URL = "https://github.com/i-machine-think/am-i-compositional/raw/master/data/pcfgset/systematicity/train.src"
PCFG_SYSTEMATICITY_TRAIN_TGT_URL = "https://github.com/i-machine-think/am-i-compositional/raw/master/data/pcfgset/systematicity/train.tgt"
PCFG_SYSTEMATICITY_TEST_SRC_URL = "https://github.com/i-machine-think/am-i-compositional/raw/master/data/pcfgset/systematicity/test.src"
PCFG_SYSTEMATICITY_TEST_TGT_URL = "https://github.com/i-machine-think/am-i-compositional/raw/master/data/pcfgset/systematicity/test.tgt"
PCFG_SYSTEMATICITY_TRAIN_SRC_FILEPATH = "./data/PCFG_systematicity_train_src.txt"
PCFG_SYSTEMATICITY_TRAIN_TGT_FILEPATH = "./data/PCFG_systematicity_train_tgt.txt"
PCFG_SYSTEMATICITY_TEST_SRC_FILEPATH = "./data/PCFG_systematicity_test_src.txt"
PCFG_SYSTEMATICITY_TEST_TGT_FILEPATH = "./data/PCFG_systematicity_test_tgt.txt"
def load_PCFG_systematicity():
    if not os.path.exists(PCFG_SYSTEMATICITY_TRAIN_SRC_FILEPATH):
        download_file(PCFG_SYSTEMATICITY_TRAIN_SRC_URL, PCFG_SYSTEMATICITY_TRAIN_SRC_FILEPATH, verbose=True)
    if not os.path.exists(PCFG_SYSTEMATICITY_TRAIN_TGT_FILEPATH):
        download_file(PCFG_SYSTEMATICITY_TRAIN_TGT_URL, PCFG_SYSTEMATICITY_TRAIN_TGT_FILEPATH, verbose=True)
    if not os.path.exists(PCFG_SYSTEMATICITY_TEST_SRC_FILEPATH):
        download_file(PCFG_SYSTEMATICITY_TEST_SRC_URL, PCFG_SYSTEMATICITY_TEST_SRC_FILEPATH, verbose=True)
    if not os.path.exists(PCFG_SYSTEMATICITY_TEST_TGT_FILEPATH):
        download_file(PCFG_SYSTEMATICITY_TEST_TGT_URL, PCFG_SYSTEMATICITY_TEST_TGT_FILEPATH, verbose=True)
    train_data = load_PCFG(PCFG_SYSTEMATICITY_TRAIN_SRC_FILEPATH, PCFG_SYSTEMATICITY_TRAIN_TGT_FILEPATH)
    test_data = load_PCFG(PCFG_SYSTEMATICITY_TEST_SRC_FILEPATH, PCFG_SYSTEMATICITY_TEST_TGT_FILEPATH)
    return train_data, test_data

PCFG_SUBSTITUTIVITY_TRAIN_SRC_URL = "https://github.com/i-machine-think/am-i-compositional/raw/master/data/pcfgset/substitutivity/primitive/train.src"
PCFG_SUBSTITUTIVITY_TRAIN_TGT_URL = "https://github.com/i-machine-think/am-i-compositional/raw/master/data/pcfgset/substitutivity/primitive/train.tgt"
PCFG_SUBSTITUTIVITY_TEST_SRC_URL = "https://github.com/i-machine-think/am-i-compositional/raw/master/data/pcfgset/substitutivity/primitive/test.src"
PCFG_SUBSTITUTIVITY_TEST_TGT_URL = "https://github.com/i-machine-think/am-i-compositional/raw/master/data/pcfgset/substitutivity/primitive/test.tgt"
PCFG_SUBSTITUTIVITY_TRAIN_SRC_FILEPATH = "./data/PCFG_substitutivity_train_src.txt"
PCFG_SUBSTITUTIVITY_TRAIN_TGT_FILEPATH = "./data/PCFG_substitutivity_train_tgt.txt"
PCFG_SUBSTITUTIVITY_TEST_SRC_FILEPATH = "./data/PCFG_substitutivity_test_src.txt"
PCFG_SUBSTITUTIVITY_TEST_TGT_FILEPATH = "./data/PCFG_substitutivity_test_tgt.txt"
def load_PCFG_substitutivity():
    if not os.path.exists(PCFG_SUBSTITUTIVITY_TRAIN_SRC_FILEPATH):
        download_file(PCFG_SUBSTITUTIVITY_TRAIN_SRC_URL, PCFG_SUBSTITUTIVITY_TRAIN_SRC_FILEPATH, verbose=True)
    if not os.path.exists(PCFG_SUBSTITUTIVITY_TRAIN_TGT_FILEPATH):
        download_file(PCFG_SUBSTITUTIVITY_TRAIN_TGT_URL, PCFG_SUBSTITUTIVITY_TRAIN_TGT_FILEPATH, verbose=True)
    if not os.path.exists(PCFG_SUBSTITUTIVITY_TEST_SRC_FILEPATH):
        download_file(PCFG_SUBSTITUTIVITY_TEST_SRC_URL, PCFG_SUBSTITUTIVITY_TEST_SRC_FILEPATH, verbose=True)
    if not os.path.exists(PCFG_SUBSTITUTIVITY_TEST_TGT_FILEPATH):
        download_file(PCFG_SUBSTITUTIVITY_TEST_TGT_URL, PCFG_SUBSTITUTIVITY_TEST_TGT_FILEPATH, verbose=True)
    train_data = load_PCFG(PCFG_SUBSTITUTIVITY_TRAIN_SRC_FILEPATH, PCFG_SUBSTITUTIVITY_TRAIN_TGT_FILEPATH)
    test_data = load_PCFG(PCFG_SUBSTITUTIVITY_TEST_SRC_FILEPATH, PCFG_SUBSTITUTIVITY_TEST_TGT_FILEPATH)
    return train_data, test_data

PCFG_PRODUCTIVITY_TRAIN_SRC_URL = "https://github.com/i-machine-think/am-i-compositional/raw/master/data/pcfgset/productivity/train.src"
PCFG_PRODUCTIVITY_TRAIN_TGT_URL = "https://github.com/i-machine-think/am-i-compositional/raw/master/data/pcfgset/productivity/train.tgt"
PCFG_PRODUCTIVITY_TEST_SRC_URL = "https://github.com/i-machine-think/am-i-compositional/raw/master/data/pcfgset/productivity/test.src"
PCFG_PRODUCTIVITY_TEST_TGT_URL = "https://github.com/i-machine-think/am-i-compositional/raw/master/data/pcfgset/productivity/test.tgt"
PCFG_PRODUCTIVITY_TRAIN_SRC_FILEPATH = "./data/PCFG_productivity_train_src.txt"
PCFG_PRODUCTIVITY_TRAIN_TGT_FILEPATH = "./data/PCFG_productivity_train_tgt.txt"
PCFG_PRODUCTIVITY_TEST_SRC_FILEPATH = "./data/PCFG_productivity_test_src.txt"
PCFG_PRODUCTIVITY_TEST_TGT_FILEPATH = "./data/PCFG_productivity_test_tgt.txt"
def load_PCFG_productivity():
    if not os.path.exists(PCFG_PRODUCTIVITY_TRAIN_SRC_FILEPATH):
        download_file(PCFG_PRODUCTIVITY_TRAIN_SRC_URL, PCFG_PRODUCTIVITY_TRAIN_SRC_FILEPATH, verbose=True)
    if not os.path.exists(PCFG_PRODUCTIVITY_TRAIN_TGT_FILEPATH):
        download_file(PCFG_PRODUCTIVITY_TRAIN_TGT_URL, PCFG_PRODUCTIVITY_TRAIN_TGT_FILEPATH, verbose=True)
    if not os.path.exists(PCFG_PRODUCTIVITY_TEST_SRC_FILEPATH):
        download_file(PCFG_PRODUCTIVITY_TEST_SRC_URL, PCFG_PRODUCTIVITY_TEST_SRC_FILEPATH, verbose=True)
    if not os.path.exists(PCFG_PRODUCTIVITY_TEST_TGT_FILEPATH):
        download_file(PCFG_PRODUCTIVITY_TEST_TGT_URL, PCFG_PRODUCTIVITY_TEST_TGT_FILEPATH, verbose=True)
    train_data = load_PCFG(PCFG_PRODUCTIVITY_TRAIN_SRC_FILEPATH, PCFG_PRODUCTIVITY_TRAIN_TGT_FILEPATH)
    test_data = load_PCFG(PCFG_PRODUCTIVITY_TEST_SRC_FILEPATH, PCFG_PRODUCTIVITY_TEST_TGT_FILEPATH)
    return train_data, test_data

PCFG_OVERGENERALIZATION_TRAIN_SRC_URL = "https://github.com/i-machine-think/am-i-compositional/raw/master/data/pcfgset/overgeneralisation/train_pcfg_ratio={}.src"
PCFG_OVERGENERALIZATION_TRAIN_TGT_URL = "https://github.com/i-machine-think/am-i-compositional/raw/master/data/pcfgset/overgeneralisation/train_pcfg_ratio={}.tgt"
PCFG_OVERGENERALIZATION_TEST_SRC_URL = "https://github.com/i-machine-think/am-i-compositional/raw/master/data/pcfgset/overgeneralisation/test_pcfg_ratio={}_exception.src"
PCFG_OVERGENERALIZATION_TEST_TGT_URL = "https://github.com/i-machine-think/am-i-compositional/raw/master/data/pcfgset/overgeneralisation/test_pcfg_ratio={}_exception.tgt"
PCFG_OVERGENERALIZATION_TRAIN_SRC_FILEPATH = "./data/PCFG_overgeneralization_train_ratio={}_src.txt"
PCFG_OVERGENERALIZATION_TRAIN_TGT_FILEPATH = "./data/PCFG_overgeneralization_train_ratio={}_tgt.txt"
PCFG_OVERGENERALIZATION_TEST_SRC_FILEPATH = "./data/PCFG_overgeneralization_test_ratio={}_exception_src.txt"
PCFG_OVERGENERALIZATION_TEST_TGT_FILEPATH = "./data/PCFG_overgeneralization_test_ratio={}_exception_tgt.txt"
legal_ratios = [0.005, 0.001, 0.0005, 0.0001]
def load_PCFG_overgeneralization(ratio=0.001):
    assert ratio in legal_ratios, "ratio must be one of {}".format(legal_ratios)
    train_src_fp = PCFG_OVERGENERALIZATION_TRAIN_SRC_FILEPATH.format(ratio)
    train_tgt_fp = PCFG_OVERGENERALIZATION_TRAIN_TGT_FILEPATH.format(ratio)
    test_src_fp = PCFG_OVERGENERALIZATION_TEST_SRC_FILEPATH.format(ratio)
    test_tgt_fp = PCFG_OVERGENERALIZATION_TEST_TGT_FILEPATH.format(ratio)
    train_src_url = PCFG_OVERGENERALIZATION_TRAIN_SRC_URL.format(ratio)
    train_tgt_url = PCFG_OVERGENERALIZATION_TRAIN_TGT_URL.format(ratio)
    test_src_url = PCFG_OVERGENERALIZATION_TEST_SRC_URL.format(ratio)
    test_tgt_url = PCFG_OVERGENERALIZATION_TEST_TGT_URL.format(ratio)
    if not os.path.exists(train_src_fp):
        download_file(train_src_url, train_src_fp, verbose=True)
    if not os.path.exists(train_tgt_fp):
        download_file(train_tgt_url, train_tgt_fp, verbose=True)
    if not os.path.exists(test_src_fp):
        download_file(test_src_url, test_src_fp, verbose=True)
    if not os.path.exists(test_tgt_fp):
        download_file(test_tgt_url, test_tgt_fp, verbose=True)
    train_data = load_PCFG(train_src_fp, train_tgt_fp)
    test_data = load_PCFG(test_src_fp, test_tgt_fp)
    return train_data, test_data


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


def preprocess(data, x_vocab, y_vocab):
    return [(x_vocab.encode(x), y_vocab.encode(y + ['<EOS>'])) for x, y in data]


class MyDataLoader:
    def __init__(self, data, batch_size, shuffle=True, x_pad_idx=0, y_pad_idx=0, max_x_seq_len=128, max_y_seq_len=128):
        self.data = data
        self.batch_size = batch_size
        self.num_batches = 0
        self.iterations = 0

        if shuffle:
            np.random.shuffle(self.data)
        self.batches = list(self.make_batches(data, batch_size, x_pad_idx, y_pad_idx, max_x_seq_len, max_y_seq_len))

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


class PrimitivePCFGTypes(IntEnum):
    '''
    TODO: implement
    '''
    pass

pcfg_word_to_type = {}
