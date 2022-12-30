import torch
import numpy as np
import re
import urllib.request
import os
from dataclasses import dataclass, field
from typing import Dict


SCAN_LENGTH_TRAIN_URL = "https://raw.githubusercontent.com/brendenlake/SCAN/master/length_split/tasks_train_length.txt"
SCAN_LENGTH_TEST_URL = "https://raw.githubusercontent.com/brendenlake/SCAN/master/length_split/tasks_test_length.txt"
SCAN_LENGTH_TRAIN_FILEPATH = "./data/SCAN_length_train.txt"
SCAN_LENGTH_TEST_FILEPATH = "./data/SCAN_length_test.txt"

SCAN_SIMPLE_TRAIN_URL = "https://raw.githubusercontent.com/brendenlake/SCAN/master/simple_split/tasks_train_simple.txt"
SCAN_SIMPLE_TEST_URL = "https://raw.githubusercontent.com/brendenlake/SCAN/master/simple_split/tasks_test_simple.txt"
SCAN_SIMPLE_TRAIN_FILEPATH = "./data/SCAN_simple_train.txt"
SCAN_SIMPLE_TEST_FILEPATH = "./data/SCAN_simple_test.txt"


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

def load_SCAN(train_fp, test_fp):
    train_data = load_SCAN_file(train_fp)
    test_data = load_SCAN_file(test_fp)
    return train_data, test_data

def load_SCAN_file(filepath):
    with open(filepath, "rt") as SCAN_f:
        data = []
        regex = re.compile("IN: (.*) OUT: (.*)")
        for line in SCAN_f:
            match = regex.match(line)
            if not match:
                raise ValueError(f"Could not parse line: \"{line}\"")
            data.append([group.split() for group in match.groups()])
    return data

def load_SCAN_length():
    if not os.path.exists(SCAN_LENGTH_TRAIN_FILEPATH):
        download_file(SCAN_LENGTH_TRAIN_URL, SCAN_LENGTH_TRAIN_FILEPATH, verbose=True)
    if not os.path.exists(SCAN_LENGTH_TEST_FILEPATH):
        download_file(SCAN_LENGTH_TEST_URL, SCAN_LENGTH_TEST_FILEPATH, verbose=True)
    return load_SCAN(SCAN_LENGTH_TRAIN_FILEPATH, SCAN_LENGTH_TEST_FILEPATH)

def load_SCAN_simple():
    if not os.path.exists(SCAN_SIMPLE_TRAIN_FILEPATH):
        download_file(SCAN_SIMPLE_TRAIN_URL, SCAN_SIMPLE_TRAIN_FILEPATH, verbose=True)
    if not os.path.exists(SCAN_SIMPLE_TEST_FILEPATH):
        download_file(SCAN_SIMPLE_TEST_URL, SCAN_SIMPLE_TEST_FILEPATH, verbose=True)
    return load_SCAN(SCAN_SIMPLE_TRAIN_FILEPATH, SCAN_SIMPLE_TEST_FILEPATH)


@dataclass
class Vocabulary:
    _token_to_idx : Dict[str, int] = field(default_factory=dict)
    _idx_to_token : Dict[str, int] = field(default_factory=dict)
    _token_to_tensor : Dict[str, int] = field(default_factory=dict)

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
            batch = data[i*batch_size : (i+1)*batch_size]
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
        X = torch.concat([torch.nn.functional.one_hot(i[0], num_classes=max_x_vocab).float() for i in self.batches], dim=0)
        Y = torch.concat([torch.nn.functional.one_hot(i[1], num_classes=max_y_vocab).float() for i in self.batches], dim=0)
        return X, Y

class PrimitiveScanTypes(Enum):
    '''
    Type 1: Q (Quantifiers): twice, thrice
    Type 2: A (Actions): walk, look, run, jump
    Type 3: M (Modifiers/Adverbs): opposite, around
    Type 4: D (Directions): left, right
    Type 5: C (Conjunctions): and, after
    Type 6: T (Turn): turn  (I wasn't sure if this can be clubbed in with any of the other types).
    '''

    Q = 1
    A = 2
    M = 3
    D = 4
    C = 5
    T = 6

scan_token_to_type = {
    "twice"     : PrimitiveScanTypes.Q,
    "thrice"    : PrimitiveScanTypes.Q,
    "walk"      : PrimitiveScanTypes.A,
    "look"      : PrimitiveScanTypes.A,
    "run"       : PrimitiveScanTypes.A,
    "jump"      : PrimitiveScanTypes.A,
    "opposite"  : PrimitiveScanTypes.M,
    "around"    : PrimitiveScanTypes.M,
    "left"      : PrimitiveScanTypes.D,
    "right"     : PrimitiveScanTypes.D,
    "and"       : PrimitiveScanTypes.C,
    "after" : PrimitiveScanTypes.C,
    "turn" : PrimitiveScanTypes.T
}