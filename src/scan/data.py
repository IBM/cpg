from enum import IntEnum

import torch
import numpy as np
import re
import urllib.request
import os
from dataclasses import dataclass, field
from typing import Dict
from lark import Lark

SCAN_LENGTH_TRAIN_URL = "https://raw.githubusercontent.com/brendenlake/SCAN/master/length_split/tasks_train_length.txt"
SCAN_LENGTH_TEST_URL = "https://raw.githubusercontent.com/brendenlake/SCAN/master/length_split/tasks_test_length.txt"
SCAN_LENGTH_TRAIN_FILEPATH = "./data/SCAN_length_train.txt"
SCAN_LENGTH_TEST_FILEPATH = "./data/SCAN_length_test.txt"

SCAN_SIMPLE_TRAIN_URL = "https://raw.githubusercontent.com/brendenlake/SCAN/master/simple_split/tasks_train_simple.txt"
SCAN_SIMPLE_TEST_URL = "https://raw.githubusercontent.com/brendenlake/SCAN/master/simple_split/tasks_test_simple.txt"
SCAN_SIMPLE_TRAIN_FILEPATH = "./data/SCAN_simple_train.txt"
SCAN_SIMPLE_TEST_FILEPATH = "./data/SCAN_simple_test.txt"

SCAN_ADD_JUMP_0_TRAIN_URL = "https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/tasks_train_addprim_jump.txt"
SCAN_ADD_JUMP_0_TEST_URL = "https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/tasks_test_addprim_jump.txt"
SCAN_ADD_JUMP_0_TRAIN_FILEPATH = "./data/SCAN_add_jump_0_train.txt"
SCAN_ADD_JUMP_0_TEST_FILEPATH = "./data/SCAN_add_jump_0_test.txt"

SCAN_ADD_JUMP_4_TRAIN_URL = "https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/with_additional_examples/tasks_train_addprim_complex_jump_num4_rep1.txt"
SCAN_ADD_JUMP_4_TEST_URL = "https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/with_additional_examples/tasks_test_addprim_complex_jump_num4_rep1.txt"
SCAN_ADD_JUMP_4_TRAIN_FILEPATH = "./data/SCAN_add_jump_4_train.txt"
SCAN_ADD_JUMP_4_TEST_FILEPATH = "./data/SCAN_add_jump_4_test.txt"

TK_SIMPLE_TRAIN_FILEPATH = "./data/simple_train.txt"
TK_SIMPLE_TEST_FILEPATH = "./data/simple_test.txt"

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

def load_TK_simple():
    return load_SCAN(TK_SIMPLE_TRAIN_FILEPATH, TK_SIMPLE_TEST_FILEPATH)


def load_SCAN_simple():
    if not os.path.exists(SCAN_SIMPLE_TRAIN_FILEPATH):
        download_file(SCAN_SIMPLE_TRAIN_URL, SCAN_SIMPLE_TRAIN_FILEPATH, verbose=True)
    if not os.path.exists(SCAN_SIMPLE_TEST_FILEPATH):
        download_file(SCAN_SIMPLE_TEST_URL, SCAN_SIMPLE_TEST_FILEPATH, verbose=True)
    return load_SCAN(SCAN_SIMPLE_TRAIN_FILEPATH, SCAN_SIMPLE_TEST_FILEPATH)

def load_SCAN_add_jump_0():
    if not os.path.exists(SCAN_ADD_JUMP_0_TRAIN_FILEPATH):
        download_file(SCAN_ADD_JUMP_0_TRAIN_URL, SCAN_ADD_JUMP_0_TRAIN_FILEPATH, verbose=True)
    if not os.path.exists(SCAN_ADD_JUMP_0_TEST_FILEPATH):
        download_file(SCAN_ADD_JUMP_0_TEST_URL, SCAN_ADD_JUMP_0_TEST_FILEPATH, verbose=True)
    return load_SCAN(SCAN_ADD_JUMP_0_TRAIN_FILEPATH, SCAN_ADD_JUMP_0_TEST_FILEPATH)

def load_SCAN_add_jump_4():
    if not os.path.exists(SCAN_ADD_JUMP_4_TRAIN_FILEPATH):
        download_file(SCAN_ADD_JUMP_4_TRAIN_URL, SCAN_ADD_JUMP_4_TRAIN_FILEPATH, verbose=True)
    if not os.path.exists(SCAN_ADD_JUMP_4_TEST_FILEPATH):
        download_file(SCAN_ADD_JUMP_4_TEST_URL, SCAN_ADD_JUMP_4_TEST_FILEPATH, verbose=True)
    return load_SCAN(SCAN_ADD_JUMP_4_TRAIN_FILEPATH, SCAN_ADD_JUMP_4_TEST_FILEPATH)


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


def preprocess(data, x_vocab, y_vocab):
    return [(x_vocab.encode(x), y_vocab.encode(y + ['<EOS>'])) for x, y in data]


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


class PrimitiveScanTypes(IntEnum):
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


scan_word_to_type = {
    "twice": PrimitiveScanTypes.Q,
    "thrice": PrimitiveScanTypes.Q,
    "walk": PrimitiveScanTypes.A,
    "look": PrimitiveScanTypes.A,
    "run": PrimitiveScanTypes.A,
    "jump": PrimitiveScanTypes.A,
    "opposite": PrimitiveScanTypes.M,
    "around": PrimitiveScanTypes.M,
    "left": PrimitiveScanTypes.D,
    "right": PrimitiveScanTypes.D,
    "and": PrimitiveScanTypes.C,
    "after": PrimitiveScanTypes.C,
    "turn": PrimitiveScanTypes.T
}

class ScanTypes(IntEnum):
    A = 0
    T = 1
    M = 2
    N = 3
    D = 4
    P = 5
    Q = 6
    I = 7
    J = 8
    C1 = 9
    C2 = 10
    F = 11
    G = 12
    S1 = 13
    S2 = 14
    V1 = 15
    V2 = 16
    V3 = 17
    V4 = 18
    V5 = 19
    V6 = 20
    E1 = 21
    E2 = 22
    E3 = 23
    E4 = 24


scan_token_to_type = {
    "twice": ScanTypes.P,
    "thrice": ScanTypes.Q,
    "walk": ScanTypes.A,
    "look": ScanTypes.A,
    "run": ScanTypes.A,
    "jump": ScanTypes.A,
    "opposite": ScanTypes.M,
    "around": ScanTypes.N,
    "left": ScanTypes.D,
    "right": ScanTypes.D,
    "and": ScanTypes.I,
    "after": ScanTypes.J,
    "turn": ScanTypes.T,
    "c1": ScanTypes.C1,
    "c2": ScanTypes.C2,
    "f": ScanTypes.F,
    "g": ScanTypes.G,
    "s1": ScanTypes.S1,
    "s2": ScanTypes.S2,
    "v1": ScanTypes.V1,
    "v2": ScanTypes.V2,
    "v3": ScanTypes.V3,
    "v4": ScanTypes.V4,
    "v5": ScanTypes.V5,
    "v6": ScanTypes.V6,
    "e1": ScanTypes.E1,
    "e2": ScanTypes.E2,
    "e3": ScanTypes.E3,
    "e4": ScanTypes.E4
}

scan_grammar = """
    start: c1 | c2 | s1 | s2 | v1 | v2 | v3 | v4 | v5 | v6 | a
    c1: f s1 | f s2 | f a | f v1 | f v2 | f v3 | f v4 | f v5 | f v6
    c2: g s1 | g s2 | g a | g v1 | g v2 | g v3 | g v4 | g v5 | g v6
    f: s1 i | s2 i | a i | v1 i | v2 i | v3 i | v4 i | v5 i | v6 i
    g: s1 j | s2 j | a j | v1 j | v2 j | v3 j | v4 j | v5 j | v6 j
    s1: v1 p | v2 p | v3 p | v4 p | v5 p | v6 p | a p
    s2: v1 q | v2 q | v3 q | v4 q | v5 q | v6 q | a q
    v1: a d
    v2: t d
    v3: e1 d
    v4: e2 d
    v5: e3 d
    v6: e4 d
    e1: a m
    e2: a n
    e3: t m
    e4: t n
    a: WALK | LOOK | RUN | JUMP
    t: TURN
    m: OPPOSITE
    n: AROUND
    d: LEFT | RIGHT
    p: TWICE
    q: THRICE
    i: AND
    j: AFTER

    AND: "and"
    AFTER: "after"
    TWICE: "twice"
    THRICE: "thrice"
    WALK: "walk"
    LOOK: "look"
    RUN: "run"
    JUMP: "jump"
    TURN: "turn"
    OPPOSITE: "opposite"
    AROUND: "around"
    LEFT: "left"
    RIGHT: "right"

    %import common.LETTER
    %import common.WS
    %ignore WS
"""

parser = Lark(scan_grammar, propagate_positions=True)

def parse_scan(scan_command):
    parse_tree = parser.parse(scan_command)
    current_position = 0
    previous_index = 0
    positions = []
    types = []
    for node in parse_tree.iter_subtrees_topdown():
        if previous_index < node.meta.start_pos:
            current_position += 1
            previous_index = node.meta.start_pos
        if node.data.value in ["a", "t", "m", "n", "d", "p", "q", "i", "j", "start"]:
            continue
        positions.append(current_position)
        types.append(scan_token_to_type[node.data.value])
    return positions, types


def get_decoding_force(dl, dr, target_types, positions):
    B, L, M, V = dl.size()
    dl_sample = dl.argmax(-1)
    dr_sample = dr.argmax(-1)
    length_l = torch.zeros(B, L).int()
    length_r = torch.zeros(B, L).int()
    for i in range(B):
        for k in range(L):
            length_l[i][k] = torch.count_nonzero(dl_sample[i][k])
            length_r[i][k] = torch.count_nonzero(dr_sample[i][k])
    templates = torch.zeros(B, L, M)
    for i in range(B):
        k = positions[i]
        l = length_l[i][k]
        r = length_r[i][k]
        # To deal with paddings
        if target_types[i] == 0:
            templates[i][k][:l] = dl_sample[i][k][:l]
        # (x_1 and) (x_2): x_1 x_2
        if target_types[i] == 9:
            templates[i][k][:l - 1] = dl_sample[i][k][:l - 1]
            templates[i][k][l - 1:l + r - 1] = dr_sample[i][k][:r]
        # (x_1 after) (x_2): x_2 x_1
        if target_types[i] == 10:
            templates[i][k][:r] = dr_sample[i][k][:r]
            templates[i][k][r:r + l - 1] = dl_sample[i][k][:l - 1]
        # (x_1) (and): x_1 and
        # (x_1) (after): x_1 after
        if target_types[i] in [11, 12]:
            templates[i][k][:l] = dl_sample[i][k][:l]
            templates[i][k][l] = dr_sample[i][k][0]
        # (x_1) (twice): x_1 x_1
        if target_types[i] == 13:
            templates[i][k][:l] = dl_sample[i][k][:l]
            templates[i][k][l:l * 2] = dl_sample[i][k][:l]
        # (x_1) (thrice): x_1 x_1 x_1s
        if target_types[i] == 14:
            templates[i][k][:l] = dl_sample[i][k][:l]
            templates[i][k][l:l * 2] = dl_sample[i][k][:l]
            templates[i][k][l * 2:l * 3] = dl_sample[i][k][:l]
        # (x_1) (x_2): x_2 x_1
        if target_types[i] == 15:
            templates[i][k][:r] = dr_sample[i][k][:r]
            templates[i][k][r:r + l] = dl_sample[i][k][:l]
        # (turn) (x_1): x_1
        if target_types[i] == 16:
            templates[i][k][:r] = dr_sample[i][k][:r]
        # (x_1 opposite) (x_2): x_2 x_2 x_1
        if target_types[i] == 17:
            templates[i][k][:r] = dr_sample[i][k][:r]
            templates[i][k][r:r * 2] = dr_sample[i][k][:r]
            templates[i][k][r * 2:r * 2 + l - 1] = dl_sample[i][k][:l - 1]
        # (x_1 around) (x_2): x_2 x_1 x_2 x_1 x_2 x_1 x_2 x_1
        if target_types[i] == 18:
            templates[i][k][:r] = dr_sample[i][k][:r]
            templates[i][k][r:r + l - 1] = dl_sample[i][k][:l - 1]
            templates[i][k][r + l - 1:r * 2 + l - 1] = dr_sample[i][k][:r]
            templates[i][k][r * 2 + l - 1:r * 2 + l * 2 - 2] = dl_sample[i][k][:l - 1]
            templates[i][k][r * 2 + l * 2 - 2:r * 3 + l * 2 - 2] = dr_sample[i][k][:r]
            templates[i][k][r * 3 + l * 2 - 2:r * 3 + l * 3 - 3] = dl_sample[i][k][:l - 1]
            templates[i][k][r * 3 + l * 3 - 3:r * 4 + l * 3 - 3] = dr_sample[i][k][:r]
            templates[i][k][r * 4 + l * 3 - 3:r * 4 + l * 4 - 4] = dl_sample[i][k][:l - 1]
        # (turn opposite) (x_1): x_1 x_1
        if target_types[i] == 19:
            templates[i][k][:r] = dr_sample[i][k][:r]
            templates[i][k][r:r * 2] = dr_sample[i][k][:r]
        # (turn around) (x_1): x_1 x_1 x_1 x_1
        if target_types[i] == 20:
            templates[i][k][:r] = dr_sample[i][k][:r]
            templates[i][k][r:r * 2] = dr_sample[i][k][:r]
            templates[i][k][r * 2:r * 3] = dr_sample[i][k][:r]
            templates[i][k][r * 3:r * 4] = dr_sample[i][k][:r]
        # (x_1) (opposite): x_1 opposite
        # (x_1) (around): x_1 around
        if target_types[i] in [21, 22]:
            templates[i][k][:l] = dl_sample[i][k][:l]
            templates[i][k][l] = dr_sample[i][k][0]
        # (turn) (opposite): turn opposite
        # (turn) (around): turn around
        if target_types[i] in [23, 24]:
            templates[i][k][0] = dl_sample[i][k][0]
            templates[i][k][1] = dr_sample[i][k][0]
    return templates