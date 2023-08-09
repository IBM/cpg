import re
from enum import IntEnum
from lark import Lark

import torch
from torch import nn
from torch.nn import functional as F

from src.model.basic import FeedForward, MyDataLoader, build_vocab, int_to_one_hot
from src.model.module import CopyTemplate


class SCANDataset(nn.Module):

    def __init__(self, word_dim, template_dim):
        super(SCANDataset, self).__init__()
        self.word_dim = word_dim
        self.template_dim = template_dim
        self.copy_template_len = 8
        self.max_span = 3
        self.term_list_len = 0
        self.type_dim = 21
        self.max_x_seq_len = 9
        self.max_y_seq_len = 49

        self.y_vocab = None
        self.x_vocab = None
        self.copy_template = None
        self.decoder_init = None

        # dictionary
        self.dictionary = {'run':'I_RUN', 'look':'I_LOOK', 'walk':'I_WALK', 'jump':'I_JUMP', 'left':'I_TURN_LEFT', 'right':'I_TURN_RIGHT'}

        self.curriculum = None
        self.reset_curriculum()

        self.parser = Lark(grammar, propagate_positions=True)

        self.initial_decodings = nn.ParameterDict()
        self.initial_decodings_current = nn.ParameterDict()

    def reset_hyperparameters(self, iteration_stage, eval=False):
        return

    def record_templates(self):
        self.initial_decodings.update(self.initial_decodings_current)
        self.initial_decodings_current = nn.ParameterDict()
        # record current copy templates
        self.copy_template.record_templates()

    def get_initial_dec_term(self, input, gumbel_temp, use_dictionary=False):
        B, L = input.size()
        V = len(self.y_vocab)
        
        decodings = torch.zeros(B, L, self.max_y_seq_len, V)
        decodings[:, :, :, self.y_vocab.token_to_idx('<PAD>')] = 1
        terms = torch.zeros(B, L, self.term_list_len, V)

        # generate initial decodings or use dictionary
        for i in range(B):
            for j in range(L):
                input_idx = input[i, j].item()
                input_token = self.x_vocab.idx_to_token(input_idx)
                if input_token in self.dictionary.keys():
                    if use_dictionary:
                        target_token = self.dictionary[input_token]
                        decodings[i, j, 0, :] = int_to_one_hot(self.y_vocab.token_to_idx(target_token), V)
                    elif input_token in self.initial_decodings.keys():
                        decodings[i, j, 0, :] = self.initial_decodings[input_token]
                    else:
                        if input_token in ['run', 'jump', 'look', 'walk']:
                            prim_type = 0
                        if input_token in ['left', 'right']:
                            prim_type = 1
                        decoding_logits = self.decoder_init[prim_type](F.one_hot(input[i, j], len(self.x_vocab)).float())
                        decodings[i, j, 0, :] = F.gumbel_softmax(decoding_logits.log_softmax(-1), tau=gumbel_temp, hard=True)
                        self.initial_decodings_current[input_token] = decodings[i, j, 0, :]
                else:
                    decodings[i, j, 0, :] = int_to_one_hot(self.y_vocab.token_to_idx('<PAD>'), V)
        
        return decodings, terms
    
    def get_new_types(self, types):
        B = len(types)
        new_types = torch.tensor([0 for _ in range(B)]) # default type is PAD
        for k in range(B):
            if types[k] != []:
                new_types[k] = types[k][-1]
                types[k].pop(-1)
        return new_types, types
    
    def get_pos_span(self, positions, spans):
        B = len(positions)
        new_positions = torch.zeros(B).long()
        new_spans = torch.zeros(B).long()
        for i in range(B):
            if positions[i] == []:
                new_positions[i] = 0 # default position is 0
            else:
                new_positions[i] = positions[i].pop(-1)
            if spans[i] == []:
                new_spans[i] = 2 # default span is 2
            else:
                new_spans[i] = spans[i].pop(-1)
        return new_positions, new_spans, positions, spans
    
    def transform(self, input_decodings, input_terms, new_types, spans, gumbel_temp):
        B = input_decodings.size(0)
        template_copy = self.copy_template.generate_template(new_types, spans, gumbel_temp)
        N = template_copy.size(2)
        # hard code copy templates for PAD type
        for i in range(B):
            if new_types[i] == 0:
                template_copy[i] = F.one_hot(int_to_one_hot(0, self.copy_template_len), N)
        output_decodings = self.copy_template.apply_template(input_decodings, spans, template_copy)
        output_terms = torch.zeros(B, self.term_list_len, len(self.y_vocab))
        return output_decodings, output_terms
    
    def normalize(self, decodings):
        return decodings
    
    def get_data(self, train_fp, test_fp):
        train_data = self.load_from_file(train_fp)
        test_data = self.load_from_file(test_fp)

        if self.x_vocab == None:
            self.x_vocab = build_vocab([x for x, _ in train_data + test_data], base_tokens=['<PAD>', '<UNK>'])
            self.y_vocab = build_vocab([y for _, y in train_data + test_data], base_tokens=['<PAD>', '<SOS>', '<EOS>', '<UNK>'])
            self.copy_template = CopyTemplate(self.y_vocab, self.template_dim, self.copy_template_len, self.max_span, self.type_dim)
            self.decoder_init = nn.ModuleList(FeedForward(len(self.x_vocab), [self.word_dim], len(self.y_vocab)) for _ in range(2))
        return train_data, test_data, self.x_vocab, self.y_vocab
    
    def load_from_file(self, filepath):
        with open(filepath, "rt") as SCAN_f:
            data = []
            regex = re.compile("IN: (.*) OUT: (.*)")
            for line in SCAN_f:
                if line == '\n':
                    continue
                match = regex.match(line)
                if not match:
                    raise ValueError(f"Could not parse line: \"{line}\"")
                data.append([group.split() for group in match.groups()])
        return data
    
    def parse(self, command):
        parse_tree = self.parser.parse(command)
        current_position = 0
        previous_index = 0
        positions = []
        types = []
        spans = []
        for node in parse_tree.iter_subtrees_topdown():
            if previous_index < node.meta.start_pos:
                current_position += 1
                previous_index = node.meta.start_pos
            if node.data.value in ['a', 't', 'm', 'n', 'd', 'p', 'q', 'i', 'j', 'l', 'start']:
                continue
            # ternary rules
            if node.data.value in ['v3', 'v4', 'v5', 'v6', 'c1', 'c2']:
                spans.append(3)
            else:
                spans.append(2)
            positions.append(current_position)
            types.append(token_to_type[node.data.value])
        return positions, types, spans
    
    def get_next_curriculum_stage(self):
        if self.curriculum == []:
            return None
        else:
            return self.curriculum.pop(0)
        
    def reset_curriculum(self):
        self.curriculum = [(i+1, i+1) for i in range(7)]
    
    def preprocess(self, data):
        return [(self.x_vocab.encode(x), self.y_vocab.encode(y)) for x, y in data]
    
    def load_data(self, data, curriculum_stage, batch_size):
        filter_fn = lambda x: curriculum_stage[0] <= len(x[0]) <= curriculum_stage[1]
        train_data_curriculum = list(filter(filter_fn, data))
        preprocessed_train_data = self.preprocess(train_data_curriculum)

        data_loader = MyDataLoader(preprocessed_train_data,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    x_pad_idx=self.x_vocab.token_to_idx('<PAD>'),
                                    y_pad_idx=self.y_vocab.token_to_idx('<PAD>'),
                                    max_x_seq_len=self.max_x_seq_len,
                                    max_y_seq_len=self.max_y_seq_len)
        return data_loader


class Types(IntEnum):
    PAD = 0
    A = 1
    T = 2
    M = 3
    N = 4
    D = 5
    P = 6
    Q = 7
    I = 8
    J = 9
    C1 = 10
    C2 = 11
    L = 12
    S1 = 13
    S2 = 14
    V1 = 15
    V2 = 16
    V3 = 17
    V4 = 18
    V5 = 19
    V6 = 20


grammar = """
    start: c1 | c2 | l
    c1: l i l
    c2: l j l
    l: s1 | s2 | a | v1 | v2 | v3 | v4 | v5 | v6
    s1: v1 p | v2 p | v3 p | v4 p | v5 p | v6 p | a p
    s2: v1 q | v2 q | v3 q | v4 q | v5 q | v6 q | a q
    v1: a d
    v2: t d
    v3: a m d
    v4: a n d
    v5: t m d
    v6: t n d
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

token_to_type = {
    "twice": Types.P,
    "thrice": Types.Q,
    "walk": Types.A,
    "look": Types.A,
    "run": Types.A,
    "jump": Types.A,
    "opposite": Types.M,
    "around": Types.N,
    "left": Types.D,
    "right": Types.D,
    "and": Types.I,
    "after": Types.J,
    "turn": Types.T,
    "c1": Types.C1,
    "c2": Types.C2,
    "l": Types.L,
    "s1": Types.S1,
    "s2": Types.S2,
    "v1": Types.V1,
    "v2": Types.V2,
    "v3": Types.V3,
    "v4": Types.V4,
    "v5": Types.V5,
    "v6": Types.V6,
}