import csv
from enum import IntEnum
from lark import Lark

import torch
from torch import functional as F
from torch import nn

from src.model.basic import MyDataLoader, build_vocab, int_to_one_hot, quote
from src.model.module import CopyTemplate, SubstitutionTemplate


class COGSDataset(nn.Module):

    def __init__(self, word_dim, template_dim):
        super(COGSDataset, self).__init__()
        self.template_dim = template_dim
        self.copy_template_len = 5
        self.max_span = 5
        self.sub_template_len = 15
        self.term_list_len = 80
        self.type_dim = 62
        self.max_x_seq_len = 20
        self.max_y_seq_len = 160
        self.predict_zero = True

        self.y_vocab = None
        self.x_vocab = None
        self.copy_template = None
        self.substitution_template = None
        self.template_type_len = torch.full([self.type_dim, self.max_span], 0)

        # dictionary
        self.initial_decodings_non_vtype = {'the' : '* y ( y )'} |\
                                           {'a' : 'y ( y )'} |\
                                           {'on' : 'y . nmod . on ( y , y )'} |\
                                           {'in' : 'y . nmod . in ( y , y )'} |\
                                           {'beside' : 'y . nmod . beside ( y , y )'}
        
        self.initial_decodings_vtype = {41: 'y . agent ( y , y )',
                                        42: 'y . agent ( y , y ) y . theme ( y , y )',
                                        43: 'y . theme ( y , y )',
                                        44: 'y . theme ( y , y ) y . agent ( y , y )',
                                        45: 'y . agent ( y , y ) y . theme ( y , y )',
                                        46: 'y . theme ( y , y )',
                                        47: 'y . theme ( y , y ) y . agent ( y , y )',
                                        48: 'y . agent ( y , y ) y . ccomp ( y , y )',
                                        49: 'y . agent ( y , y ) y . xcomp ( y , y )',
                                        50: 'y . agent ( y , y ) y . theme ( y , y )',
                                        51: 'y . theme ( y , y )',
                                        52: 'y . theme ( y , y )',
                                        53: 'y . theme ( y , y ) y . agent ( y , y )',
                                        54: 'y . agent ( y , y )',
                                        55: 'y . agent ( y , y )',
                                        56: 'y . agent ( y , y ) y . theme ( y , y ) y . recipient ( y , y )',
                                        57: 'y . agent ( y , y ) y . recipient ( y , y ) y . theme ( y , y )',
                                        58: 'y . theme ( y , y ) y . recipient ( y , y )',
                                        59: 'y . theme ( y , y ) y . recipient ( y , y ) y . agent ( y , y )',
                                        60: 'y . recipient ( y , y ) y . theme ( y , y )',
                                        61: 'y . recipient ( y , y ) y . theme ( y , y ) y . agent ( y , y )'}
        
        self.initial_terms = {noun : noun for noun in set(noun_list)} | verbs_lemmas | {verb : verb for verb in set(V_inf)}

        self.curriculum = None
        self.reset_curriculum()

        self.parser = Lark(grammar, propagate_positions=True)

    def reset_hyperparameters(self, iteration_stage):
        if iteration_stage > 2000:
            self.predict_zero = True
        else:
            self.predict_zero = False
    
    def record_templates(self):
        self.substitution_template.record_templates()
    
    def get_initial_dec_term(self, input, gumbel_temp):
        B, L = input.size()
        V = len(self.y_vocab)

        decodings = torch.zeros(B, L, self.max_y_seq_len, V)
        decodings[:, :, :, self.y_vocab.token_to_idx('<PAD>')] = 1
        terms = torch.zeros(B, L, self.term_list_len, V)

        for i in range(B):
            for j in range(L):
                input_token = self.x_vocab.idx_to_token(input[i, j].item())
                if input_token == '<PAD>':
                    continue
                if input_token in self.initial_decodings_non_vtype.keys():
                    target_tokens = self.initial_decodings_non_vtype[input_token]
                else:
                    target_tokens = ''
                # get initial terms
                if input_token in self.initial_terms.keys():
                    target_term = self.initial_terms[input_token]
                    # switch order of variables and constants for proper nouns
                    if input_token in proper_nouns:
                        terms[i, j, 1, :] = int_to_one_hot(self.y_vocab.token_to_idx(str(j)), V)
                        terms[i, j, 0, :] = int_to_one_hot(self.y_vocab.token_to_idx(target_term), V)
                    else:
                        terms[i, j, 0, :] = int_to_one_hot(self.y_vocab.token_to_idx(str(j)), V)
                        terms[i, j, 1, :] = int_to_one_hot(self.y_vocab.token_to_idx(target_term), V)
                # get initial decodings for non-verb tokens
                target_tokens = [token for token in target_tokens.split(' ') if token != '']
                for k in range(len(target_tokens)):
                    decodings[i, j, k] = int_to_one_hot(self.y_vocab.token_to_idx(target_tokens[k]), V)
        return decodings, terms

    def get_new_types(self, types):
        B = len(types)
        new_types = torch.tensor([0 for _ in range(B)])
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
                new_positions[i] = 0
            else:
                new_positions[i] = positions[i].pop(-1)
            if spans[i] == []:
                new_spans[i] = 2
            else:
                new_spans[i] = spans[i].pop(-1)
        return new_positions, new_spans, positions, spans
    
    def transform(self, input_decodings, input_terms, new_types, spans, gumbel_temp):
        B, N, M, V = input_decodings.size()

        # hard code copy templates
        template_copy = self.copy_template.generate_template(new_types, spans, gumbel_temp, concat=True)
        output_decodings = self.copy_template.apply_template(input_decodings, spans, template_copy)

        # get initial decodings for verbs
        for i in range(B):
            new_type = new_types[i].item()
            if new_type not in self.initial_decodings_vtype.keys():
                continue
            target_tokens = self.initial_decodings_vtype[new_type]
            target_tokens = [token for token in target_tokens.split(' ') if token != '']
            for k in range(len(target_tokens)):
                output_decodings[i, k] = int_to_one_hot(self.y_vocab.token_to_idx(target_tokens[k]), V)

        # concatenate input term lists
        template_idx = []
        output_terms = torch.zeros(B, self.term_list_len, V)
        for i in range(B):
            new_type = new_types[i].item()
            # copy input variables
            var_idx = 0
            type_idx = []
            for j in range(N):
                type_idx.append(0)
                for k in range(self.term_list_len):
                    if not torch.equal(input_terms[i, j, k], torch.zeros(V)):
                        # do not copy terms past the initial input term list lengths
                        if not torch.equal(self.template_type_len[new_type], torch.zeros(self.max_span)) \
                                and type_idx[j] == self.template_type_len[new_type, j].item():
                            break
                        output_terms[i, var_idx] = input_terms[i, j, k]
                        var_idx += 1
                        type_idx[j] += 1
            # save the number of terms from each input type
            if torch.equal(self.template_type_len[new_type], torch.zeros(self.max_span)):
                self.template_type_len[new_type, :len(type_idx)] = torch.tensor(type_idx).float()
            template_idx.append(self.template_type_len[new_type].sum().int())

        # generate substitution templates
        template_sub = self.substitution_template.generate_template(new_types, template_idx, gumbel_temp, self.predict_zero)
        for i in range(B):
            new_type = new_types[i].item()
            # skip substitution process for padding and some verbs
            if new_type == 0 or (40 < new_type < 62 and new_type not in [41, 51, 54]):
                template_sub[i] = torch.zeros(self.sub_template_len, self.term_list_len+1)
                template_sub[i, :, 0] = torch.ones(self.sub_template_len)
        
        output_decodings = self.substitution_template.apply_template(output_decodings, output_terms, template_sub)
        return output_decodings, output_terms
    
    def normalize(self, decodings):
        B, M, V = decodings.size()

        new_d = torch.zeros(B, M, V)
        new_d[:, :, self.y_vocab.token_to_idx('<PAD>')] = 1
        ast_vector = int_to_one_hot(self.y_vocab.token_to_idx('*'), V)
        rp_vector = int_to_one_hot(self.y_vocab.token_to_idx(')'), V)
        pad_vector = int_to_one_hot(self.y_vocab.token_to_idx('<PAD>'), V)

        for i in range(B):
            idx = 0
            # copy the existentially quantified expressions
            for k in range(M):
                if not torch.equal(decodings[i, k], ast_vector):
                    continue
                exp_idx = k
                while exp_idx == k or (exp_idx != k and not torch.equal(new_d[i, idx-1], rp_vector)):
                    new_d[i, idx] = decodings[i, exp_idx]
                    decodings[i, exp_idx] = pad_vector
                    exp_idx += 1
                    idx += 1
            # copy the rest
            for k in range(M):
                if not torch.equal(decodings[i, k], pad_vector) and not torch.equal(decodings[i, k], torch.zeros(V)):
                    new_d[i, idx] = decodings[i, k]
                    idx += 1
        return new_d
        
    def preprocess(self, data):
        return [(self.x_vocab.encode(x), self.y_vocab.encode(y)) for x, y in data]
    
    def get_data(self, train_fp, test_fp):
        train_data = self.load_from_file(train_fp)
        test_data = self.load_from_file(test_fp)

        self.x_vocab = build_vocab([x for x, _ in train_data + test_data], base_tokens=['<PAD>', '<UNK>'])
        self.y_vocab = build_vocab([y for _, y in train_data + test_data], base_tokens=['<PAD>', '<SOS>', '<EOS>', '<UNK>'])
        self.y_vocab.add_token("y")
        for i in range(self.max_x_seq_len):
            self.y_vocab.add_token(str(i))

        if self.copy_template == None:
            self.copy_template = CopyTemplate(self.y_vocab, self.template_dim, self.copy_template_len, self.max_span, self.type_dim)
            self.substitution_template = SubstitutionTemplate(self.y_vocab, self.template_dim, self.sub_template_len, self.term_list_len, self.type_dim)
        else:
            self.copy_template.y_vocab = self.y_vocab
            self.substitution_template.y_vocab = self.y_vocab
        return train_data, test_data, self.x_vocab, self.y_vocab
    
    def load_from_file(self, filepath):
        with open(filepath, "rt") as file:
            tsv_file = csv.reader(file, delimiter="\t")
            data = []
            for line in tsv_file:
                x = line[:2][0].replace(".", "").lower().split()
                # remove 'x', '_' , ';', 'AND'
                y = line[:2][1].replace(' x ', '').replace('_', '').replace('AND', '').replace(';', '').lower().split()
                data.append([x, y])
        return data
    
    def parse(self, command):
        # create parse tree
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
            if node.data in exclude_types:
                continue
            # determine span length
            if node.data.value in one_span_types:
                spans.append(1)
            elif node.data.value in two_span_types:
                spans.append(2)
            elif node.data.value in three_span_types:
                spans.append(3)
            elif node.data.value in four_span_types:
                spans.append(4)
            elif node.data.value in five_span_types:
                spans.append(5)
            else:
                assert(False)
            positions.append(current_position)
            types.append(token_to_type[node.data.value])
        return positions, types, spans
    
    def get_next_curriculum_stage(self):
        if self.curriculum == []:
            return None
        else:
            return self.curriculum.pop(0)
        
    def reset_curriculum(self):
        self.curriculum = [(2, 3)]
        self.curriculum.extend([(3*i+4, 3*i+6) for i in range(5)])
    
    def preprocess(self, data):
        return [(self.x_vocab.encode(x), self.y_vocab.encode(y)) for x, y in data]
    
    def load_data(self, data, curriculum_stage, batch_size):
        filter_fn = lambda x: curriculum_stage[0] <= len(x[0]) <= curriculum_stage[1]
        data_curriculum = list(filter(filter_fn, data))
        preprocessed_data = self.preprocess(data_curriculum)

        data_loader = MyDataLoader(preprocessed_data,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    x_pad_idx=self.x_vocab.token_to_idx('<PAD>'),
                                    y_pad_idx=self.y_vocab.token_to_idx('<PAD>'),
                                    max_x_seq_len=self.max_x_seq_len,
                                    max_y_seq_len=self.max_y_seq_len)
        return data_loader


animate_nouns = [
    'girl', 'boy', 'cat', 'dog', 'baby', 'child', 'teacher', 'frog', 'chicken', 'mouse',
    'lion', 'monkey', 'bear', 'giraffe', 'horse', 'bird', 'duck', 'bunny', 'butterfly', 'penguin',
    'student', 'professor', 'monster', 'hero', 'sailor', 'lawyer', 'customer', 'scientist', 'princess', 'president',
    'cow', 'crocodile', 'goose', 'hen', 'deer', 'donkey', 'bee', 'fly', 'kitty', 'tiger',
    'wolf', 'zebra', 'mother', 'father', 'patient', 'manager', 'director', 'king', 'queen', 'kid',
    'fish', 'moose', 'pig', 'pony', 'puppy', 'sheep', 'squirrel', 'lamb', 'turkey', 'turtle',
    'doctor', 'pupil', 'prince', 'driver', 'consumer', 'writer', 'farmer', 'friend', 'judge', 'visitor',
    'guest', 'servant', 'chief', 'citizen', 'champion', 'prisoner', 'captain', 'soldier', 'passenger', 'tenant',
    'politician', 'resident', 'buyer', 'spokesman', 'governor', 'guard', 'creature', 'coach', 'producer', 'researcher',
    'guy', 'dealer', 'duke', 'tourist', 'landlord', 'human', 'host', 'priest', 'journalist', 'poet', 'hedgehog',
    'shark', 'cockroach', 'cobra', 'hippo'
]

inanimate_nouns = [
    'cake', 'donut', 'cookie', 'box', 'rose', 'drink', 'raisin', 'melon', 'sandwich', 'strawberry',
    'ball', 'balloon', 'bat', 'block', 'book', 'crayon', 'chalk', 'doll', 'game', 'glue',
    'lollipop', 'hamburger', 'banana', 'biscuit', 'muffin', 'pancake', 'pizza', 'potato', 'pretzel', 'pumpkin',
    'sweetcorn', 'yogurt', 'pickle', 'jigsaw', 'pen', 'pencil', 'present', 'toy', 'cracker', 'brush',
    'radio', 'cloud', 'mandarin', 'hat', 'basket', 'plant', 'flower', 'chair', 'spoon', 'pillow',
    'gumball', 'scarf', 'shoe', 'jacket', 'hammer', 'bucket', 'knife', 'cup', 'plate', 'towel',
    'bottle', 'bowl', 'can', 'clock', 'jar', 'penny', 'purse', 'soap', 'toothbrush', 'watch',
    'newspaper', 'fig', 'bag', 'wine', 'key', 'weapon', 'brain', 'tool', 'crown', 'ring',
    'leaf', 'fruit', 'mirror', 'beer', 'shirt', 'guitar', 'chemical', 'seed', 'shell', 'brick',
    'bell', 'coin', 'button', 'needle', 'molecule', 'crystal', 'flag', 'nail', 'bean', 'liver'
]

proper_nouns = [
    'Emma', 'Liam', 'Olivia', 'Noah', 'Ava', 'William', 'Isabella', 'James', 'Sophia', 'Oliver',
    'Charlotte', 'Benjamin', 'Mia', 'Elijah', 'Amelia', 'Lucas', 'Harper', 'Mason', 'Evelyn', 'Logan',
    'Abigail', 'Alexander', 'Emily', 'Ethan', 'Elizabeth', 'Jacob', 'Mila', 'Michael', 'Ella', 'Daniel',
    'Avery', 'Henry', 'Sofia', 'Jackson', 'Camila', 'Sebastian', 'Aria', 'Aiden', 'Scarlett', 'Matthew',
    'Victoria', 'Samuel', 'Madison', 'David', 'Luna', 'Joseph', 'Grace', 'Carter', 'Chloe', 'Owen',
    'Penelope', 'Wyatt', 'Layla', 'John', 'Riley', 'Jack', 'Zoey', 'Luke', 'Nora', 'Jayden',
    'Lily', 'Dylan', 'Eleanor', 'Grayson', 'Hannah', 'Levi', 'Lillian', 'Isaac', 'Addison', 'Gabriel',
    'Aubrey', 'Julian', 'Ellie', 'Mateo', 'Stella', 'Anthony', 'Natalie', 'Jaxon', 'Zoe', 'Lincoln',
    'Leah', 'Joshua', 'Hazel', 'Christopher', 'Violet', 'Andrew', 'Aurora', 'Theodore', 'Savannah', 'Caleb',
    'Audrey', 'Ryan', 'Brooklyn', 'Asher', 'Bella', 'Nathan', 'Claire', 'Thomas', 'Skylar', 'Leo', 'Lina',
    'Paula', 'Charlie'
]

proper_nouns = list(map(lambda n: n.lower(), proper_nouns))

on_nouns = [
    'table', 'stage', 'bed', 'chair', 'stool', 'road', 'tree', 'box', 'surface', 'seat',
    'speaker', 'computer', 'rock', 'boat', 'cabinet', 'tv', 'plate', 'desk', 'bowl', 'bench',
    'shelf', 'cloth', 'piano', 'bible', 'leaflet', 'sheet', 'cupboard', 'truck', 'tray', 'notebook',
    'blanket', 'deck', 'coffin', 'log', 'ladder', 'barrel', 'rug', 'canvas', 'tiger', 'towel',
    'throne', 'booklet', 'sock', 'corpse', 'sofa', 'keyboard', 'book', 'pillow', 'pad', 'train',
    'couch', 'bike', 'pedestal', 'platter', 'paper', 'rack', 'board', 'panel', 'tripod', 'branch',
    'machine', 'floor', 'napkin', 'cookie', 'block', 'cot', 'device', 'yacht', 'dog', 'mattress',
    'ball', 'stand', 'stack', 'windowsill', 'counter', 'cushion', 'hanger', 'trampoline', 'gravel', 'cake',
    'carpet', 'plaque', 'boulder', 'leaf', 'mound', 'bun', 'dish', 'cat', 'podium', 'tabletop',
    'beach', 'bag', 'glacier', 'brick', 'crack', 'vessel', 'futon', 'turntable', 'rag', 'chessboard'
]

in_nouns = [
    'house', 'room', 'car', 'garden', 'box', 'cup', 'glass', 'bag', 'vehicle', 'hole',
    'cabinet', 'bottle', 'shoe', 'storage', 'cot', 'vessel', 'pot', 'pit', 'tin', 'can',
    'cupboard', 'envelope', 'nest', 'bush', 'coffin', 'drawer', 'container', 'basin', 'tent', 'soup',
    'well', 'barrel', 'bucket', 'cage', 'sink', 'cylinder', 'parcel', 'cart', 'sack', 'trunk',
    'wardrobe', 'basket', 'bin', 'fridge', 'mug', 'jar', 'corner', 'pool', 'blender', 'closet',
    'pile', 'van', 'trailer', 'saucepan', 'truck', 'taxi', 'haystack', 'dumpster', 'puddle', 'bathtub',
    'pod', 'tub', 'trap', 'bun', 'microwave', 'bookstore', 'package', 'cafe', 'train', 'castle',
    'bunker', 'vase', 'backpack', 'tube', 'hammock', 'stadium', 'backyard', 'swamp', 'monastery', 'refrigerator',
    'palace', 'cubicle', 'crib', 'condo', 'tower', 'crate', 'dungeon', 'teapot', 'tomb', 'casket',
    'jeep', 'shoebox', 'wagon', 'bakery', 'fishbowl', 'kennel', 'china', 'spaceship', 'penthouse', 'pyramid'
]

beside_nouns = [
    'table', 'stage', 'bed', 'chair', 'book', 'road', 'tree', 'machine', 'house', 'seat',
    'speaker', 'computer', 'rock', 'car', 'box', 'cup', 'glass', 'bag', 'flower', 'boat',
    'vehicle', 'key', 'painting', 'cabinet', 'tv', 'bottle', 'cat', 'desk', 'shoe', 'mirror',
    'clock', 'bench', 'bike', 'lamp', 'lion', 'piano', 'crystal', 'toy', 'duck', 'sword',
    'sculpture', 'rod', 'truck', 'basket', 'bear', 'nest', 'sphere', 'bush', 'surgeon', 'poster',
    'throne', 'giant', 'trophy', 'hedge', 'log', 'tent', 'ladder', 'helicopter', 'barrel', 'yacht',
    'statue', 'bucket', 'skull', 'beast', 'lemon', 'whale', 'cage', 'gardner', 'fox', 'sink',
    'trainee', 'dragon', 'cylinder', 'monk', 'bat', 'headmaster', 'philosopher', 'foreigner', 'worm', 'chemist',
    'corpse', 'wolf', 'torch', 'sailor', 'valve', 'hammer', 'doll', 'genius', 'baron', 'murderer',
    'bicycle', 'keyboard', 'stool', 'pepper', 'warrior', 'pillar', 'monkey', 'cassette', 'broker', 'bin'

]

noun_list = animate_nouns + inanimate_nouns + proper_nouns + on_nouns + in_nouns + beside_nouns

V_trans_omissible = [
    'ate', 'painted', 'drew', 'cleaned', 'cooked',
    'dusted', 'hunted', 'nursed', 'sketched', 'juggled',
    'called', 'heard', 'packed', 'saw', 'noticed',
    'studied', 'examined', 'observed', 'knew', 'investigated', 'baked'
]

V_trans_omissible_pp = [
    'eaten', 'painted', 'drawn', 'cleaned', 'cooked',
    'dusted', 'hunted', 'nursed', 'sketched', 'juggled',
    'called', 'heard', 'packed', 'seen', 'noticed',
    'studied', 'examined', 'observed', 'known', 'investigated'
]

V_trans_not_omissible = [
    'liked', 'helped', 'found', 'loved', 'poked',
    'admired', 'adored', 'appreciated', 'missed', 'respected',
    'threw', 'tolerated', 'valued', 'worshipped', 'discovered',
    'held', 'stabbed', 'touched', 'pierced', 'tossed'
]

V_trans_not_omissible_pp = [
    'liked', 'helped', 'found', 'loved', 'poked',
    'admired', 'adored', 'appreciated', 'missed', 'respected',
    'thrown', 'tolerated', 'valued', 'worshipped', 'discovered',
    'held', 'stabbed', 'touched', 'pierced', 'tossed'
]

V_cp_taking = [
    'liked', 'hoped', 'said', 'noticed', 'believed',
    'confessed', 'declared', 'proved', 'thought', 'admired',
    'appreciated', 'respected', 'supported', 'tolerated', 'valued',
    'wished', 'dreamed', 'expected', 'imagined', 'meant'
]

V_inf_taking = [
    'wanted', 'preferred', 'needed', 'intended', 'tried',
    'attempted', 'planned', 'expected', 'hoped', 'wished',
    'craved', 'liked', 'hated', 'loved', 'enjoyed',
    'dreamed', 'meant', 'longed', 'yearned', 'itched'
]

V_unacc = [
    'rolled', 'froze', 'burned', 'shortened', 'floated',
    'grew', 'slid', 'broke', 'crumpled', 'split',
    'changed', 'snapped', 'disintegrated', 'collapsed', 'decomposed',
    'doubled', 'improved', 'inflated', 'enlarged', 'reddened',
    'shattered', 'blessed', 'squeezed'
]

V_unacc_pp = [
    'rolled', 'frozen', 'burned', 'shortened', 'floated',
    'grown', 'slid', 'broken', 'crumpled', 'split',
    'changed', 'snapped', 'disintegrated', 'collapsed', 'decomposed',
    'doubled', 'improved', 'inflated', 'enlarged', 'reddened',
    'shattered', 'blessed', 'squeezed'
]

V_unerg = [
    'slept', 'smiled', 'laughed', 'sneezed', 'cried',
    'talked', 'danced', 'jogged', 'walked', 'ran',
    'napped', 'snoozed', 'screamed', 'stuttered', 'frowned',
    'giggled', 'scoffed', 'snored', 'smirked', 'gasped'
]

V_inf = [
    'walk', 'run', 'sleep', 'sneeze', 'nap',
    'eat', 'read', 'cook', 'hunt', 'paint',
    'talk', 'dance', 'giggle', 'jog', 'smirk',
    'call', 'sketch', 'dust', 'clean', 'investigate', 'crawl'
]

V_dat = [
    'gave', 'lended', 'sold', 'offered', 'fed',
    'passed', 'sent', 'rented', 'served', 'awarded',
    'brought', 'handed', 'forwarded', 'promised', 'mailed',
    'loaned', 'posted', 'returned', 'slipped', 'wired',
    'teleported', 'shipped'
]

V_dat_pp = [
    'given', 'lended', 'sold', 'offered', 'fed',
    'passed', 'sent', 'rented', 'served', 'awarded',
    'brought', 'handed', 'forwarded', 'promised', 'mailed',
    'loaned', 'posted', 'returned', 'slipped', 'wired'
]

verbs_lemmas = {
    'ate': 'eat', 'painted': 'paint', 'drew': 'draw', 'cleaned': 'clean',
    'cooked': 'cook', 'dusted': 'dust', 'hunted': 'hunt', 'nursed': 'nurse',
    'sketched': 'sketch', 'washed': 'wash', 'juggled': 'juggle', 'called': 'call',
    'eaten': 'eat', 'drawn': 'draw', 'baked': 'bake', 'liked': 'like', 'knew': 'know',
    'helped': 'help', 'saw': 'see', 'found': 'find', 'heard': 'hear', 'noticed': 'notice',
    'loved': 'love', 'admired': 'admire', 'adored': 'adore', 'appreciated': 'appreciate',
    'missed': 'miss', 'respected': 'respect', 'tolerated': 'tolerate', 'valued': 'value',
    'worshipped': 'worship', 'observed': 'observe', 'discovered': 'discover', 'held': 'hold',
    'stabbed': 'stab', 'touched': 'touch', 'pierced': 'pierce', 'poked': 'poke',
    'known': 'know', 'seen': 'see', 'hit': 'hit', 'hoped': 'hope', 'said': 'say',
    'believed': 'believe', 'confessed': 'confess', 'declared': 'declare', 'proved': 'prove',
    'thought': 'think', 'supported': 'support', 'wished': 'wish', 'dreamed': 'dream',
    'expected': 'expect', 'imagined': 'imagine', 'envied': 'envy', 'wanted': 'want',
    'preferred': 'prefer', 'needed': 'need', 'intended': 'intend', 'tried': 'try',
    'attempted': 'attempt', 'planned': 'plan', 'craved': 'crave', 'hated': 'hate', 'loved': 'love',
    'enjoyed': 'enjoy', 'rolled': 'roll', 'froze': 'freeze', 'burned': 'burn', 'shortened': 'shorten',
    'floated': 'float', 'grew': 'grow', 'slid': 'slide', 'broke': 'break', 'crumpled': 'crumple',
    'split': 'split', 'changed': 'change', 'snapped': 'snap', 'tore': 'tear', 'collapsed': 'collapse',
    'decomposed': 'decompose', 'doubled': 'double', 'improved': 'improve', 'inflated': 'inflate',
    'enlarged': 'enlarge', 'reddened': 'redden', 'popped': 'pop', 'disintegrated': 'disintegrate',
    'expanded': 'expand', 'cooled': 'cool', 'soaked': 'soak', 'frozen': 'freeze', 'grown': 'grow',
    'broken': 'break', 'torn': 'tear', 'slept': 'sleep', 'smiled': 'smile', 'laughed': 'laugh',
    'sneezed': 'sneeze', 'cried': 'cry', 'talked': 'talk', 'danced': 'dance', 'jogged': 'jog',
    'walked': 'walk', 'ran': 'run', 'napped': 'nap', 'snoozed': 'snooze', 'screamed': 'scream',
    'stuttered': 'stutter', 'frowned': 'frown', 'giggled': 'giggle', 'scoffed': 'scoff',
    'snored': 'snore', 'snorted': 'snort', 'smirked': 'smirk', 'gasped': 'gasp',
    'gave': 'give', 'lended': 'lend', 'sold': 'sell', 'offered': 'offer', 'fed': 'feed',
    'passed': 'pass', 'rented': 'rent', 'served': 'serve', 'awarded': 'award', 'promised': 'promise',
    'brought': 'bring', 'sent': 'send', 'handed': 'hand', 'forwarded': 'forward', 'mailed': 'mail',
    'posted': 'post', 'given': 'give', 'shipped': 'ship', 'packed': 'pack', 'studied': 'study',
    'examined': 'examine', 'investigated': 'investigate', 'thrown': 'throw', 'threw': 'throw',
    'tossed': 'toss', 'meant': 'mean', 'longed': 'long', 'yearned': 'yearn', 'itched': 'itch',
    'loaned': 'loan', 'returned': 'return', 'slipped': 'slip', 'wired': 'wire', 'crawled': 'crawl',
    'shattered': 'shatter', 'bought': 'buy', 'squeezed': 'squeeze', 'teleported': 'teleport',
    'melted': 'melt', 'blessed': 'bless'
}

grammar = """
    start: s1 | s2 | s3 | s4 | vp_internal
    s1: np vp_external
    s2: np vp_passive
    s3: np vp_passive_dat
    s4: np vp_external4
    vp_external: v_unerg | v_trans_omissible_p1 | vp_external1 | vp_external2 | vp_external3 | vp_external5 | vp_external6 | vp_external7
    vp_external1: v_unacc_p1 np
    vp_external2: v_trans_omissible_p2 np
    vp_external3: v_trans_not_omissible np
    vp_external4: v_inf_taking to v_inf
    vp_external5: v_cp_taking that start
    vp_external6: v_dat_p1 np pp_iobj
    vp_external7: v_dat_p2 np np
    vp_internal: np v_unacc_p2
    vp_passive: vp_passive1 | vp_passive2 | vp_passive3 | vp_passive4 | vp_passive5 | vp_passive6 | vp_passive7 | vp_passive8
    vp_passive1: was v_trans_not_omissible_pp_p1
    vp_passive2: was v_trans_not_omissible_pp_p2 by np
    vp_passive3: was v_trans_omissible_pp_p1
    vp_passive4: was v_trans_omissible_pp_p2 by np
    vp_passive5: was v_unacc_pp_p1
    vp_passive6: was v_unacc_pp_p2 by np
    vp_passive7: was v_dat_pp_p1 pp_iobj
    vp_passive8: was v_dat_pp_p2 pp_iobj by np
    vp_passive_dat: vp_passive_dat1 | vp_passive_dat2
    vp_passive_dat1: was v_dat_pp_p3 np
    vp_passive_dat2: was v_dat_pp_p4 np by np
    np: np_prop | np_det | np_pp
    np_prop: proper_noun
    np_det: det common_noun
    np_pp: det common_noun pp_loc
    pp_loc: pp np
    pp_iobj: to np
    det: \"the\" | \"a\"
    pp: \"on\" | \"in\" | \"beside\"
    was: \"was\"
    by: \"by\"
    to: \"to\"
    that: \"that\"
    common_noun: {animate_nouns_str} | {inanimate_nouns_str} | {on_nouns_str} | {in_nouns_str} | {beside_nouns_str}
    proper_noun: {proper_nouns_str}
    v_trans_omissible_p1: {V_trans_omissible_str}
    v_trans_omissible_p2: {V_trans_omissible_str}
    v_trans_omissible_pp_p1: {V_trans_omissible_pp_str}
    v_trans_omissible_pp_p2: {V_trans_omissible_pp_str}
    v_trans_not_omissible: {V_trans_not_omissible_str}
    v_trans_not_omissible_pp_p1: {V_trans_not_omissible_pp_str}
    v_trans_not_omissible_pp_p2: {V_trans_not_omissible_pp_str}
    v_cp_taking: {V_cp_taking_str}
    v_inf_taking: {V_inf_taking_str}
    v_unacc_p1: {V_unacc_str}
    v_unacc_p2: {V_unacc_str}
    v_unacc_pp_p1: {V_unacc_pp_str}
    v_unacc_pp_p2: {V_unacc_pp_str}
    v_unerg: {V_unerg_str}
    v_inf: {V_inf_str}
    v_dat_p1: {V_dat_str}
    v_dat_p2: {V_dat_str}
    v_dat_pp_p1: {V_dat_pp_str}
    v_dat_pp_p2: {V_dat_pp_str}
    v_dat_pp_p3: {V_dat_pp_str}
    v_dat_pp_p4: {V_dat_pp_str}
    
    %import common.LETTER
    %import common.WS
    %ignore WS

""".format(animate_nouns_str=" | ".join(quote(animate_nouns)),
           inanimate_nouns_str=" | ".join(quote(inanimate_nouns)),
           proper_nouns_str=" | ".join(quote(proper_nouns)),
           in_nouns_str=" | ".join(quote(in_nouns)),
           on_nouns_str=" | ".join(quote(on_nouns)),
           beside_nouns_str=" | ".join(quote(beside_nouns)),
           V_trans_omissible_str=" | ".join(quote(V_trans_omissible)),
           V_trans_omissible_pp_str=" | ".join(quote(V_trans_omissible_pp)),
           V_trans_not_omissible_str=" | ".join(quote(V_trans_not_omissible)),
           V_trans_not_omissible_pp_str=" | ".join(quote(V_trans_not_omissible_pp)),
           V_cp_taking_str=" | ".join(quote(V_cp_taking)),
           V_inf_taking_str=" | ".join(quote(V_inf_taking)),
           V_unacc_str=" | ".join(quote(V_unacc)),
           V_unacc_pp_str=" | ".join(quote(V_unacc_pp)),
           V_unerg_str=" | ".join(quote(V_unerg)),
           V_inf_str=" | ".join(quote(V_inf)),
           V_dat_str=" | ".join(quote(V_dat)),
           V_dat_pp_str=" | ".join(quote(V_dat_pp))
           )


class Types(IntEnum):
    PAD = 0
    START = 1
    S1 = 2
    S2 = 3
    S3 = 4
    S4 = 5
    VP_EXTERNAL = 6
    VP_EXTERNAL1 = 7
    VP_EXTERNAL2 = 8
    VP_EXTERNAL3 = 9
    VP_EXTERNAL4 = 10
    VP_EXTERNAL5 = 11
    VP_EXTERNAL6 = 12
    VP_EXTERNAL7 = 13
    VP_INTERNAL = 14
    VP_PASSIVE = 15
    VP_PASSIVE1 = 16
    VP_PASSIVE2 = 17
    VP_PASSIVE3 = 18
    VP_PASSIVE4 = 19
    VP_PASSIVE5 = 20
    VP_PASSIVE6 = 21
    VP_PASSIVE7 = 22
    VP_PASSIVE8 = 23
    VP_PASSIVE_DAT = 24
    VP_PASSIVE_DAT1 = 25
    VP_PASSIVE_DAT2 = 26
    NP = 27
    NP_PROP = 28
    NP_DET = 29
    NP_PP = 30
    PP_LOC = 31
    PP_IOBJ = 32
    DET = 33
    PP = 34
    WAS = 35
    BY = 36
    TO = 37
    THAT = 38
    COMMON_NOUN = 39
    PROPER_NOUN = 40
    V_TRANS_OMISSABLE_P1 = 41
    V_TRANS_OMISSABLE_P2 = 42
    V_TRANS_OMISSABLE_PP_P1 = 43
    V_TRANS_OMISSABLE_PP_P2 = 44
    V_TRANS_NOT_OMISSABLE = 45
    V_TRANS_NOT_OMISSABLE_PP_P1 = 46
    V_TRANS_NOT_OMISSABLE_PP_P2 = 47
    V_CP_TAKING = 48
    V_INF_TAKING = 49
    V_UNACC_P1 = 50
    V_UNACC_P2 = 51
    V_UNACC_PP_P1 = 52
    V_UNACC_PP_P2 = 53
    V_UNERG = 54
    V_INF = 55
    V_DAT_P1 = 56
    V_DAT_P2 = 57
    V_DAT_PP_P1 = 58
    V_DAT_PP_P2 = 59
    V_DAT_PP_P3 = 60
    V_DAT_PP_P4 = 61


token_to_type = {
    "start": Types.START,
    "s1": Types.S1,
    "s2": Types.S2,
    "s3": Types.S3,
    "s4": Types.S4,
    "vp_external": Types.VP_EXTERNAL,
    "vp_external1": Types.VP_EXTERNAL1,
    "vp_external2": Types.VP_EXTERNAL2,
    "vp_external3": Types.VP_EXTERNAL3,
    "vp_external4": Types.VP_EXTERNAL4,
    "vp_external5": Types.VP_EXTERNAL5,
    "vp_external6": Types.VP_EXTERNAL6,
    "vp_external7": Types.VP_EXTERNAL7,
    "vp_internal": Types.VP_INTERNAL,
    "vp_passive": Types.VP_PASSIVE,
    "vp_passive1": Types.VP_PASSIVE1,
    "vp_passive2": Types.VP_PASSIVE2,
    "vp_passive3": Types.VP_PASSIVE3,
    "vp_passive4": Types.VP_PASSIVE4,
    "vp_passive5": Types.VP_PASSIVE5,
    "vp_passive6": Types.VP_PASSIVE6,
    "vp_passive7": Types.VP_PASSIVE7,
    "vp_passive8": Types.VP_PASSIVE8,
    "vp_passive_dat": Types.VP_PASSIVE_DAT,
    "vp_passive_dat1": Types.VP_PASSIVE_DAT1,
    "vp_passive_dat2": Types.VP_PASSIVE_DAT2,
    "np": Types.NP,
    "np_prop": Types.NP_PROP,
    "np_det": Types.NP_DET,
    "np_pp": Types.NP_PP,
    "pp_loc": Types.PP_LOC,
    "pp_iobj": Types.PP_IOBJ,
    "det": Types.DET,
    "pp": Types.PP,
    "was": Types.WAS,
    "by": Types.BY,
    "to": Types.TO,
    "that": Types.THAT,
    "common_noun": Types.COMMON_NOUN,
    "proper_noun": Types.PROPER_NOUN,
    "v_trans_omissible_p1": Types.V_TRANS_OMISSABLE_P1,
    "v_trans_omissible_p2": Types.V_TRANS_OMISSABLE_P2,
    "v_trans_omissible_pp_p1": Types.V_TRANS_OMISSABLE_PP_P1,
    "v_trans_omissible_pp_p2": Types.V_TRANS_OMISSABLE_PP_P2,
    "v_trans_not_omissible": Types.V_TRANS_NOT_OMISSABLE,
    "v_trans_not_omissible_pp_p1": Types.V_TRANS_NOT_OMISSABLE_PP_P1,
    "v_trans_not_omissible_pp_p2": Types.V_TRANS_NOT_OMISSABLE_PP_P2,
    "v_cp_taking": Types.V_CP_TAKING,
    "v_inf_taking": Types.V_INF_TAKING,
    "v_unacc_p1": Types.V_UNACC_P1,
    "v_unacc_p2": Types.V_UNACC_P2,
    "v_unacc_pp_p1": Types.V_UNACC_PP_P1,
    "v_unacc_pp_p2": Types.V_UNACC_PP_P2,
    "v_unerg": Types.V_UNERG,
    "v_inf": Types.V_INF,
    "v_dat_p1": Types.V_DAT_P1,
    "v_dat_p2": Types.V_DAT_P2,
    "v_dat_pp_p1": Types.V_DAT_PP_P1,
    "v_dat_pp_p2": Types.V_DAT_PP_P2,
    "v_dat_pp_p3": Types.V_DAT_PP_P3,
    "v_dat_pp_p4": Types.V_DAT_PP_P4
}


exclude_types = ['start', 'det', 'pp', 'was', 'by', 'to', 'that', 'common_noun', 'proper_noun',
                 'vp_external', 'vp_passive', 'vp_passive_dat', 'np']

five_span_types = ['vp_passive8', 'vp_passive_dat2']

four_span_types = ['vp_passive2', 'vp_passive4', 'vp_passive6']

three_span_types = ['vp_external4', 'vp_external5', 'vp_external6', 'vp_external7', 'vp_passive7',
                    'vp_passive_dat1', 'np_pp']

two_span_types = ['s1', 's2', 's3', 's4', 'vp_external1', 'vp_external2', 'vp_external3', 'vp_internal',
                  'vp_passive1', 'vp_passive3', 'vp_passive5', 'np_det', 'pp_iobj', 'pp_loc']

one_span_types = ['common_noun', 'proper_noun', 'v_trans_omissible_p1', 'v_trans_omissible_p2',
                  'v_trans_omissible_pp_p1', 'v_trans_omissible_pp_p2', 'v_trans_not_omissible',
                  'v_trans_not_omissible_pp_p1', 'v_trans_not_omissible_pp_p2', 'v_cp_taking',
                  'v_inf_taking', 'v_unacc_p1', 'v_unacc_p2', 'v_unacc_pp_p1', 'v_unacc_pp_p2',
                  'v_unerg', 'v_inf', 'v_dat_p1', 'v_dat_p2', 'v_dat_pp_p1', 'v_dat_pp_p2',
                  'v_dat_pp_p3', 'v_dat_pp_p4', 'det', 'pp', 'was', 'by', 'to', 'that', 'np_prop',
                  'vp_passive_dat', 'vp_passive', 'vp_external', 'start']
