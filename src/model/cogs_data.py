from enum import IntEnum

import torch
import numpy as np
import re
import urllib.request
import os
from dataclasses import dataclass, field
from typing import Dict
from lark import Lark
from src.model.data import download_file

SCAN_LENGTH_TRAIN_URL = "https://raw.githubusercontent.com/brendenlake/SCAN/master/length_split/tasks_train_length.txt"
SCAN_LENGTH_TEST_URL = "https://raw.githubusercontent.com/brendenlake/SCAN/master/length_split/tasks_test_length.txt"
SCAN_LENGTH_TRAIN_FILEPATH = "./src/model/scan_data/SCAN_length_train.txt"
SCAN_LENGTH_TEST_FILEPATH = "./src/model/scan_data/SCAN_length_test.txt"





def load_SCAN(train_fp, test_fp):
    train_data = load_SCAN_file(train_fp)
    test_data = load_SCAN_file(test_fp)
    return train_data, test_data


def load_SCAN_file(filepath):
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


def load_SCAN_length():
    if not os.path.exists(SCAN_LENGTH_TRAIN_FILEPATH):
        download_file(SCAN_LENGTH_TRAIN_URL, SCAN_LENGTH_TRAIN_FILEPATH, verbose=True)
    if not os.path.exists(SCAN_LENGTH_TEST_FILEPATH):
        download_file(SCAN_LENGTH_TEST_URL, SCAN_LENGTH_TEST_FILEPATH, verbose=True)
    return load_SCAN(SCAN_LENGTH_TRAIN_FILEPATH, SCAN_LENGTH_TEST_FILEPATH)

class CogsTypes(IntEnum):
    PAD = 25


scan_token_to_type = {
    "twice": ScanTypes.P,
    "<PAD>": ScanTypes.PAD
}

# 100 nouns, picked from the MacArthur Communicative Development Inventory and the BNC top frequent nouns
# BNC freq rank: http://ucrel.lancs.ac.uk/bncfreq/flists.html
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
    'guy', 'dealer', 'duke', 'tourist', 'landlord', 'human', 'host', 'priest', 'journalist', 'poet'
]
assert len(set(animate_nouns)) == 100

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

assert len(set(inanimate_nouns)) == 100

# 100 names, picked from https://www.ssa.gov/OACT/babynames/
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
    'Audrey', 'Ryan', 'Brooklyn', 'Asher', 'Bella', 'Nathan', 'Claire', 'Thomas', 'Skylar', 'Leo'
]

assert len(set(proper_nouns)) == 100

# P + N: N from BNC + COCA

# 100 nouns that can appear with "on"
on_nouns = [
    'table', 'stage', 'bed', 'chair', 'stool', 'road', 'tree', 'box', 'surface', 'seat',
    'speaker', 'computer', 'rock', 'boat', 'cabinet', 'TV', 'plate', 'desk', 'bowl', 'bench',
    'shelf', 'cloth', 'piano', 'bible', 'leaflet', 'sheet', 'cupboard', 'truck', 'tray', 'notebook',
    'blanket', 'deck', 'coffin', 'log', 'ladder', 'barrel', 'rug', 'canvas', 'tiger', 'towel',
    'throne', 'booklet', 'sock', 'corpse', 'sofa', 'keyboard', 'book', 'pillow', 'pad', 'train',
    'couch', 'bike', 'pedestal', 'platter', 'paper', 'rack', 'board', 'panel', 'tripod', 'branch',
    'machine', 'floor', 'napkin', 'cookie', 'block', 'cot', 'device', 'yacht', 'dog', 'mattress',
    'ball', 'stand', 'stack', 'windowsill', 'counter', 'cushion', 'hanger', 'trampoline', 'gravel', 'cake',
    'carpet', 'plaque', 'boulder', 'leaf', 'mound', 'bun', 'dish', 'cat', 'podium', 'tabletop',
    'beach', 'bag', 'glacier', 'brick', 'crack', 'vessel', 'futon', 'turntable', 'rag', 'chessboard'
]

# 100 nouns that can appear with "in"
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

# 100 nouns that can appear with "beside"
beside_nouns = [
    'table', 'stage', 'bed', 'chair', 'book', 'road', 'tree', 'machine', 'house', 'seat',
    'speaker', 'computer', 'rock', 'car', 'box', 'cup', 'glass', 'bag', 'flower', 'boat',
    'vehicle', 'key', 'painting', 'cabinet', 'TV', 'bottle', 'cat', 'desk', 'shoe', 'mirror',
    'clock', 'bench', 'bike', 'lamp', 'lion', 'piano', 'crystal', 'toy', 'duck', 'sword',
    'sculpture', 'rod', 'truck', 'basket', 'bear', 'nest', 'sphere', 'bush', 'surgeon', 'poster',
    'throne', 'giant', 'trophy', 'hedge', 'log', 'tent', 'ladder', 'helicopter', 'barrel', 'yacht',
    'statue', 'bucket', 'skull', 'beast', 'lemon', 'whale', 'cage', 'gardner', 'fox', 'sink',
    'trainee', 'dragon', 'cylinder', 'monk', 'bat', 'headmaster', 'philosopher', 'foreigner', 'worm', 'chemist',
    'corpse', 'wolf', 'torch', 'sailor', 'valve', 'hammer', 'doll', 'genius', 'baron', 'murderer',
    'bicycle', 'keyboard', 'stool', 'pepper', 'warrior', 'pillar', 'monkey', 'cassette', 'broker', 'bin'

]

assert len(set(on_nouns)) == len(set(in_nouns)) == len(set(beside_nouns)) == 100
noun_list = animate_nouns + inanimate_nouns + proper_nouns + on_nouns + in_nouns + beside_nouns
print(len(set(noun_list)))

# Levin, '1.2.1 Unspecified Object Alternation'
# And some intuition-based selection.
V_trans_omissible = [
    'ate', 'painted', 'drew', 'cleaned', 'cooked',
    'dusted', 'hunted', 'nursed', 'sketched', 'juggled',
    'called', 'heard', 'packed', 'saw', 'noticed',
    'studied', 'examined', 'observed', 'knew', 'investigated'
]
V_trans_omissible_pp = [
    'eaten', 'painted', 'drawn', 'cleaned', 'cooked',
    'dusted', 'hunted', 'nursed', 'sketched', 'juggled',
    'called', 'heard', 'packed', 'seen', 'noticed',
    'studied', 'examined', 'observed', 'known', 'investigated'
]

assert len(set(V_trans_omissible)) == len(set(V_trans_omissible_pp)) == 20

# Levin class 30. Verbs of Perception, 31.2 Admire Verbs, VerbNet poke-19, throw-17.1.1
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

assert set(V_trans_omissible).isdisjoint(set(V_trans_not_omissible))
assert set(V_trans_omissible_pp).isdisjoint(set(V_trans_not_omissible_pp))

assert len(set(V_trans_not_omissible)) == len(set(V_trans_not_omissible_pp)) == 20

# Levin 29.4 Declare verbs, Levin 30. Verbs of Perception, VerbNet admire-31.2, VerbNet wish-62
V_cp_taking = [
    'liked', 'hoped', 'said', 'noticed', 'believed',
    'confessed', 'declared', 'proved', 'thought', 'admired',
    'appreciated', 'respected', 'supported', 'tolerated', 'valued',
    'wished', 'dreamed', 'expected', 'imagined', 'meant'
]

assert len(set(V_cp_taking)) == 20

# VerbNet want-32.1, VerbNet try-61, VerbNet wish-62, VerbNet long-32.2, VerbNet admire-31.2-1
V_inf_taking = [
    'wanted', 'preferred', 'needed', 'intended', 'tried',
    'attempted', 'planned', 'expected', 'hoped', 'wished',
    'craved', 'liked', 'hated', 'loved', 'enjoyed',
    'dreamed', 'meant', 'longed', 'yearned', 'itched'
]
assert len(set(V_inf_taking)) == 20

# 1.1.2.1 Causative-Inchoative Alternation
V_unacc = [
    'rolled', 'froze', 'burned', 'shortened', 'floated',
    'grew', 'slid', 'broke', 'crumpled', 'split',
    'changed', 'snapped', 'disintegrated', 'collapsed', 'decomposed',
    'doubled', 'improved', 'inflated', 'enlarged', 'reddened',
]
V_unacc_pp = [
    'rolled', 'frozen', 'burned', 'shortened', 'floated',
    'grown', 'slid', 'broken', 'crumpled', 'split',
    'changed', 'snapped', 'disintegrated', 'collapsed', 'decomposed',
    'doubled', 'improved', 'inflated', 'enlarged', 'reddened'
]
assert len(set(V_unacc)) == len(set(V_unacc_pp)) == 20

V_unerg = [
    'slept', 'smiled', 'laughed', 'sneezed', 'cried',
    'talked', 'danced', 'jogged', 'walked', 'ran',
    'napped', 'snoozed', 'screamed', 'stuttered', 'frowned',
    'giggled', 'scoffed', 'snored', 'smirked', 'gasped'
]
assert len(set(V_unerg)) == 20

# 10 DO omissible transitives, 10 unergatives
V_inf = [
    'walk', 'run', 'sleep', 'sneeze', 'nap',
    'eat', 'read', 'cook', 'hunt', 'paint',
    'talk', 'dance', 'giggle', 'jog', 'smirk',
    'call', 'sketch', 'dust', 'clean', 'investigate'
]
assert len(set(V_inf)) == 20

V_dat = [
    'gave', 'lended', 'sold', 'offered', 'fed',
    'passed', 'sent', 'rented', 'served', 'awarded',
    'brought', 'handed', 'forwarded', 'promised', 'mailed',
    'loaned', 'posted', 'returned', 'slipped', 'wired'
]
V_dat_pp = [
    'given', 'lended', 'sold', 'offered', 'fed',
    'passed', 'sent', 'rented', 'served', 'awarded',
    'brought', 'handed', 'forwarded', 'promised', 'mailed',
    'loaned', 'posted', 'returned', 'slipped', 'wired'
]

assert len(set(V_dat)) == len(set(V_dat_pp)) == 20

print(len(set(V_trans_omissible + V_trans_not_omissible + V_cp_taking + V_unacc + V_unerg + V_dat)))

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

pos_d = {
    'a': 'DET',
    'the': 'DET',
    'to': 'ADP',
    'on': 'ADP',
    'in': 'ADP',
    'beside': 'ADP',
    'that': 'SCONJ',
    'was': 'AUX',
    'by': 'ADP'
}

# held out vocab items for gen set
only_seen_as_subject = 'hedgehog'
only_seen_as_noun_prim = 'shark'
only_seen_as_object = 'cockroach'
only_seen_as_subject_proper_noun = 'Lina'
only_seen_as_proper_noun_prim = 'Paula'
only_seen_as_object_proper_noun = 'Charlie'
only_seen_as_transitive_obj_omissible = 'baked'
only_seen_as_unaccuative = 'shattered'
only_seen_as_verb_prim = 'crawl'
only_seen_as_transitive_subject_animate = 'cobra'
only_seen_as_unaccusative_subject_animate = 'hippo'
only_seen_as_active = 'blessed'
only_seen_as_passive = 'squeezed'
only_seen_as_double_object = 'teleported'
only_seen_as_pp = 'shipped'

target_item_nouns = [only_seen_as_subject, only_seen_as_noun_prim, only_seen_as_object,
                     only_seen_as_transitive_subject_animate, only_seen_as_unaccusative_subject_animate]

target_item_props = [only_seen_as_subject_proper_noun, only_seen_as_proper_noun_prim,
                     only_seen_as_object_proper_noun]

pos_d.update({n: 'PROPN' for n in proper_nouns + target_item_props})
pos_d.update({n: 'NOUN' for n in noun_list + target_item_nouns})


cogs_grammar_str = """

S -> NP_animate_nsubj VP_external | VP_internal  \
   | NP_inanimate_nsubjpass VP_passive | NP_animate_nsubjpass VP_passive_dat
   
VP_external -> V_unerg | V_unacc NP_dobj \
             | V_trans_omissible | V_trans_omissible NP_dobj \
             | V_trans_not_omissible NP_dobj | V_inf_taking INF V_inf \
             | V_cp_taking C S \
             | V_dat NP_inanimate_dobj PP_iobj  | V_dat NP_animate_iobj NP_inanimate_dobj
             
VP_internal -> NP_unacc_subj V_unacc

VP_passive -> AUX V_trans_not_omissible_pp | AUX V_trans_not_omissible_pp BY NP_animate_nsubj |  \
              AUX V_trans_omissible_pp | AUX V_trans_omissible_pp BY NP_animate_nsubj  |  \
              AUX V_unacc_pp | AUX V_unacc_pp BY NP_animate_nsubj | \
              AUX V_dat_pp PP_iobj | AUX V_dat_pp PP_iobj BY NP_animate_nsubj
              
VP_passive_dat -> AUX V_dat_pp NP_inanimate_dobj | AUX V_dat_pp NP_inanimate_dobj BY NP_animate_nsubj

NP_dobj -> NP_inanimate_dobj | NP_animate_dobj

NP_unacc_subj -> NP_inanimate_dobj_noPP | NP_animate_dobj_noPP

NP_animate_dobj_noPP -> Det N_common_animate_dobj | N_prop_dobj

NP_animate_dobj -> Det N_common_animate_dobj | Det N_common_animate_dobj PP_loc \
                 | N_prop_dobj
                 
NP_animate_iobj -> Det N_common_animate_iobj | N_prop_iobj

NP_animate_nsubj -> Det N_common_animate_nsubj | N_prop_nsubj

NP_animate_nsubjpass -> Det N_common_animate_nsubjpass | N_prop_nsubjpass

NP_inanimate_dobj -> Det N_common_inanimate_dobj | Det N_common_inanimate_dobj PP_loc

NP_inanimate_dobj_noPP -> Det N_common_inanimate_dobj

NP_inanimate_nsubjpass -> Det N_common_inanimate_nsubjpass

NP_on -> Det N_on PP_loc | Det N_on

NP_in -> Det N_in PP_loc | Det N_in

NP_beside -> Det N_beside PP_loc | Det N_beside

Det -> 'the' | 'a'
C -> 'that'
AUX -> 'was'
BY -> 'by'
N_common_animate_dobj -> {animate_nouns_str}
N_common_animate_iobj -> {animate_nouns_str}
N_common_animate_nsubj -> {animate_nouns_str}
N_common_animate_nsubjpass -> {animate_nouns_str}
N_common_inanimate_dobj -> {inanimate_nouns_str}
N_common_inanimate_nsubjpass -> {inanimate_nouns_str}
N_prop_dobj -> {proper_nouns_str}
N_prop_iobj -> {proper_nouns_str}
N_prop_nsubj -> {proper_nouns_str}
N_prop_nsubjpass -> {proper_nouns_str}
N_on -> {on_nouns_str}
N_in -> {in_nouns_str}
N_beside -> {beside_nouns_str}
V_trans_omissible -> {V_trans_omissible_str}
V_trans_omissible_pp -> {V_trans_omissible_pp_str}
V_trans_not_omissible -> {V_trans_not_omissible_str}
V_trans_not_omissible_pp -> {V_trans_not_omissible_pp_str}
V_cp_taking -> {V_cp_taking_str}
V_inf_taking -> {V_inf_taking_str}
V_unacc -> {V_unacc_str}
V_unacc_pp -> {V_unacc_pp_str}
V_unerg -> {V_unerg_str}
V_inf -> {V_inf_str}
V_dat -> {V_dat_str}
V_dat_pp -> {V_dat_pp_str}

PP_iobj -> P_iobj NP_animate_iobj

PP_loc -> P_on NP_on | P_in NP_in | P_beside NP_beside

P_iobj -> 'to'
P_on -> 'on'
P_in -> 'in'
P_beside -> 'beside'
INF -> 'to'

""".format(animate_nouns_str=animate_nouns,
           inanimate_nouns_str=inanimate_nouns,
           proper_nouns_str=proper_nouns,
           in_nouns_str=in_nouns,
           on_nouns_str=on_nouns,
           beside_nouns_str=beside_nouns,
           V_trans_omissible_str=V_trans_omissible,
           V_trans_omissible_pp_str=V_trans_omissible_pp,
           V_trans_not_omissible_str=V_trans_not_omissible,
           V_trans_not_omissible_pp_str=V_trans_not_omissible_pp,
           V_cp_taking_str=V_cp_taking,
           V_inf_taking_str=V_inf_taking,
           V_unacc_str=V_unacc,
           V_unacc_pp_str=V_unacc_pp,
           V_unerg_str=V_unerg,
           V_inf_str=V_inf,
           V_dat_str=V_dat,
           V_dat_pp_str=V_dat_pp
          )


parser = Lark(cogs_grammar, propagate_positions=True)

def parse_cogs(scan_command):
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