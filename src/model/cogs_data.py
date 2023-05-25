import csv
from enum import IntEnum
from lark import Lark


def load_COGS(train_fp, test_fp):
    train_data = load_COGS_file(train_fp)
    test_data = load_COGS_file(test_fp)
    return train_data, test_data


def load_COGS_file(filepath):
    with open(filepath, "rt") as file:
        tsv_file = csv.reader(file, delimiter="\t")
        data = []
        for line in tsv_file:
            x = line[:2][0].replace(".", "").lower().split()
            # remove x _ ; AND
            y = line[:2][1].replace(' x ', '').replace('_', '').replace('AND', '').replace(';', '').lower().split()
            data.append([x, y])
    return data


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
    'Audrey', 'Ryan', 'Brooklyn', 'Asher', 'Bella', 'Nathan', 'Claire', 'Thomas', 'Skylar', 'Leo', 'Lina',
    'Paula', 'Charlie'
]
# convert to lower case
proper_nouns = list(map(lambda n: n.lower(), proper_nouns))

# P + N: N from BNC + COCA

# 100 nouns that can appear with "on"
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

# Levin, '1.2.1 Unspecified Object Alternation'
# And some intuition-based selection.
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

# Levin 29.4 Declare verbs, Levin 30. Verbs of Perception, VerbNet admire-31.2, VerbNet wish-62
V_cp_taking = [
    'liked', 'hoped', 'said', 'noticed', 'believed',
    'confessed', 'declared', 'proved', 'thought', 'admired',
    'appreciated', 'respected', 'supported', 'tolerated', 'valued',
    'wished', 'dreamed', 'expected', 'imagined', 'meant'
]

# VerbNet want-32.1, VerbNet try-61, VerbNet wish-62, VerbNet long-32.2, VerbNet admire-31.2-1
V_inf_taking = [
    'wanted', 'preferred', 'needed', 'intended', 'tried',
    'attempted', 'planned', 'expected', 'hoped', 'wished',
    'craved', 'liked', 'hated', 'loved', 'enjoyed',
    'dreamed', 'meant', 'longed', 'yearned', 'itched'
]

# 1.1.2.1 Causative-Inchoative Alternation
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

# 10 DO omissible transitives, 10 unergatives
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

def quote(l):
    return map(lambda x: "\"" + x + "\"", l)

cogs_grammar = """
    start: s1 | s2 | s3 | s4 | vp_internal
    s1: np_animate_nsubj vp_external
    s2: np_inanimate_nsubjpass vp_passive
    s3: np_animate_nsubjpass vp_passive_dat
    s4: np_animate_nsubj vp_external4
    vp_external: v_unerg | v_trans_omissible_p1 | vp_external1 | vp_external2 | vp_external3 | vp_external5 | vp_external6 | vp_external7
    vp_external1: v_unacc_p1 np_dobj
    vp_external2: v_trans_omissible_p2 np_dobj
    vp_external3: v_trans_not_omissible np_dobj
    vp_external4: v_inf_taking inf v_inf
    vp_external5: v_cp_taking c start
    vp_external6: v_dat_p1 np_inanimate_dobj pp_iobj
    vp_external7: v_dat_p2 np_animate_iobj np_inanimate_dobj
    vp_internal: np_unacc_subj v_unacc_p2
    vp_passive: vp_passive1 | vp_passive2 | vp_passive3 | vp_passive4 | vp_passive5 | vp_passive6 | vp_passive7 | vp_passive8
    vp_passive1: aux v_trans_not_omissible_pp_p1
    vp_passive2: aux v_trans_not_omissible_pp_p2 by np_animate_nsubj
    vp_passive3: aux v_trans_omissible_pp_p1
    vp_passive4: aux v_trans_omissible_pp_p2 by np_animate_nsubj
    vp_passive5: aux v_unacc_pp_p1
    vp_passive6: aux v_unacc_pp_p2 by np_animate_nsubj
    vp_passive7: aux v_dat_pp_p1 pp_iobj
    vp_passive8: aux v_dat_pp_p2 pp_iobj by np_animate_nsubj
    vp_passive_dat: vp_passive_dat1 | vp_passive_dat2
    vp_passive_dat1: aux v_dat_pp_p3 np_inanimate_dobj
    vp_passive_dat2: aux v_dat_pp_p4 np_inanimate_dobj by np_animate_nsubj
    np_dobj: np_inanimate_dobj | np_animate_dobj
    np_unacc_subj: np_inanimate_dobj_nopp | np_animate_dobj_nopp
    np_animate_dobj_nopp: np_animate_dobj_nopp1 | n_prop_dobj
    np_animate_dobj_nopp1: det n_common_animate_dobj
    np_animate_dobj: np_animate_dobj1 | np_animate_dobj2 | n_prop_dobj
    np_animate_dobj1: det n_common_animate_dobj
    np_animate_dobj2: det n_common_animate_dobj pp_loc            
    np_animate_iobj: np_animate_iobj1 | n_prop_iobj
    np_animate_iobj1: det n_common_animate_iobj
    np_animate_nsubj: np_animate_nsubj1 | n_prop_nsubj
    np_animate_nsubj1: det n_common_animate_nsubj
    np_animate_nsubjpass: np_animate_nsubjpass1 | n_prop_nsubjpass
    np_animate_nsubjpass1: det n_common_animate_nsubjpass
    np_inanimate_dobj: np_inanimate_dobj1 | np_inanimate_dobj2
    np_inanimate_dobj1: det n_common_inanimate_dobj
    np_inanimate_dobj2: det n_common_inanimate_dobj pp_loc
    np_inanimate_dobj_nopp: det n_common_inanimate_dobj
    np_inanimate_nsubjpass: det n_common_inanimate_nsubjpass
    np_on: np_on1 | np_on2
    np_on1: det n_on pp_loc
    np_on2: det n_on
    np_in: np_in1 | np_in2
    np_in1: det n_in pp_loc
    np_in2: det n_in
    np_beside: np_bedside1 | np_bedside2
    np_bedside1: det n_beside pp_loc
    np_bedside2: det n_beside
    det: \"the\" | \"a\"
    c: \"that\"
    aux: \"was\"
    by: \"by\"
    n_common_animate_dobj: {animate_nouns_str}
    n_common_animate_iobj: {animate_nouns_str}
    n_common_animate_nsubj: {animate_nouns_str}
    n_common_animate_nsubjpass: {animate_nouns_str}
    n_common_inanimate_dobj: {inanimate_nouns_str}
    n_common_inanimate_nsubjpass: {inanimate_nouns_str}
    n_prop_dobj: {proper_nouns_str}
    n_prop_iobj: {proper_nouns_str}
    n_prop_nsubj: {proper_nouns_str}
    n_prop_nsubjpass: {proper_nouns_str}
    n_on: {on_nouns_str}
    n_in: {in_nouns_str}
    n_beside: {beside_nouns_str}
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
    pp_iobj: piobj np_animate_iobj
    pp_loc: pp_loc1 | pp_loc2 | pp_loc3
    pp_loc1: pon np_on
    pp_loc2: pin np_in
    pp_loc3: pbedside np_beside
    piobj: \"to\"
    pon: \"on\"
    pin: \"in\"
    pbedside: \"beside\"
    inf: \"to\"
    
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


class Cogs_Types(IntEnum):
    PAD = 0
    START = 1
    S1 = 2
    S2 = 3
    S3 = 4
    VP_EXTERNAL = 5
    VP_EXTERNAL1 = 6
    VP_EXTERNAL2 = 7
    VP_EXTERNAL3 = 8
    VP_EXTERNAL4 = 9
    VP_EXTERNAL5 = 10
    VP_EXTERNAL6 = 11
    VP_EXTERNAL7 = 12
    VP_INTERNAL = 13
    VP_PASSIVE = 14
    VP_PASSIVE1 = 15
    VP_PASSIVE2 = 16
    VP_PASSIVE3 = 17
    VP_PASSIVE4 = 18
    VP_PASSIVE5 = 19
    VP_PASSIVE6 = 20
    VP_PASSIVE7 = 21
    VP_PASSIVE8 = 22
    VP_PASSIVE_DAT = 23
    VP_PASSIVE_DAT1 = 24
    VP_PASSIVE_DAT2 = 25
    NP_DOBJ = 26
    NP_UNACC_SUBJ = 27
    NP_ANIMATE_DOBJ_NOPP = 28
    NP_ANIMATE_DOBJ_NOPP1 = 29
    NP_ANIMATE_DOBJ = 30
    NP_ANIMATE_DOBJ1 = 31
    NP_ANIMATE_DOBJ2 = 32
    NP_ANIMATE_IOBJ = 33
    NP_ANIMATE_IOBJ1 = 34
    NP_ANIMATE_NSUBJ = 35
    NP_ANIMATE_NSUBJ1 = 36
    NP_ANIMATE_NSUBJPASS = 37
    NP_ANIMATE_NSUBJPASS1 = 38
    NP_INANIMATE_DOBJ = 39
    NP_INANIMATE_DOBJ1 = 40
    NP_INANIMATE_DOBJ2 = 41
    NP_INANIMATE_DOBJ_NOPP = 42
    NP_INANIMATE_NSUBJPASS = 43
    NP_ON = 44
    NP_ON1 = 45
    NP_ON2 = 46
    NP_IN = 47
    NP_IN1 = 48
    NP_IN2 = 49
    NP_BEDSIDE = 50
    NP_BEDSIDE1 = 51
    NP_BEDSIDE2 = 52
    DET = 53
    C = 54
    AUX = 55
    BY = 56
    N_COMMON_ANIMATE_DOBJ = 57
    N_COMMON_ANIMATE_IOBJ = 58
    N_COMMON_ANIMATE_NSUBJ = 59
    N_COMMON_ANIMATE_NSUBJPASS = 60
    N_COMMON_INANIMATE_DOBJ = 61
    N_COMMON_INANIMATE_NSUBJPASS = 62
    N_PROP_DOBJ = 63
    N_PROP_IOBJ = 64
    N_PROP_NSUBJ = 65
    N_PROP_NSUBJPASS = 66
    N_ON = 67
    N_IN = 68
    N_BEDSIDE = 69
    V_TRANS_OMISSABLE_P1 = 70
    V_TRANS_OMISSABLE_P2 = 71
    V_TRANS_OMISSABLE_PP_P1 = 72
    V_TRANS_OMISSABLE_PP_P2 = 73
    V_TRANS_NOT_OMISSABLE = 74
    V_TRANS_NOT_OMISSABLE_PP_P1 = 75
    V_TRANS_NOT_OMISSABLE_PP_P2 = 76
    V_CP_TAKING = 77
    V_INF_TAKING = 78
    V_UNACC_P1 = 79
    V_UNACC_P2 = 80
    V_UNACC_PP_P1 = 81
    V_UNACC_PP_P2 = 82
    V_UNERG = 83
    V_INF = 84
    V_DAT_P1 = 85
    V_DAT_P2 = 86
    V_DAT_PP_P1 = 87
    V_DAT_PP_P2 = 88
    V_DAT_PP_P3 = 89
    V_DAT_PP_P4 = 90
    PP_IOBJ = 91
    PP_LOC = 92
    PP_LOC1 = 93
    PP_LOC2 = 94
    PP_LOC3 = 95
    P_IOBJ = 96
    P_ON = 97
    P_IN = 98
    P_BEDSIDE = 99
    INF = 100
    S4 = 101


cogs_token_to_type = {
    "start": Cogs_Types.START,
    "s1": Cogs_Types.S1,
    "s2": Cogs_Types.S2,
    "s3": Cogs_Types.S3,
    "s4": Cogs_Types.S4,
    "vp_external": Cogs_Types.VP_EXTERNAL,
    "vp_external1": Cogs_Types.VP_EXTERNAL1,
    "vp_external2": Cogs_Types.VP_EXTERNAL2,
    "vp_external3": Cogs_Types.VP_EXTERNAL3,
    "vp_external4": Cogs_Types.VP_EXTERNAL4,
    "vp_external5": Cogs_Types.VP_EXTERNAL5,
    "vp_external6": Cogs_Types.VP_EXTERNAL6,
    "vp_external7": Cogs_Types.VP_EXTERNAL7,
    "vp_internal": Cogs_Types.VP_INTERNAL,
    "vp_passive": Cogs_Types.VP_PASSIVE,
    "vp_passive1": Cogs_Types.VP_PASSIVE1,
    "vp_passive2": Cogs_Types.VP_PASSIVE2,
    "vp_passive3": Cogs_Types.VP_PASSIVE3,
    "vp_passive4": Cogs_Types.VP_PASSIVE4,
    "vp_passive5": Cogs_Types.VP_PASSIVE5,
    "vp_passive6": Cogs_Types.VP_PASSIVE6,
    "vp_passive7": Cogs_Types.VP_PASSIVE7,
    "vp_passive8": Cogs_Types.VP_PASSIVE8,
    "vp_passive_dat": Cogs_Types.VP_PASSIVE_DAT,
    "vp_passive_dat1": Cogs_Types.VP_PASSIVE_DAT1,
    "vp_passive_dat2": Cogs_Types.VP_PASSIVE_DAT2,
    "np_dobj": Cogs_Types.NP_DOBJ,
    "np_unacc_subj": Cogs_Types.NP_UNACC_SUBJ,
    "np_animate_dobj_nopp": Cogs_Types.NP_ANIMATE_DOBJ_NOPP,
    "np_animate_dobj_nopp1": Cogs_Types.NP_ANIMATE_DOBJ_NOPP1,
    "np_animate_dobj": Cogs_Types.NP_ANIMATE_DOBJ,
    "np_animate_dobj1": Cogs_Types.NP_ANIMATE_DOBJ1,
    "np_animate_dobj2": Cogs_Types.NP_ANIMATE_DOBJ2,
    "np_animate_iobj": Cogs_Types.NP_ANIMATE_IOBJ,
    "np_animate_iobj1": Cogs_Types.NP_ANIMATE_IOBJ1,
    "np_animate_nsubj": Cogs_Types.NP_ANIMATE_NSUBJ,
    "np_animate_nsubj1": Cogs_Types.NP_ANIMATE_NSUBJ1,
    "np_animate_nsubjpass": Cogs_Types.NP_ANIMATE_NSUBJPASS,
    "np_animate_nsubjpass1": Cogs_Types.NP_ANIMATE_NSUBJPASS1,
    "np_inanimate_dobj": Cogs_Types.NP_INANIMATE_DOBJ,
    "np_inanimate_dobj1": Cogs_Types.NP_INANIMATE_DOBJ1,
    "np_inanimate_dobj2": Cogs_Types.NP_INANIMATE_DOBJ2,
    "np_inanimate_dobj_nopp": Cogs_Types.NP_INANIMATE_DOBJ_NOPP,
    "np_inanimate_nsubjpass": Cogs_Types.NP_INANIMATE_NSUBJPASS,
    "np_on": Cogs_Types.NP_ON,
    "np_on1": Cogs_Types.NP_ON1,
    "np_on2": Cogs_Types.NP_ON2,
    "np_in": Cogs_Types.NP_IN,
    "np_in1": Cogs_Types.NP_IN1,
    "np_in2": Cogs_Types.NP_IN2,
    "np_beside": Cogs_Types.NP_BEDSIDE,
    "np_bedside1": Cogs_Types.NP_BEDSIDE1,
    "np_bedside2": Cogs_Types.NP_BEDSIDE2,
    "det": Cogs_Types.DET,
    "c": Cogs_Types.C,
    "aux": Cogs_Types.AUX,
    "by": Cogs_Types.BY,
    "n_common_animate_dobj": Cogs_Types.N_COMMON_ANIMATE_DOBJ,
    "n_common_animate_iobj": Cogs_Types.N_COMMON_ANIMATE_IOBJ,
    "n_common_animate_nsubj": Cogs_Types.N_COMMON_ANIMATE_NSUBJ,
    "n_common_animate_nsubjpass": Cogs_Types.N_COMMON_ANIMATE_NSUBJPASS,
    "n_common_inanimate_dobj": Cogs_Types.N_COMMON_INANIMATE_DOBJ,
    "n_common_inanimate_nsubjpass": Cogs_Types.N_COMMON_INANIMATE_NSUBJPASS,
    "n_prop_dobj": Cogs_Types.N_PROP_DOBJ,
    "n_prop_iobj": Cogs_Types.N_PROP_IOBJ,
    "n_prop_nsubj": Cogs_Types.N_PROP_NSUBJ,
    "n_prop_nsubjpass": Cogs_Types.N_PROP_NSUBJPASS,
    "n_on": Cogs_Types.N_ON,
    "n_in": Cogs_Types.N_IN,
    "n_beside": Cogs_Types.N_BEDSIDE,
    "v_trans_omissible_p1": Cogs_Types.V_TRANS_OMISSABLE_P1,
    "v_trans_omissible_p2": Cogs_Types.V_TRANS_OMISSABLE_P2,
    "v_trans_omissible_pp_p1": Cogs_Types.V_TRANS_OMISSABLE_PP_P1,
    "v_trans_omissible_pp_p2": Cogs_Types.V_TRANS_OMISSABLE_PP_P2,
    "v_trans_not_omissible": Cogs_Types.V_TRANS_NOT_OMISSABLE,
    "v_trans_not_omissible_pp_p1": Cogs_Types.V_TRANS_NOT_OMISSABLE_PP_P1,
    "v_trans_not_omissible_pp_p2": Cogs_Types.V_TRANS_NOT_OMISSABLE_PP_P2,
    "v_cp_taking": Cogs_Types.V_CP_TAKING,
    "v_inf_taking": Cogs_Types.V_INF_TAKING,
    "v_unacc_p1": Cogs_Types.V_UNACC_P1,
    "v_unacc_p2": Cogs_Types.V_UNACC_P2,
    "v_unacc_pp_p1": Cogs_Types.V_UNACC_PP_P1,
    "v_unacc_pp_p2": Cogs_Types.V_UNACC_PP_P2,
    "v_unerg": Cogs_Types.V_UNERG,
    "v_inf": Cogs_Types.V_INF,
    "v_dat_p1": Cogs_Types.V_DAT_P1,
    "v_dat_p2": Cogs_Types.V_DAT_P2,
    "v_dat_pp_p1": Cogs_Types.V_DAT_PP_P1,
    "v_dat_pp_p2": Cogs_Types.V_DAT_PP_P2,
    "v_dat_pp_p3": Cogs_Types.V_DAT_PP_P3,
    "v_dat_pp_p4": Cogs_Types.V_DAT_PP_P4,
    "pp_iobj": Cogs_Types.PP_IOBJ,
    "pp_loc": Cogs_Types.PP_LOC,
    "pp_loc1": Cogs_Types.PP_LOC1,
    "pp_loc2": Cogs_Types.PP_LOC2,
    "pp_loc3": Cogs_Types.PP_LOC3,
    "piobj": Cogs_Types.P_IOBJ,
    "pon": Cogs_Types.P_ON,
    "pin": Cogs_Types.P_IN,
    "pbedside": Cogs_Types.P_BEDSIDE,
    "inf": Cogs_Types.INF,
    "np_animate_dobj_nopp1": Cogs_Types.NP_ANIMATE_DOBJ_NOPP1
}

exclude_types = ["piobj", "pon", "pin", "pbedside", "det", "c", "aux", "by", "inf",
                 "n_common_animate_dobj", "n_common_animate_iobj", "n_common_animate_nsubj",
                 "n_common_animate_nsubjpass", "n_common_inanimate_dobj", "n_common_inanimate_nsubjpass",
                 "n_prop_dobj", "n_prop_iobj", "n_prop_nsubj", "n_prop_nsubjpass", "n_on", "n_in",
                 "n_beside", "start", "vp_external", "vp_passive", "vp_passive_dat", "np_dobj", "np_unacc_subj",
                 "np_animate_dobj_nopp", "np_animate_dobj", "np_animate_iobj", "np_animate_nsubj",
                 "np_animate_nsubjpass", "np_inanimate_dobj", "np_on", "np_in", "np_beside", "pp_loc"]

five_span_types = ['vp_passive8', 'vp_passive_dat2']

four_span_types = ['vp_passive2', 'vp_passive4', 'vp_passive6']

three_span_types = ['vp_external4', 'vp_external5', 'vp_external6', 'vp_external7', 'vp_passive7', \
                    'vp_passive_dat1', 'np_animate_dobj2', 'np_inanimate_dobj2', 'np_on1', 'np_in1', 'np_bedside1']

two_span_types = ['s1', 's2', 's3', 's4', 'vp_external1', 'vp_external2', 'vp_external3', 'vp_internal', \
                  'vp_passive1', 'vp_passive3', 'vp_passive5', 'np_animate_dobj1', 'np_animate_iobj1', \
                  'np_animate_nsubj1', 'np_inanimate_dobj1', 'np_inanimate_dobj_nopp', 'np_inanimate_nsubjpass', \
                  'np_on2', 'np_in2', 'np_bedside2', 'pp_iobj', 'pp_loc1', 'pp_loc2', 'pp_loc3', \
                  'np_animate_nsubjpass1', 'np_animate_dobj_nopp1']

one_span_types = ['n_common_animate_dobj', 'n_common_animate_iobj', 'n_common_animate_nsubj', \
                  'n_common_animate_nsubjpass', 'n_common_inanimate_dobj', 'n_common_inanimate_nsubjpass', \
                  'n_prop_dobj', 'n_prop_iobj', 'n_prop_nsubj', 'n_prop_nsubjpass', 'n_on', 'n_in', 'n_beside', \
                  'v_trans_omissible_p1', 'v_trans_omissible_p2', 'v_trans_omissible_pp_p1', 'v_trans_omissible_pp_p2', \
                  'v_trans_not_omissible', 'v_trans_not_omissible_pp_p1', 'v_trans_not_omissible_pp_p2', \
                  'v_cp_taking', 'v_inf_taking', 'v_unacc_p1', 'v_unacc_p2', 'v_unacc_pp_p1', 'v_unacc_pp_p2', \
                  'v_unerg', 'v_inf', 'v_dat_p1', 'v_dat_p2', 'v_dat_pp_p1', 'v_dat_pp_p2', 'v_dat_pp_p3', 'v_dat_pp_p4', \
                  'det', 'c', 'aux', 'by', 'piobj', 'pon', 'pin', 'pbedside', 'inf', 'pp_loc', 'np_beside', 'np_in', \
                  'np_on', 'np_inanimate_dobj', 'np_animate_nsubjpass', 'np_animate_nsubj', 'np_animate_iobj', \
                  'np_animate_dobj', 'np_unacc_subj', 'np_dobj', 'vp_passive_dat', 'vp_passive', 'vp_external', \
                  'np_animate_dobj_nopp', 'det', 'start']

def parse_cogs(parser, cogs_command):
    # create parse tree
    parse_tree = parser.parse(cogs_command)
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
        types.append(cogs_token_to_type[node.data.value])
    return positions, types, spans


# initial decoding and variables
init_dec_token_cogs = {'the' : '* y ( y )'} |\
                      {'a' : 'y ( y )'} |\
                      {'on' : 'y . nmod . on ( y , y )'} |\
                      {'in' : 'y . nmod . in ( y , y )'} |\
                      {'beside' : 'y . nmod . beside ( y , y )'}

init_dec_vtype_cogs = {70: 'y . agent ( y , y )',
                       71: 'y . agent ( y , y ) y . theme ( y , y )',
                       72: 'y . theme ( y , y )',
                       73: 'y . theme ( y , y ) y . agent ( y , y )',
                       74 : 'y . agent ( y , y ) y . theme ( y , y )',
                       75: 'y . theme ( y , y )',
                       76: 'y . theme ( y , y ) y . agent ( y , y )',
                       77: 'y . agent ( y , y ) y . ccomp ( y , y )',
                       78 : 'y . agent ( y , y ) y . xcomp ( y , y )',
                       79: 'y . agent ( y , y ) y . theme ( y , y )',
                       80: 'y . theme ( y , y )',
                       81: 'y . theme ( y , y )',
                       82: 'y . theme ( y , y ) y . agent ( y , y )',
                       83: 'y . agent ( y , y )',
                       84: 'y . agent ( y , y )',
                       85 : 'y . agent ( y , y ) y . theme ( y , y ) y . recipient ( y , y )',
                       86: 'y . agent ( y , y ) y . recipient ( y , y ) y . theme ( y , y )',
                       87 : 'y . theme ( y , y ) y . recipient ( y , y )',
                       88 : 'y . theme ( y , y ) y . recipient ( y , y ) y . agent ( y , y )',
                       89 : 'y . recipient ( y , y ) y . theme ( y , y )',
                       90 : 'y . recipient ( y , y ) y . theme ( y , y ) y . agent ( y , y )'}

initial_variables_cogs = {noun : noun for noun in set(noun_list)} | verbs_lemmas | {verb : verb for verb in set(V_inf)}