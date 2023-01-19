try:
    input = raw_input   # For Python2 compatibility
except NameError:
    pass

import turtle

from lark import Lark

scan_grammar = """
    start: c
    c: f e | g s | s
    f: s AND
    g: s AFTER
    s: v TWICE | v THRICE | v
    e: a m | t m
    v: e d
    a: WALK | LOOK | RUN | JUMP
    t: TURN
    m: OPPOSITE | AROUND
    d: LEFT | RIGHT
    
    AND: LETTER+
    AFTER: LETTER+
    TWICE: LETTER+
    THRICE: LETTER+
    WALK: LETTER+
    LOOK: LETTER+
    RUN: LETTER+
    JUMP: LETTER+
    TURN: LETTER+
    OPPOSITE: LETTER+
    AROUND: LETTER+
    LEFT: LETTER+
    RIGHT: LETTER+

    %import common.LETTER
    %import common.WS
    %ignore WS
"""

parser = Lark(scan_grammar)

def run_scan_parse(program):
    parse_tree = parser.parse(program)
    for inst in parse_tree.children:
        print(inst)

def main():
    while True:
        scan_command = input('> ')
        try:
            run_scan_parse(scan_command)
        except Exception as e:
            print(e)

if __name__ == '__main__':
    main()