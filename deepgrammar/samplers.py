import os

import numpy as np

from grammaropt.grammar import as_str
from grammaropt.random import RandomWalker

from .grammar import grammar


classifier_tpl = open(os.path.join(os.path.dirname(__file__), 'classifier_tpl.py')).read()


def random():
    rng = np.random
    random_state = rng.randint(1, 2**32)
    rng = np.random.RandomState(random_state)
    depth = rng.randint(5, 12)
    min_depth = depth
    max_depth = depth
    wl = RandomWalker(grammar, min_depth=min_depth, max_depth=max_depth, random_state=random_state)
    wl.walk()
    architecture = as_str(wl.terminals)
    code = format_code(architecture)
    print(code)
    out = {
        'codes': {
            'classifier': code
        },
        'info': {
            'depth': depth,
            'random_state': random_state,
            'architecture': architecture,
        }
    }
    return out

nb_spaces_indent = 8

def format_code(architecture):
    architecture = _indent(architecture, nb_spaces_indent)
    code = classifier_tpl.format(architecture=architecture)
    return code

def get_architecture_from_code(code):
    lines = code.split('\n')
    first, last = None, None
    for i, line in enumerate(lines):
        if 'activation = ' in line:
            first = i
        if 'opt = ' in line:
            last = i
    assert first and last
    lines = lines[first:last + 1]
    lines = [l[nb_spaces_indent:] for l in lines]
    return '\n'.join(lines) + '\n'


def _indent(s, nb_spaces):
    return '\n'.join([" " * 8 + line for line in s.split('\n')])
