import os

import numpy as np

from grammaropt.grammar import as_str
from grammaropt.random import RandomWalker

from .grammar import grammar


classifier_tpl = open(os.path.join(os.path.dirname(__file__), 'classifier_tpl.py')).read()


def random():
    rng = np.random
    random_state = rng.randint(1, 2**32)
    depth = 5
    min_depth = depth
    max_depth = depth
    wl = RandomWalker(grammar, min_depth=min_depth, max_depth=max_depth, random_state=random_state)
    wl.walk()
    architecture = as_str(wl.terminals)
    architecture = _indent(architecture, 8)
    code = classifier_tpl.format(architecture=architecture)
    print(code)
    out = {
        'codes': {
            'classifier': code
        },
        'info': {
            'depth': depth,
            'random_state': random_state
        }
    }
    return out


def _indent(s, nb_spaces):
    return '\n'.join([" " * 8 + line for line in s.split('\n')])
