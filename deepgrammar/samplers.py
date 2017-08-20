import os

import numpy as np
import torch


from grammaropt.grammar import as_str, extract_rules_from_grammar, rule_depth as _rule_depth
from grammaropt.random import RandomWalker
from grammaropt.rnn import RnnAdapter
from grammaropt.rnn import RnnWalker

from .grammar import grammar
from .utils import get_tok_to_id


nb_spaces_indent = 8
classifier_tpl = open(os.path.join(os.path.dirname(__file__), 'classifier_tpl.py')).read()

def rule_depth(rule):
    depth = _rule_depth(rule)
    if 'poolingop' in str(rule) or 'dropout' in str(rule):
        return 0
    return depth


class ControlledRandomWalker(RandomWalker):

    def _filter_by_depth(self, rules, depth):
        if self.min_depth is not None and depth <= self.min_depth:
            if self.strict_depth_limit:
                return []
            depths = list(map(rule_depth, rules))
            max_depth = max(depths)
            rules = [r for r, d in zip(rules, depths) if d == max_depth]
            return rules
        elif self.max_depth is not None and depth >= self.max_depth:
            if self.strict_depth_limit:
                return []
            depths = list(map(rule_depth, rules))
            min_depth = min(depths)
            rules = [r for r, d in zip(rules, depths) if d == min_depth]
            return rules
        else:
            return rules


def rnn():
    rng = np.random
    random_state = rng.randint(1, 2**32)
    rng = np.random.RandomState(random_state)
    model = torch.load('rnn.th', map_location=lambda storage, loc: storage)
    model.use_cuda = False
    rules = extract_rules_from_grammar(grammar)
    tok_to_id = get_tok_to_id(rules)
    rnn = RnnAdapter(model, tok_to_id, random_state=random_state)
    wl = RnnWalker(grammar=grammar, rnn=rnn)
    return _gen_from_walker(wl, random_state)


def random():
    rng = np.random
    random_state = rng.randint(1, 2**32)
    rng = np.random.RandomState(random_state)
    depth = rng.randint(5, 30)
    min_depth = depth
    max_depth = depth
    wl = ControlledRandomWalker(
        grammar, 
        min_depth=min_depth, 
        max_depth=max_depth, 
        random_state=random_state,
    )
    return _gen_from_walker(wl, random_state)


def _gen_from_walker(wl, random_state):
    wl.walk()
    architecture = as_str(wl.terminals)
    code = format_code(architecture)
    print(code)
    out = {
        'codes': {
            'classifier': code
        },
        'info': {
            'random_state': random_state,
            'architecture': architecture,
            'min_depth': wl.max_depth,
            'max_depth': wl.max_depth,
        }
    }
    return out
 

def format_code(architecture):
    architecture = _indent(architecture, nb_spaces_indent)
    code = classifier_tpl.format(architecture=architecture)
    return code

def _indent(s, nb_spaces):
    return '\n'.join([" " * 8 + line for line in s.split('\n')])
