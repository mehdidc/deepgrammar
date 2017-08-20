from clize import run

from lightjob.cli import load_db

import numpy as np

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform
from torch.optim import Adam
from torch.autograd import Variable

from grammaropt.grammar import Vectorizer
from grammaropt.grammar import as_str
from grammaropt.grammar import NULL_SYMBOL
from grammaropt.rnn import RnnModel
from grammaropt.rnn import RnnAdapter
from grammaropt.rnn import RnnWalker

from deepgrammar.grammar import grammar
from deepgrammar.samplers import random

def acc(pred, true_classes):
    _, pred_classes = pred.max(1)
    acc = (pred_classes == true_classes).float().mean()
    return acc


def _weights_init(m, ih_std=0.08, hh_std=0.08):
    if isinstance(m, nn.LSTM):
        m.weight_ih_l0.data.normal_(0, ih_std)
        m.weight_hh_l0.data.normal_(0, hh_std)
    elif isinstance(m, nn.Linear):
        xavier_uniform(m.weight.data)
        m.bias.data.fill_(0)



def train():
    epochs = 1000
    hidden_size = 128
    emb_size = 128
    resample = True
    gamma = 0.99
    lr = 1e-4
    batch_size = 64
    use_cuda = True

    # data
    db = load_db()
    jobs = db.jobs_with(state='success')
    jobs = list(jobs)
    X = [get_architecture_from_code(j['content']['codes']['classifier']) for j in jobs]
    max_depth = max([j['content']['info'].get('depth', 0) for j in jobs])
    Y = [max(j['stats']['valid']['accuracy']) for j in jobs]
    if resample:
        X, Y = _resample(X, Y, nb=10)
 
    vect = Vectorizer(grammar, pad=True)
    X = vect.transform(X)
    X = [[0] + x for x in X]
    X = np.array(X).astype('int32')
    print(X.shape)

    print('Number of training data : {}'.format(len(X)))

    # model
    vocab_size = len(vect.tok_to_id)
    
    model = RnnModel(vocab_size=vocab_size, emb_size=emb_size, hidden_size=hidden_size, use_cuda=use_cuda)
    model.vect = vect
    model.apply(_weights_init)
    if use_cuda:
        model = model.cuda()

    optim = Adam(model.parameters(), lr=lr)
    adp = RnnAdapter(model, tok_to_id=vect.tok_to_id, begin_tok=NULL_SYMBOL)
    wl = RnnWalker(grammar, adp, temperature=1.0, min_depth=1, max_depth=max_depth)

    # Training
    I = X[:, 0:-1]
    O = X[:, 1:]
    crit = nn.CrossEntropyLoss()
    avg_loss = 0.
    avg_precision = 0.
    nupdates = 0
    for i in range(epochs):
        for j in range(0, len(I), batch_size):
            inp = I[j:j+batch_size]
            out = O[j:j+batch_size]
            out = out.flatten()
            inp = torch.from_numpy(inp).long()
            inp = Variable(inp)
            out = torch.from_numpy(out).long()
            out = Variable(out)
            if use_cuda:
                inp = inp.cuda()
                out = out.cuda()
            
            model.zero_grad()
            y = model(inp)
            loss = crit(y, out)
            precision = acc(y, out)
            loss.backward()
            optim.step()

            avg_loss = avg_loss * gamma + loss.data[0] * (1 - gamma)
            avg_precision = avg_precision * gamma + precision.data[0] * (1 - gamma)
            if nupdates % 100 == 0:
                print('Epoch : {:05d} Avg loss : {:.6f} Avg Precision : {:.6f}'.format(i, avg_loss, avg_precision))
                print('Generated :')
                wl.walk()
                expr = as_str(wl.terminals)
                print(expr)
            nupdates += 1
    

def compute_acc(wl):
    p_list = []
    for dc in wl._decisions:
        if dc.action == 'rule':
            p = torch.exp(wl.rnn.token_logp(dc.gen, dc.pred))
            p_list.append(p.data[0])
    return np.mean(p_list)


def _resample(X, Y, nb=1):
    p = np.array(Y)
    p /= p.sum()
    X = np.random.choice(X, size=nb * len(X), p=p, replace=True)
    Y = np.ones(len(X))
    return X, Y

def sample():
    random()

if __name__ == '__main__':
    run([train, sample])
