from clize import run

from lightjob.cli import load_db

import numpy as np
from sklearn.utils import shuffle

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform
from torch.optim import Adam
from torch.autograd import Variable

from grammaropt.grammar import Vectorizer
from grammaropt.random import RandomWalker
from grammaropt.rnn import RnnAdapter
from grammaropt.rnn import RnnWalker
from grammaropt.rnn import RnnModel

from deepgrammar.grammar import grammar
from deepgrammar.samplers import random
from deepgrammar.samplers import rnn

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



def fit():
    epochs = 50000
    hidden_size = 128
    emb_size = 128
    resample = False
    gamma = 0.99
    lr = 1e-4
    batch_size = 64
    use_cuda = True
    random_state = 42
    num_layers = 1

    # data
    db = load_db()

    # success
    jobs = db.jobs_with(state='success')
    #jobs = db.all_jobs()
    jobs = list(jobs)
    #jobs = [j for j in jobs if j['content']['info']['max_depth'] == 5]
    X = [j['content']['info']['architecture'] for j in jobs]
    R = [max(j['stats']['valid']['accuracy']) if j['state'] == 'success' else -0.1 for j in jobs]

    #threshold = 0.8
    #X = [x for x, r in zip(X, R) if r > threshold]
    #R = [1 for r in R if r > threshold]

    R = np.array(R)
    vect = Vectorizer(grammar, pad=True)
    X = vect.transform(X)
    X = [[0] + x for x in X]
    X = np.array(X).astype('int32')
    print(X.shape)
    
    X, R = shuffle(X, R, random_state=random_state)
    n_train = int(len(X) * 0.8)

    X_train = X[0:n_train]
    R_train = R[0:n_train]
    X_test = X[n_train:]
    R_test = R[n_train:]

    if resample:
        X_train, R_train = _resample(X_train, R_train, nb=10)
    
    print('Number of training data : {}'.format(len(X_train)))

    # model
    vocab_size = len(vect.tok_to_id)
    
    model = RnnModel(
        vocab_size=vocab_size, 
        emb_size=emb_size, 
        hidden_size=hidden_size, 
        num_layers=num_layers,
        use_cuda=use_cuda,
    )
    model.vect = vect
    model.grammar = grammar
    model.apply(_weights_init)
    if use_cuda:
        model = model.cuda()

    optim = Adam(model.parameters(), lr=lr)

    # Training
    
    I_train = X_train[:, 0:-1]
    O_train = X_train[:, 1:]
    
    I_test = X_test[:, 0:-1]
    O_test = X_test[:, 1:]

    avg_loss = 0.
    avg_precision = 0.
    nupdates = 0
    best_loss = float('inf')
    last_epoch_annealing = 0
    last_epoch_improving = 0
    for i in range(epochs):
        model.train()
        for j in range(0, len(I_train), batch_size):
            inp = I_train[j:j+batch_size]
            out = O_train[j:j+batch_size]
            r = R_train[j:j+batch_size]

            out = out.flatten()
            inp = torch.from_numpy(inp).long()
            inp = Variable(inp)
            out = torch.from_numpy(out).long()
            out = Variable(out)

            r = torch.from_numpy(r).float() 
            r = r.repeat(1, O_train.shape[1])
            r = r.view(-1, 1)
            r = Variable(r)
            r = r.cuda()

            if use_cuda:
                inp = inp.cuda()
                out = out.cuda()
            
            model.zero_grad()
            y = model(inp)

            true = out.data
            pred = y.data
            ind = torch.arange(0, true.size(0)).long().cuda()
            ind = ind[true != 0]

            loss = nn.functional.nll_loss(r[ind] * nn.functional.log_softmax(y[ind]), out[ind])
            precision = acc(pred[ind], true[ind])

            loss.backward()
            optim.step()

            avg_loss = avg_loss * gamma + loss.data[0] * (1 - gamma)
            avg_precision = avg_precision * gamma + precision * (1 - gamma)
            nupdates += 1
        print('Epoch : {:05d}, Train loss : {:.6f}, Train Precision : {:.6f}'.format(i, avg_loss, avg_precision))
        precisions = []
        losses = []
        model.eval()
        for j in range(0, len(I_test), batch_size):
            inp = I_test[j:j+batch_size]
            out = O_test[j:j+batch_size]
            r = R_test[j:j+batch_size]

            out = out.flatten()
            inp = torch.from_numpy(inp).long()
            inp = Variable(inp)
            out = torch.from_numpy(out).long()
            out = Variable(out)

            r = torch.from_numpy(r).float() 
            r = r.repeat(1, O_train.shape[1])
            r = r.view(-1, 1)
            r = Variable(r)
            r = r.cuda()

            if use_cuda:
                inp = inp.cuda()
                out = out.cuda()

            y = model(inp)

            true = out.data
            pred = y.data
            ind = torch.arange(0, true.size(0)).long().cuda()
            ind = ind[true != 0]

            loss = nn.functional.nll_loss(r[ind] * nn.functional.log_softmax(y[ind]), out[ind])
            precision = acc(pred[ind], true[ind])
            precisions.append(precision)
            losses.append(loss.data[0])
        
        mean_precision = np.mean(precisions)
        mean_loss = np.mean(losses)
        print('Epoch : {:05d}, Test loss  : {:.6f}, Test precision  : {:.6f}'.format(i, mean_loss, mean_precision))

        if mean_loss < best_loss:
            best_loss = mean_loss
            print('Improved score, saving the model.')
            torch.save(model, 'rnn.th')
            last_epoch_improving = i
        else:
            print('No improvements.')
        if i - last_epoch_improving >= 100 and i - last_epoch_annealing  >= 100:
            last_epoch_annealing = i
            print('Annealing learning rate.')
            for param_group in optim.param_groups:
                param_group['lr'] *= 0.1
    

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
    print(X.shape, p.shape)
    ind = np.arange(len(X))
    ind = np.random.choice(ind, size=nb * len(X), p=p, replace=True)
    X = X[ind]
    Y = np.ones(len(X))
    return X, Y

def sample(source='random'):
    if source == 'random':
        random()
    elif source == 'rnn':
        rnn()


def example():
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    from grammaropt.grammar import as_str
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import RandomForestRegressor
    reg = make_pipeline(
        TfidfVectorizer(),
        RandomForestRegressor(max_depth=80),
    )
    db = load_db()
    jobs = db.jobs_with(state='success')
    jobs = list(jobs)
    jobs = [j for j in jobs if j['sampler'] == 'deepgrammar.samplers.random']
    X = [j['content']['info']['architecture'] for j in jobs]
    R = np.array([max(j['stats']['valid']['accuracy']) for j in jobs])
    reg.fit(X, R)
    print((np.abs(reg.predict(X) - R) < 0.05).mean())
    model = torch.load('rnn.th', map_location=lambda storage, loc: storage)
    model.train()
    model.use_cuda = False
    rnn = RnnAdapter(model, model.vect.tok_to_id, random_state=42)
    wl = RnnWalker(grammar=model.vect.grammar, rnn=rnn)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    R_avg = 0.
    R_list_rnn = []
    nb_iter = 500
    #rnn
    for _ in range(nb_iter):
        wl.walk()
        code = as_str(wl.terminals)
        R = float(reg.predict([code])[0])
        R_avg = R_avg * 0.9 + R * 0.1
        model.zero_grad()
        loss = (R - R_avg) * wl.compute_loss() / len(wl._decisions)
        loss.backward()
        optim.step()
        R_list_rnn.append(R_avg)
    #random
    wl = RandomWalker(grammar=model.vect.grammar, min_depth=5, max_depth=30)
    R_list_random = []
    for _ in range(nb_iter):
        wl.walk()
        code = as_str(wl.terminals)
        R = float(reg.predict([code])[0])
        R_list_random.append(R)
    #R_list_random = np.maximum.accumulate(R_list_random)
    #R_list_rnn = np.maximum.accumulate(R_list_rnn)
    fig = plt.figure(figsize=(15, 5))
    plt.plot(R_list_random, color='b', label='random')
    plt.plot(R_list_rnn, color='g', label='rnn')
    plt.legend()
    plt.savefig('rnn.png')

def check():
    import editdistance
    db = load_db()
    jobs = db.jobs_with(state='success')
    jobs = list(jobs)
    jobs = [j for j in jobs if j['sampler'] == 'deepgrammar.samplers.random']
    X = [j['content']['info']['architecture'] for j in jobs]
    R = np.array([max(j['stats']['valid']['accuracy']) for j in jobs])
    for _ in range(10):
        code = rnn()['info']['architecture']
        indices = np.argsort([editdistance.eval(x, code) for x in X])
        print(R[indices[0:5]])


if __name__ == '__main__':
    run([fit, sample, check, example])
