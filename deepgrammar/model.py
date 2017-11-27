import torch.nn as nn

from grammaropt.rnn import RnnModel

class Model(RnnModel):
    def __init__(self, vocab_size=10, emb_size=128, hidden_size=128, num_layers=1, nb_features=1, use_cuda=False):
        super().__init__()   
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda

        self.emb = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True, num_layers=num_layers, dropout=0.5)
        self.out_token  = nn.Linear(hidden_size, vocab_size)
        self.out_value = nn.Linear(hidden_size, nb_features)
