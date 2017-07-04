from grammaropt.grammar import build_grammar
from grammaropt.grammar import as_str
from grammaropt.random import RandomWalker


keras_classifier = """
import numpy as np
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.models import Model
from keras import optimizers
from keras.utils import to_categorical

EPS = 1e-7

class Classifier:
    
    def fit(self, X, y):
        axes = tuple(set(range(len(X.shape))) - set([1]))
        self.mu = X.mean(axis=axes, keepdims=True)
        self.std = X.std(axis=axes, keepdims=True)
        X = (X - self.mu) / (self.std + EPS)
        n_outputs = len(np.unique(y))
        
        inp = Input(X.shape[1:])
        x = inp
        {architecture}
        x = Flatten()(x)
        out = Dense(n_outputs, activation='softmax')(x)
        self.model = Model(inp, out)
        self.model.summary()
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
        )
        y = to_categorical(y, n_outputs)
        self.model.fit(X, y, epochs=epochs)
        self.model.predict_proba = self.model.predict
        self.model.predict = lambda X: self.model.predict_proba(X).argmax(axis=1)
        
    def predict(self, X):
        X = (X - self.mu) / (self.std + EPS)
        return self.model.predict(X)

    def predict_proba(self, X):
        X = (X - self.mu) / (self.std + EPS)
        return self.model.predict_proba(X)
"""


grammar = r"""
archi = optim_stmt archi_stmts
optim_stmt = "opt = optimizers." (sgd / adam / rmsprop) "\n        epochs = " epochs "\n"
sgd = "SGD(lr=" lr ", momentum=" momentum ", decay=" decay ", nesterov=" nesterov ")"
adam = "Adam(lr=" lr ", beta_1=" beta_1 ", beta_2=" beta_2 ", decay=" decay ")"
rmsprop = "RMSprop(lr=" lr "," "rho=" rho ",decay=" decay ")"
lr = "0.001"
rho = "0.9"
momentum = "0.9"
beta_1 = "0.9"
beta_2 = "0.999"
nesterov = bool
bool = "True" / "False"
decay = "0.0"
archi_stmts = (archi_stmt archi_stmts) / archi_stmt
archi_stmt = "        x = Conv2D(" filters ", " kernel_size ", strides=" strides ", activation=" activation ", kernel_initializer=" init ")(x)\n"
filters = "16" / "32"
kernel_size =  "1" / "3" / "5"
strides = "(1,1)"
activation = "\"relu\""
init = "\"glorot_uniform\""
epochs = "30"
"""
grammar = build_grammar(grammar)

def classifier():
    wl = RandomWalker(grammar, min_depth=1, max_depth=10)
    wl.walk()
    architecture = as_str(wl.terminals)
    code = keras_classifier.format(architecture=architecture)
    print(code)
    out = {
        'codes': {
            'classifier': code
        },
        'info': {}
    }
    return out


if __name__ == "__main__":
    out = classifier()
    print(out['codes']['classifier'])
