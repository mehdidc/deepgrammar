import numpy as np
from grammaropt.grammar import build_grammar
from grammaropt.grammar import as_str
from grammaropt.random import RandomWalker

keras_classifier = """
from functools import partial
import numpy as np
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.models import Model
from keras import optimizers
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler
from keras.regularizers import l2, l1
from keras.backend import epsilon

nb_epochs_before_reduce = 10
max_times_reduce_lr = 12
lr_reduce_factor = 0.5
reduce_wait = 8

def schedule(epoch, nb_epochs, model, init_lr):
    hist = model.history.history
    old_lr = float(model.optimizer.lr.get_value())
    new_lr = old_lr
    # do deep convnets really need to be deep LR schedule
    if 'val_acc' in hist:
        valid_accs = hist['val_acc']
        if not hasattr(model, 'last_reduced'):
            model.last_reduced = 0
            model.nb_reduce_lr = 0
        if len(valid_accs) > nb_epochs_before_reduce and (epoch - model.last_reduced) > reduce_wait:
            up_to_last = valid_accs[0:-nb_epochs_before_reduce]
            max_valid = max(up_to_last)
            max_last = max(valid_accs[-nb_epochs_before_reduce:])
            if max_valid >= max_last:
                new_lr = old_lr * lr_reduce_factor 
                model.nb_reduce_lr += 1
                model.last_reduced = epoch
                print('%d epochs without improvement : reduce lr from %.6f to %.6f' % (nb_epochs_before_reduce, old_lr, new_lr))
                if model.nb_reduce_lr == max_times_reduce_lr:
                    print('reduced lr %d times, quit.' % max_times_reduce_lr)
                    model.stop_training = True
    # DenseNet-kind LR schedule
    else:
        ratio = float(epoch) / nb_epochs
        if ratio < 0.5:
            new_lr = init_lr
        elif 0.5 <= ratio <= 0.75:
            new_lr = init_lr / 10.
        elif ratio > 0.75:
            new_lr = init_lr / 100.
    return new_lr

class Classifier:
    
    def fit(self, X, y):
        axes = tuple(set(range(len(X.shape))) - set([1]))
        self.mu = X.mean(axis=axes, keepdims=True)
        self.std = X.std(axis=axes, keepdims=True)
        n_outputs = len(np.unique(y))
        
        inp = Input(X.shape[1:])
        x = inp
{architecture}
        out = Dense(n_outputs, activation='softmax', kernel_initializer='glorot_uniform')(x)
        self.model = Model(inp, out)
        self.model.summary()
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
        )
        callbacks = [
            LearningRateScheduler(partial(schedule, nb_epochs=epochs, model=self.model, init_lr=lr))
        ]
        X = self.transform(X)
        y = to_categorical(y, n_outputs)
        self.model.fit(X, y, epochs=epochs, callbacks=callbacks, batch_size=batch_size, validation_split=ratio_valid)
        return self
      
    def transform(self, X): 
        return (X - self.mu) / (self.std + epsilon())

    def predict(self, X):
        X = self.transform(X)
        return self.model.predict(X).argmax(axis=1)

    def predict_proba(self, X):
        X = self.transform(X)
        return self.model.predict(X)
"""


grammar = r"""
archi = optim_stmts archi_stmts
optim_stmts = lr_stmt "\n" epochs_stmt "\n" batch_size_stmt "\n" ratio_valid_stmt "\n" optimizer_stmt "\n"  activation_stmt "\n"
optimizer_stmt = "opt = optimizers." opt
epochs_stmt = "epochs = " epochs 
batch_size_stmt = "batch_size = " batch_size
ratio_valid_stmt = "ratio_valid = " ratio_valid
activation_stmt = "activation = " activation
lr_stmt = "lr = " lr
opt = sgd
sgd = "SGD(lr=lr, momentum=" momentum ", decay=" decay ", nesterov=" nesterov ")"
adam = "Adam(lr=" lr ", beta_1=" beta_1 ", beta_2=" beta_2 ", decay=" decay ")"
rmsprop = "RMSprop(lr=" lr "," "rho=" rho ",decay=" decay ")"
ratio_valid = "0.1"
lr = "0.1" / "0.05" / "0.01" / "0.005" / "0.001"
rho = "0.9"
batch_size = "64"
momentum = "0.9"
beta_1 = "0.9"
beta_2 = "0.999"
nesterov = "False"
bool = "True" / "False"
decay = "0.0"
archi_stmts = conv_archi_stmts flatten fc_archi_stmts
conv_archi_stmts = (conv_archi_stmt conv_archi_stmts) / conv_archi_stmt
conv_archi_stmt = (conv dropout pooling) / (conv dropout) / conv
flatten = "x = Flatten()(x)\n"
fc_archi_stmts = (fc_archi_stmt fc_archi_stmt) / fc_archi_stmt
fc_archi_stmt = (fc dropout) / fc 
fc = "x = Dense(" nb_units ", activation=activation, kernel_initializer=" init ", kernel_regularizer = " reg ")(x)\n"
conv = "x = Conv2D(" filters ", " kernel_size ", strides=" strides ", activation=activation, kernel_initializer=" init ", kernel_regularizer=" reg  ", padding=" padding ")(x)\n"
reg = "l2(1e-4)"
padding = "\"valid\"" / "\"same\""
filters = "16" / "32" / "48" / "64" / "80" / "96" / "112"
kernel_size =  "1" / "3" / "5" / "7"
strides = "(1, 1)"
activation = "\"relu\"" / "\"elu\"" 
pooling = "x = " poolingop "(2)(x)\n"
poolingop = "MaxPooling2D"
init = "\"glorot_uniform\""
dropout = "x = Dropout(" proba ")(x)\n"
proba = "0.1" / "0.3"/ "0.5"
epochs = "150"
nb_units = "128" / "256" / "512" / "1024" / "2048"
"""
grammar = build_grammar(grammar)


def classifier():
    min_depth = np.random.choice((5, 10, 15))
    max_depth = min_depth + 10
    wl = RandomWalker(grammar, min_depth=min_depth, max_depth=max_depth)
    wl.walk()
    architecture = as_str(wl.terminals)
    architecture = _indent(architecture, 8)
    code = keras_classifier.format(architecture=architecture)
    print(code)
    out = {
        'codes': {
            'classifier': code
        },
        'info': {}
    }
    return out


def _indent(s, nb_spaces):
    return '\n'.join([" " * 8 + line for line in s.split('\n')])


if __name__ == "__main__":
    out = classifier()
    print(out['codes']['classifier'])
