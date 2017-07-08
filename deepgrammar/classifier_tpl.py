import time
from functools import partial
import numpy as np
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import SeparableConv2D
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.models import Model
from keras import optimizers
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.regularizers import l2, l1

from sklearn.utils import shuffle


class LearningRateScheduler(Callback):
    
    def __init__(self, init_lr, nb_epochs):
        super().__init__()
        self.init_lr = init_lr
        self.nb_epochs = nb_epochs

    def on_epoch_begin(self, epoch, logs):
        ratio = float(epoch) / self.nb_epochs
        if ratio < 0.5:
            new_lr = self.init_lr
        elif 0.5 <= ratio <= 0.75:
            new_lr = self.init_lr / 10.
        elif ratio > 0.75:
            new_lr = self.init_lr / 100.
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
            LearningRateScheduler(lr, epochs),
        ]
        X = self.transform(X)
        y = to_categorical(y, n_outputs)
        X_train = X
        y_train = y
        X_train_flip = X_train[:,:,:,::-1]
        y_train_flip = y_train
        X_train = np.concatenate((X_train, X_train_flip),axis=0)
        y_train = np.concatenate((y_train, y_train_flip),axis=0)
        X_train, y_train = shuffle(X_train, y_train)
        self.model.fit(
            X_train, y_train, 
            epochs=epochs, 
            callbacks=callbacks, 
            batch_size=batch_size, 
        )
        return self
      
    def transform(self, X): 
        return (X - self.mu) / self.std

    def predict(self, X):
        X = self.transform(X)
        return self.model.predict(X).argmax(axis=1)

    def predict_proba(self, X):
        X = self.transform(X)
        return self.model.predict(X)
