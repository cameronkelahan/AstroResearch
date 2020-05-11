#    name: m2.py
# purpose: builds a model for handwrited digit classification
# -------------------------------------------------


import numpy as np
from keras.callbacks import LambdaCallback, ModelCheckpoint

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model

# from multiDataPreprocessing import processTestData
import argparse
import time


def parseArguments():
    parser = argparse.ArgumentParser(
        description='Build a Keras model for Image classification')

    parser.add_argument('--training_x', action='store',
                        dest='XFile', default="", required=True,
                        help='matrix of training images in npy')
    parser.add_argument('--training_y', action='store',
                        dest='yFile', default="", required=True,
                        help='labels for training set')

    parser.add_argument('--outModelFile', action='store',
                        dest='outModelFile', default="", required=True,
                        help='model name for your Keras model')

    return parser.parse_args()

def main():
    np.random.seed(1671)

    parms = parseArguments()

    X_train = np.load(parms.XFile)
    y_train = np.load(parms.yFile)

    (X_train, y_train) = processTestData(X_train,y_train)

    print('KERA modeling build starting...')
    ## Build your model here
    num_epochs = 600
    period=num_epochs//20
    model = Sequential()
    model.add(Dense(10, input_shape=(28*28, ), activation='relu'))
    model.add(Dense(25, input_shape=(10, ), activation='relu'))
    model.add(Dense(10, input_shape=(25, ), activation='relu'))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])
    tic = time.time()
    model.fit(X_train, y_train, batch_size=1000, epochs=num_epochs, validation_split=0.2, verbose=0, callbacks=[
        ModelCheckpoint(parms.outModelFile[:-3] + "_epoch_{epoch:d}.h5", period=period),
        LambdaCallback(on_epoch_end=lambda epoch, logs: print(epoch+1, time.time() - tic, logs['val_acc'], sep=',') if (epoch+1)%period==0 else None)
    ])

    ## save your model
    model.save(parms.outModelFile)


if __name__ == '__main__':
    main()
