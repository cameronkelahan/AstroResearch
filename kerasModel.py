#    name: kerasModel.py
# purpose: builds a model for handwrited digit classification
# -------------------------------------------------


import numpy as np
import random
import sys
from keras.callbacks import ModelCheckpoint, CSVLogger, LambdaCallback
from numpy import genfromtxt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
import time

# from multiDataPreprocessing import processTestData
import argparse

# Build NN models based on the "best" dataset found by the UnwKNN model

# def parseArguments():
#     parser = argparse.ArgumentParser(
#         description='Build a Keras model for Image classification')
#
#     parser.add_argument('--training_x', action='store',
#                         dest='XFile', default="", required=True,
#                         help='matrix of training images in npy')
#     parser.add_argument('--training_y', action='store',
#                         dest='yFile', default="", required=True,
#                         help='labels for training set')
#
#     parser.add_argument('--outModelFile', action='store',
#                         dest='outModelFile', default="", required=True,
#                         help='model name for your Keras model')
#
#     return parser.parse_args()

def print_stats(epoch, logs):
    if (epoch + 1) % 100 == 0:
        print(epoch+1, logs['val_acc'])

def main():
    # Load in the saved undersampled dataset
    trainX = np.load('./DataSet80+Acc/DataTrainingXKNNUnweighted.npy')
    trainY = np.load('./DataSet80+Acc/DataTrainingYKNNUnweighted.npy')
    validX = np.load('./DataSet80+Acc/DataValidationXKNNUnweighted.npy')
    validY = np.load('./DataSet80+Acc/DataValidationYKNNUnweighted.npy')
    testX = np.load('./DataSet80+Acc/DataTestXKNNUnweighted.npy')
    testY = np.load('./DataSet80+Acc/DataTestYKNNUnweighted.npy')

    # trainX = np.load('DataTrainingXKNNUnweighted.npy')
    # trainY = np.load('DataTrainingYKNNUnweighted.npy')
    # validX = np.load('DataValidationXKNNUnweighted.npy')
    # validY = np.load('DataValidationYKNNUnweighted.npy')
    # testX= np.load('DataTestXKNNUnweighted.npy')
    # testY = np.load('DataTestYKNNUnweighted.npy')


    np.random.seed(1671)

    # parms = parseArguments()

    print("Data: ", trainX)
    print("Length of Data X = ", len(trainX))
    print("Type of data: ", type(trainX))

    print('KERA modeling build starting...')
    # Build your model here
    num_epochs = 550
    period=num_epochs//20

    # # Original Model Code
    # model = Sequential()
    # model.add(Dense(1, input_dim=2, activation='relu'))
    # model.add(Activation('softmax'))
    # model.summary()
    # model.compile(loss='binary_crossentropy', optimizer=SGD(), metrics=['accuracy'])
    # tic = time.time()

    # # 4 layer model code
    # model = Sequential()
    # model.add(Dense(8, input_dim=2, activation='relu'))
    # model.add(Dense(12, activation='relu'))
    # model.add(Dense(4, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # # 3 Layer Model Code
    # model = Sequential()
    # model.add(Dense(12, input_dim=2, activation='relu'))
    # model.add(Dense(8, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 6 Layer Node
    model = Sequential()
    model.add(Dense(10, input_dim=2, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(17, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # # 10 Layer Node
    # model = Sequential()
    # model.add(Dense(10, input_dim=2, activation='relu'))
    # model.add(Dense(10, activation='relu'))
    # model.add(Dense(20, activation='relu'))
    # model.add(Dense(25, activation='relu'))
    # model.add(Dense(30, activation='relu'))
    # model.add(Dense(50, activation='relu'))
    # model.add(Dense(17, activation='relu'))
    # model.add(Dense(12, activation='relu'))
    # model.add(Dense(8, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    # # Original model.fit code
    # model.fit(X_train, y_train, batch_size=1000, epochs=num_epochs, validation_split=0.2, verbose=0, callbacks=[
    #           ModelCheckpoint(parms.outModelFile[:-3] + "_epoch_{epoch:d}.h5", period=period),
    #           LambdaCallback(on_epoch_end=lambda epoch, logs: print(epoch+1, time.time() - tic, logs['val_acc'], sep=",") if (epoch+1)%period==0 else None)
    #          ])

    # Troubleshooting model.fit code
    model.fit(trainX, trainY, epochs=num_epochs, batch_size=1000, validation_data=(validX, validY), verbose=1)
    yTrainPred = model.predict_classes(trainX)
    yTestPred = model.predict_classes(testX)

    # print(yTestPred)
    #
    # # Calculate accuracy and F1 Score
    # _, trainAcc = model.evaluate(trainX, trainY)
    # print('Train Acc = %.2f' % (trainAcc * 100))
    # trainF1 = metrics.f1_score(trainY, yTrainPred)
    # print('Train F1 = %.3f' % trainF1)
    # testF1 = metrics.f1_score(testY, yTestPred)
    # print('Test F1 Score: %.3f' % testF1)
    # _, testAccuracy = model.evaluate(testX, testY)
    # print('Test Accuracy: %.2f' % (testAccuracy * 100))

    # # File writing and model saving for 3 layer model
    # f = open('3LayerKerasDataStats12_8_1.txt', 'w')
    # f.write("Number of Layers = 3\n")
    # f.write("Layer1 = 12 Nodes ReLU\nLayer2 = 8 Nodes ReLU\nLayer3 = 1 Node Sigmoid\n")
    # f.write("Best Keras Traing Accuracy = %.3f\n" % trainAcc)
    # f.write("Best Keras Training F1 Score = %.3f\n" % trainF1)
    # f.write("Best Keras Test Accuracy = %.3f\n" % testAccuracy)
    # f.write("Best Keras Test F1 Score = %.3f" % testF1)
    # f.close()
    #
    # # save your mode
    # model.save('./DataSet80+Acc/kerasModel3Layer_12_8_1.h5')

    # # File writing and model saving for 4 layer model
    # f = open('4LayerKerasDataStats_8_12_4_1.txt', 'w')
    # f.write("Number of Layers = 4\n")
    # f.write('Layer1 = 8 Nodes ReLU\nLayer2 = 12 Nodes ReLU\nLayer3 = 4 Nodes ReLU\nLayer4 = 1 Node Sigmoid')
    # f.write("Best Keras Training Accuracy = %.3f\n" % trainAcc)
    # f.write("Best Keras Training F1 Score = %.3f\n" % trainF1)
    # f.write("Best Keras Test Accuracy = %.3f\n" % testAccuracy)
    # f.write("Best Keras Test F1 Score = %.3f" % testF1)
    # f.close()
    # # save your mode
    # model.save('./DataSet80+Acc/kerasModel4Layer_8_12_4_1.h5')

    # # File writing and model saving for 6 layer model
    # f = open('./DataSet80+Acc/6LayerKerasDataStatsEpoch600_10_10_20_50_17_1.txt', 'w')
    # f.write("Number of Layers = 6\n")
    # f.write('''Layer1 = 10 Nodes ReLU\nLayer2 = 10 Nodes ReLU\nLayer3 = 20 Nodes ReLU\nLayer4 = 50 Node ReLU\n
    # Layer5 = 17 Nodes ReLU\nLayer6 = 1 Node Sigmoid\n''')
    # f.write("Best Keras Training Accuracy = %.3f\n" % trainAcc)
    # f.write("Best Keras Training F1 Score = %.3f\n" % trainF1)
    # f.write("Best Keras Test Accuracy = %.3f\n" % testAccuracy)
    # f.write("Best Keras Test F1 Score = %.3f" % testF1)
    # f.close()
    # # save your model
    # model.save('./DataSet80+Acc/kerasModel6LayerEpoch550_10_10_20_50_17_1.h5')

#     # File writing and model saving for 10 layer model
#     f = open('./DataSetUnwKNN80+Acc/10LayerKerasDataStatsEpoch600_10_10_20_25_30_50_17_12_8_1.txt', 'w')
#     f.write("Number of Layers = 10\n")
#     f.write('''Layer1 = 10 Nodes ReLU\nLayer2 = 10 Nodes ReLU\nLayer3 = 20 Nodes ReLU\nLayer4 = 25 Node ReLU
# Layer5 = 30 Nodes ReLU\nLayer6 = 50 Nodes ReLU\nLayer7 = 17 Nodes ReLU\nLayer8 = 12 Nodes ReLU
# Layer9 = 8 Nodes ReLU\nLayer10 = 1 Node Sigmoid\n''')
#     f.write("Best Keras Training Accuracy = %.3f\n" % trainAcc)
#     f.write("Best Keras Training F1 Score = %.3f\n" % trainF1)
#     f.write("Best Keras Test Accuracy = %.3f\n" % testAccuracy)
#     f.write("Best Keras Test F1 Score = %.3f" % testF1)
#     f.close()
#
#     # save your model
#     model.save('./DataSetUnwKNN80+Acc/kerasModel10LayerEpoch550_10_10_20_25_30_50_17_12_8_1.h5')



if __name__ == '__main__':
    main()
