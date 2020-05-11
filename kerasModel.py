#    name: kerasModel.py
# purpose: builds a model for handwrited digit classification
# -------------------------------------------------


import numpy as np
import sys
from keras.callbacks import ModelCheckpoint, CSVLogger, LambdaCallback

from numpy import genfromtxt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from sklearn import metrics
import time

# from multiDataPreprocessing import processTestData
import argparse


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
    if (epoch + 1) % 5 == 0:
        print(epoch+1, logs['val_acc'])

def main():
    # ############################################ Read In Data From File ####################################################
    # # Read in column 0 from Table1 for the name of the galaxy
    # galaxyName = genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=0)
    #
    # # Read in Column 6 from Table1 (Maser Classification)
    # maserType = genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=6)
    #
    # # Read in L12 from Table1
    # L12 = genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=7)
    #
    # # Read in Lobs from Table2
    # Lobs = genfromtxt(sys.argv[2], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=4)
    #
    # ########################################## Normalize the Data ##########################################################
    # # Normalize L12
    # maxValL12 = np.amax(L12)
    # minValL12 = np.amin(L12)
    # countL12 = 0
    # for value in L12:
    #     L12[countL12] = (value - minValL12) / (maxValL12 - minValL12)
    #     countL12 += 1
    #
    # # Normalize Lobs
    # maxValLobs = np.amax(Lobs)
    # minValLobs = np.amin(Lobs)
    # countLobs = 0
    # for value in Lobs:
    #     Lobs[countLobs] = (value - minValLobs) / (maxValLobs - minValLobs)
    #     countLobs += 1
    #
    # ########################################## Reshape the Data Matrix #####################################################
    # # Currently, the shape of the data matrix is flipped
    # # Reshape the data matrix to have 2 columns, one for each attribute, and as many rows as there are examples (galaxies)
    # data = []
    # count = 0
    # for value in L12:
    #     data.append([L12[count], Lobs[count]])
    #     count += 1
    #
    # if len(data) == 642 and len(data[0]) == 2 and len(maserType) == 642:
    #     print("Data loaded properly")
    # else:
    #     exit("Data loaded improperly")
    #
    # print("Length of data = ", len(data))
    #     print("Length of Data[0]", len(data[0]))
    #     print("Length of MaserType[] = ", len(maserType))
    #
    #     # ########################################## Sort the Masers from the Non-Masers #########################################
    #     # # Sort out the masers and non masers for selection of training data
    #     # # Change all non-zero values of maser classification to 1 for easy binary classification
    #     # # Create a list of all non-masers and masers
    #     # masers = []
    #     # nonMasers = []
    #     #
    #     # count = 0
    #     # # This is the number of masers; will be used to know how many non-masers to choose for the training data
    #     # maserCount = 0
    #     # for value in maserType:
    #     #     if value > 0:
    #     #         maserType[count] = 1
    #     #         maserCount += 1
    #     #         masers.append(data[count])
    #     #         count += 1
    #     #     else:
    #     #         nonMasers.append(data[count])
    #     #         count += 1
    #     #
    #     # if len(masers) == 68 and len(nonMasers) == 574:
    #     #     print("Total Masers and NonMasers Separated Correctly")
    #     #     print("Number of Total Maser Galaxies = ", len(masers))
    #     #     print("Number of Total NonMaser Galaxies = ", len(nonMasers))
    #     # else:
    #     #     exit("Maser and NonMaser Separation Error")





    # # Load in the saved undersampled dataset
    # bestTrainSetX1 = np.load('./DataSet80+Acc/DataTrainingXKNNUnweighted.npy')
    # bestTrainSetY1 = np.load('./DataSet80+Acc/DataTrainingYKNNUnweighted.npy')
    # bestValidSetX1 = np.load('./DataSet80+Acc/DataValidationXKNNUnweighted.npy')
    # bestValidSetY1 = np.load('./DataSet80+Acc/DataValidationYKNNUnweighted.npy')
    # bestTestSetX1 = np.load('./DataSet80+Acc/DataTestXKNNUnweighted.npy')
    # bestTestSetY1 = np.load('./DataSet80+Acc/DataTestYKNNUnweighted.npy')

    trainX = np.load('DataTrainingXKNNUnweighted.npy')
    trainY = np.load('DataTrainingYKNNUnweighted.npy')
    validX = np.load('DataValidationXKNNUnweighted.npy')
    validY = np.load('DataValidationYKNNUnweighted.npy')
    testX= np.load('DataTestXKNNUnweighted.npy')
    testY = np.load('DataTestYKNNUnweighted.npy')


    np.random.seed(1671)

    # parms = parseArguments()

    print("Data: ", trainX)
    print("Length of Data X = ", len(trainX))
    print("Type of data: ", type(trainX))

    print('KERA modeling build starting...')
    # Build your model here
    num_epochs = 600
    period=num_epochs//20

    # # Original Model Code
    # model = Sequential()
    # model.add(Dense(1, input_dim=2, activation='relu'))
    # model.add(Activation('softmax'))
    # model.summary()
    # model.compile(loss='binary_crossentropy', optimizer=SGD(), metrics=['accuracy'])
    # tic = time.time()

    # 4 layer model code
    model = Sequential()
    model.add(Dense(8, input_dim=2, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # # 3 Layer Model Code
    # model = Sequential()
    # model.add(Dense(12, input_dim=2, activation='relu'))
    # model.add(Dense(8, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    # # Original model.fit code
    # model.fit(X_train, y_train, batch_size=1000, epochs=num_epochs, validation_split=0.2, verbose=0, callbacks=[
    #           ModelCheckpoint(parms.outModelFile[:-3] + "_epoch_{epoch:d}.h5", period=period),
    #           LambdaCallback(on_epoch_end=lambda epoch, logs: print(epoch+1, time.time() - tic, logs['val_acc'], sep=",") if (epoch+1)%period==0 else None)
    #          ])

    # Troubleshooting model.fit code
    model.fit(trainX, trainY, epochs=num_epochs, batch_size=1000, validation_data=(validX, validY), verbose = 1)
    yTrainPred = model.predict_classes(trainX)
    yTestPred = model.predict_classes(testX)

    print(yTestPred)

    # Calculate accuracy and F1 Score
    _, trainAcc = model.evaluate(trainX, trainY)
    print('Train Acc = %.2f' % (trainAcc * 100))
    trainF1 = metrics.f1_score(trainY, yTrainPred)
    print('Train F1 = %.3f' % trainF1)
    testF1 = metrics.f1_score(testY, yTestPred)
    print('Test F1 Score: %.3f' % testF1)
    _, testAccuracy = model.evaluate(testX, testY)
    print('Test Accuracy: %.2f' % (testAccuracy * 100))

    # f = open('3LayerKerasDataStats12_8_1.txt', 'w')
    # f.write("Number of Layers = 3\n")
    # f.write("Layer1 = 12 Nodes ReLU\nLayer2 = 8 Nodes ReLU\nLayer3 = 1 Node Sigmoid\n")
    # f.write("Best Keras Traing Accuracy = %.3f\n" % trainAcc)
    # f.write("Best Keras Training F1 Score = %.3f\n" % trainF1)
    # f.write("Best Keras Test Accuracy = %.3f\n" % testAccuracy)
    # f.write("Best Keras Test F1 Score = %.3f" % testF1)
    # f.close()
    #
    # # save your model
    # model.save('kerasModel3Layer_12_8_1.h5')

    f = open('4LayerKerasDataStats8_12_4_1.txt', 'w')
    f.write("Number of Layers = 4\n")
    f.write("Layer1 = 8 Nodes ReLU\nLayer2 = 12 Nodes ReLU\nLayer3 = 4 Nodes ReLU\nLayer4 = 1 Node Sigmoid\n")
    f.write("Best Keras Traing Accuracy = %.3f\n" % trainAcc)
    f.write("Best Keras Training F1 Score = %.3f\n" % trainF1)
    f.write("Best Keras Test Accuracy = %.3f\n" % testAccuracy)
    f.write("Best Keras Test F1 Score = %.3f" % testF1)
    f.close()

    # save your model
    model.save('kerasModel4Layer_8_12_4_1.h5')



if __name__ == '__main__':
    main()
