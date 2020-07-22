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

from playsound import playsound

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
    if (epoch + 1) % 100 == 0:
        print(epoch+1, logs['val_acc'])

def main():
    ############################################ Read In Data From File ################################################
    # Read in column 0 from Table1 for the name of the galaxy
    galaxyName = genfromtxt('Paper2Table1.csv', delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=0)

    # Read in Column 6 from Table1 (Maser Classification)
    maserType = genfromtxt('Paper2Table1.csv', delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=6)

    # Read in L12 from Table1
    L12 = genfromtxt('Paper2Table1.csv', delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=7)

    # Read in Lobs from Table2
    Lobs = genfromtxt('Paper2Table2.csv', delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=4)

    ########################################## Normalize the Data ######################################################
    # Normalize L12
    maxValL12 = np.amax(L12)
    minValL12 = np.amin(L12)
    countL12 = 0
    for value in L12:
        L12[countL12] = (value - minValL12) / (maxValL12 - minValL12)
        countL12 += 1

    # Normalize Lobs
    maxValLobs = np.amax(Lobs)
    minValLobs = np.amin(Lobs)
    countLobs = 0
    for value in Lobs:
        Lobs[countLobs] = (value - minValLobs) / (maxValLobs - minValLobs)
        countLobs += 1

    ########################################## Reshape the Data Matrix #################################################
    # Currently, the shape of the data matrix is flipped
    # Reshape the data matrix to have 2 columns, one for each attribute
    # and as many rows as there are examples (galaxies)
    data = []
    count = 0
    for value in L12:
        data.append([L12[count], Lobs[count]])
        count += 1

    if len(data) == 642 and len(data[0]) == 2 and len(maserType) == 642:
        print("Data loaded properly")
    else:
        exit("Data loaded improperly")

    print("Length of data = ", len(data))
    print("Length of Data[0]", len(data[0]))
    print("Length of MaserType[] = ", len(maserType))

    ########################################## Sort the Masers from the Non-Masers #####################################
    # Sort out the masers and non masers for selection of training data
    # Change all non-zero values of maser classification to 1 for easy binary classification
    # Create a list of all non-masers and masers
    masers = []
    nonMasers = []

    count = 0
    # This is the number of masers; will be used to know how many non-masers to choose for the training data
    maserCount = 0
    for value in maserType:
        if value > 0:
            maserType[count] = 1
            maserCount += 1
            masers.append(data[count])
            count += 1
        else:
            nonMasers.append(data[count])
            count += 1

    if len(masers) == 68 and len(nonMasers) == 574:
        print("Total Masers and NonMasers Separated Correctly")
        print("Number of Total Maser Galaxies = ", len(masers))
        print("Number of Total NonMaser Galaxies = ", len(nonMasers))
    else:
        exit("Maser and NonMaser Separation Error")

    ####################################################################################################################
    ######################################## Perform Undersampling of NonMaser Data ####################################
    # Create a random sampling of training data from the nonMaser list
    # Creates a data range the size of the nonMaser dataset for undersampling purposes
    upperBound = len(nonMasers)
    dataRange = range(0, upperBound)

    ######################################## Outer Loop: Choosing Random Data ##########################################
    # Chooses random data numOfIterations times to perform KNN analysis and Stratified Validation

    # kAverages = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # # Used to graph accuracy of each k value; k = 1-15
    # f1Averages = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # # Used to graph f1 score of each k value; k = 1-15

    # kAverages = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # # Used to graph accuracy of each k value; k = 1-30
    # f1Averages = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # # Used to graph f1 score of each k value; k = 1-30
    numOfIterations = 1000
    print("Number of Iterations This Run = ", numOfIterations)
    dataIterations = range(0, numOfIterations)

    bestAcc = 0
    bestF1Score = 0
    bestTrainSetX = []
    bestTrainSetY = []
    bestValidationSetX = []
    bestValidationSetY = []
    bestTestSetX = []
    bestTestSetY = []
    # bestKVal = 0
    bestXDataSet = []

    for num in dataIterations:
        if (num % (numOfIterations / 10)) == 0:
            print("Iteration Number ", num)
        # Choose k number of random nonMaser galaxies where k = number of Maser galaxies
        chosen = random.sample(dataRange, k=maserCount)

        # Build the X dataset for use in the neural network training based on the randomly selected nonMaser galaxies
        # ALTERNATE adding maser, non-maser to X data set
        X = []
        # Create the class value list to go with the data set for accuracy testing
        Class = []
        count = 0
        for value in chosen:
            X.append(nonMasers[value])
            Class.append(0)
            X.append((masers[count]))
            Class.append(1)
            count += 1

        # print(X)
        # print(Class)

        #################################### Testing the Neural network Model ##########################################
        # Implements Stratified Test Set Validation to test accuracy of KNN model

        # Creates a random selection of Train and Test data
        # Test data is 20%, Train data is 80%
        randNum = random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(X, Class, test_size=0.2, random_state=randNum)

        # Creates a random selection of Train and Validation data
        # Validation is 25% of Train data, which is 20% of the total data
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=randNum)

        # Create lists used to store all the masers and nonmasers from the training and validation set
        # Used for graphing the training and validation set for visualization
        trainMaserListX = []
        trainMaserListY = []
        trainNonMaserListX = []
        trainNonMaserListY = []
        valMaserListX = []
        valMaserListY = []
        valNonMaserListX = []
        valNonMaserListY = []

        # Add masers from training set to the training maser list
        # Add nonmasers from the training set to the training nonmaser list
        count = 0
        for value in y_train:
            if value == 0:
                for val in X_train[count]:
                    trainNonMaserListX.append(X_train[count][0])
                    trainNonMaserListY.append((X_train[count][1]))
            else:
                for val in X_train[count]:
                    trainMaserListX.append(X_train[count][0])
                    trainMaserListY.append((X_train[count][1]))
            count += 1

        # Add masers from validation set to the validation maser list
        # Add nonmasers from the validation set to the validation nonmaser list
        count = 0
        for value in y_val:
            if value == 0:
                for val in X_val[count]:
                    valNonMaserListX.append(X_val[count][0])
                    valNonMaserListY.append(X_val[count][1])
            else:
                for val in X_val[count]:
                    valMaserListX.append(X_val[count][0])
                    valMaserListY.append(X_val[count][1])
            count += 1

        # ######## Create the Large 6 Layer NN model to train
        # num_epochs = 550
        #
        # model = Sequential()
        # model.add(Dense(10, input_dim=2, activation='relu'))
        # model.add(Dense(10, activation='relu'))
        # model.add(Dense(20, activation='relu'))
        # model.add(Dense(50, activation='relu'))
        # model.add(Dense(17, activation='relu'))
        # model.add(Dense(1, activation='sigmoid'))
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        ######## Create the 3 Layer NN model to train
        num_epochs = 1000

        model = Sequential()
        model.add(Dense(5, input_dim=2, activation='relu'))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # ######## Create the 4 Layer NN model to train
        # num_epochs = 1000
        #
        # model = Sequential()
        # model.add(Dense(6, input_dim=2, activation='relu'))
        # model.add(Dense(4, activation='relu'))
        # model.add(Dense(2, activation='relu'))
        # model.add(Dense(1, activation='sigmoid'))
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # ######## Create the Small 6 Layer NN model to train
        # num_epochs = 550
        #
        # model = Sequential()
        # model.add(Dense(6, input_dim=2, activation='relu'))
        # model.add(Dense(5, activation='relu'))
        # model.add(Dense(4, activation='relu'))
        # model.add(Dense(3, activation='relu'))
        # model.add(Dense(2, activation='relu'))
        # model.add(Dense(1, activation='sigmoid'))
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Fit the model on the training set and predict the labels of the X-validation set
        model.fit(np.array(X_train), y_train, epochs=num_epochs, batch_size=68,
                  validation_data=(np.array(X_val), y_val), verbose=0)
        yValPred = model.predict_classes(np.array(X_val))
        # print("yValPred: ", yValPred)
        # print("Type of yValPred: ", type(yValPred))

        # Compute the accuracy of the predicted values
        sklearn_acc = metrics.accuracy_score(y_val, yValPred)
        # print('accuracy from sklearn is:', sklearn_acc)

        # # Compute the f1 score of the predicted values
        f1 = metrics.f1_score(y_val, yValPred)
        # print('f1 is:', f1)

        # If the F1 score of this k value and training set is better than any other previous f1 value,
        # store the accuracy, f1 score, training set, test set, and k value for use later
        # Will be used to rebuild this exact model to test the test data on
        if f1 > bestF1Score:
            bestF1Score = f1
            bestAcc = sklearn_acc
            bestTrainSetX = X_train
            bestTrainSetY = y_train
            bestTestSetX = X_test
            bestTestSetY = y_test
            # model.save('bestNNModel.h5')
            model.save('./DataSetNN_3Layer/bestNNModel.h5')
            bestXDataSet = X

    # print("bestTrainSetX = ", bestTrainSetX)

    # testLarge6LayerModel = load_model('./DatasetNN_6LayerLarge/bestNNModel.h5')
    # testSmall6LayerModel = load_model('./DatasetNN_6LayerSmall/bestNNModel.h5')
    # test4LayerModel = load_model('./DataSetNN_4Layer/bestNNModel.h5')
    test3LayerModel = load_model('./DataSetNN_3Layer/bestNNModel.h5')
    testYPred = test3LayerModel.predict_classes(np.array(bestTestSetX))
    testAcc = metrics.accuracy_score(bestTestSetY, testYPred)
    testF1 = metrics.f1_score(bestTestSetY, testYPred)

    f = open('./DataSetNN_3Layer/bestNNDataStats.txt', 'w')
    # f = open('./DataSetNN_4Layer/bestNNDataStats.txt', 'w')
    # f = open('./DataSetNN_6LayerLarge/bestNNDataStats.txt', 'w')
    # f = open('./DataSetNN_6LayerSmall/bestNNDataStats.txt', 'w')
    f.write("Best NN Training Accuracy = %.3f\n" % bestAcc)
    f.write("Best NN Training F1 Score = %.3f\n" % bestF1Score)
    f.write("Best Unweighted Test Accuracy = %.3f\n" % testAcc)
    f.write("Best Unweighted Test F1 Score = %.3f" % testF1)
    # f.write("Number of Iterations of Undersampled NonMaser Datasets = " % numOfIterations)
    f.close()

    # Constant names for the long string names of the saved .npy files for the 3 layer model
    xNNDataTrain = './DataSetNN_3Layer/DataTrainingXNN'
    yNNDataTrain = './DataSetNN_3Layer/DataTrainingYNN'
    xNNDataVal = './DataSetNN_3Layer/DataValidationXNN'
    yNNDataVal = './DataSetNN_3Layer/DataValidationYNN'
    xNNDataTest = './DataSetNN_3Layer/DataTestXNN'
    yNNDataTest = './DataSetNN_3Layer/DataTestYNN'
    bestNNDataStats = './DataSetNN_3Layer/BestNNDataSetStats'
    bestXDataSetStr = './DataSetNN_3Layer/BestXDataSet'

    # # Constant names for the long string names of the saved .npy files for the 4 layer model
    # xNNDataTrain = './DataSetNN_4Layer/DataTrainingXNN'
    # yNNDataTrain = './DataSetNN_4Layer/DataTrainingYNN'
    # xNNDataVal = './DataSetNN_4Layer/DataValidationXNN'
    # yNNDataVal = './DataSetNN_4Layer/DataValidationYNN'
    # xNNDataTest = './DataSetNN_4Layer/DataTestXNN'
    # yNNDataTest = './DataSetNN_4Layer/DataTestYNN'
    # bestNNDataStats = './DataSetNN_4Layer/BestNNDataSetStats'
    # bestXDataSetStr = './DataSetNN_4Layer/BestXDataSet'

    # # Constant names for the long string names of the saved .npy files for the 6 layer large model
    # xNNDataTrain = './DataSetNN_6LayerLarge/DataTrainingXNN'
    # yNNDataTrain = './DataSetNN_6LayerLarge/DataTrainingYNN'
    # xNNDataVal = './DataSetNN_6LayerLarge/DataValidationXNN'
    # yNNDataVal = './DataSetNN_6LayerLarge/DataValidationYNN'
    # xNNDataTest = './DataSetNN_6LayerLarge/DataTestXNN'
    # yNNDataTest = './DataSetNN_6LayerLarge/DataTestYNN'
    # bestNNDataStats = './DataSetNN_6LayerLarge/BestNNDataSetStats'
    # bestXDataSetStr = './DataSetNN_6LayerLarge/BestXDataSet'

    # # Constant names for the long string names of the saved .npy files for the 6 layer small model
    # xNNDataTrain = './DataSetNN_6LayerSmall/DataTrainingXNN'
    # yNNDataTrain = './DataSetNN_6LayerSmall/DataTrainingYNN'
    # xNNDataVal = './DataSetNN_6LayerSmall/DataValidationXNN'
    # yNNDataVal = './DataSetNN_6LayerSmall/DataValidationYNN'
    # xNNDataTest = './DataSetNN_6LayerSmall/DataTestXNN'
    # yNNDataTest = './DataSetNN_6LayerSmall/DataTestYNN'
    # bestNNDataStats = './DataSetNN_6LayerSmall/BestNNDataSetStats'
    # bestXDataSetStr = './DataSetNN_6LayerSmall/BestXDataSet'

    np.save(xNNDataTrain, bestTrainSetX)
    np.save(yNNDataTrain, bestTrainSetY)
    np.save(xNNDataVal, X_val)
    np.save(yNNDataVal, y_val)
    np.save(xNNDataTest, bestTestSetX)
    np.save(yNNDataTest, bestTestSetY)
    np.save(bestXDataSetStr, bestXDataSet)

    print("Number of Iterations = ", numOfIterations)
    print("Best Training Accuracy = ", bestAcc)
    print("Best Training F1 Score = ", bestF1Score)
    print("Test Accuracy = ", testAcc)
    print("Test F1 Score = ", testF1)

    # #### Play a fun sound to alert me to it being done
    # playsound('collect_coin_8bit.mp3')
    # playsound('collect_coin_8bit.mp3')
    # playsound('collect_coin_8bit.mp3')

if __name__ == '__main__':
    main()
