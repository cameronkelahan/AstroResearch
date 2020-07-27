import numpy as np
import random
import tensorflow as tf
import keras
from numpy import genfromtxt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.utils import Sequence

from playsound import playsound

# This program removes a test set from the entire data set (20%) then creates a list of 1000 randomly undersampled
# data sets. The model then trains over these 1000 train sets with a new train set after each epoch.

def main():

    # My sequence class declaration
    class CIFAR10Sequence(Sequence):

        # Here, `x_set` is list of path to the images
        # and `y_set` are the associated classes.
        # ~ My best guess is that I set x_set as a n-dimensional array with 1000 rows, each containing the information
        # ~ of one of the 1000 randomly undersampled datasets. I then set y_set as the n-dimensional array of lists of
        # ~ the proper class for the data values.
        # ~
        # ~ The value of batch size is a bit of a mystery to me. From what I can tell, it has to be between 1 and the size
        # ~ of the dataset being used as training data, and it will divide the total number of data values by the batch_size
        # ~ variable to determine how many batches there are in each epoch. I'm not sure what value to pick, however.
        # ~ It is currently set at 68, so there are 2 batches for every epoch, not sure if a lower number is better.
        def __init__(self, x_set, y_set, batch_size):
            self.x, self.y = x_set, y_set
            self.epoch = 0
            self.batch_size = batch_size

        def __len__(self):
            # return int(np.ceil(len(self.x) / float(self.batch_size)))

            # Number of batches per epoch
            return 1

        def __getitem__(self, idx):
            # batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            # batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

            batch_x = self.x[self.epoch]
            batch_y = self.y[self.epoch]

            # Not sure what to return here. The commented code below is what came with this general layout, geared
            # towards images.

            print("Batch_x shape = ", batch_x.shape, 'epoch = ', self.epoch, 'idx = ', idx, 'batch_size = ', self.batch_size)
            return np.array(batch_x), np.array(batch_y)

        def __data_generation__(self, x):
            print("called data_generation")

        # ~ This is the method called at the end of each epoch
        def on_epoch_end(self):
            # Activates after every epoch
            if self.epoch % 1 == 0:
                pass
            self.epoch += 1

    ###########################

    # ###########################
    # # Create custom callback capabilities to track what occurs during fit/evaluate/predict
    # class CustomCallback(keras.callbacks.Callback):
    #     def on_train_begin(self, logs=None):
    #         keys = list(logs.keys())
    #         print("Starting training; got log keys: {}".format(keys))
    #
    #     def on_train_end(self, logs=None):
    #         keys = list(logs.keys())
    #         print("Stop training; got log keys: {}".format(keys))
    #
    #     def on_epoch_begin(self, epoch, logs=None):
    #         keys = list(logs.keys())
    #         print("Start epoch {} of training; got log keys: {}".format(epoch, keys))
    #
    #     def on_epoch_end(self, epoch, logs=None):
    #         keys = list(logs.keys())
    #         print("End epoch {} of training; got log keys: {}".format(epoch, keys))
    #
    #     def on_test_begin(self, logs=None):
    #         keys = list(logs.keys())
    #         print("Start testing; got log keys: {}".format(keys))
    #
    #     def on_test_end(self, logs=None):
    #         keys = list(logs.keys())
    #         print("Stop testing; got log keys: {}".format(keys))
    #
    #     def on_predict_begin(self, logs=None):
    #         keys = list(logs.keys())
    #         print("Start predicting; got log keys: {}".format(keys))
    #
    #     def on_predict_end(self, logs=None):
    #         keys = list(logs.keys())
    #         print("Stop predicting; got log keys: {}".format(keys))
    #
    #     def on_train_batch_begin(self, batch, logs=None):
    #         keys = list(logs.keys())
    #         print("...Training: start of batch {}; got log keys: {}".format(batch, keys))
    #
    #     def on_train_batch_end(self, batch, logs=None):
    #         keys = list(logs.keys())
    #         print("...Training: end of batch {}; got log keys: {}".format(batch, keys))
    #
    #     def on_test_batch_begin(self, batch, logs=None):
    #         keys = list(logs.keys())
    #         print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))
    #
    #     def on_test_batch_end(self, batch, logs=None):
    #         keys = list(logs.keys())
    #         print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))
    #
    #     def on_predict_batch_begin(self, batch, logs=None):
    #         keys = list(logs.keys())
    #         print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))
    #
    #     def on_predict_batch_end(self, batch, logs=None):
    #         keys = list(logs.keys())
    #         print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))
    #
    # #####################

    #####################

    class LossAndErrorPrintingCallback(keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

        def on_test_batch_end(self, batch, logs=None):
            print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

        def on_epoch_end(self, epoch, logs=None):
            print(
                "The average loss for epoch {} is {:7.2f} "
                "and mean absolute error is {:7.2f}.".format(
                    epoch, logs["loss"], logs["mean_absolute_error"]
                )
            )

    #####################

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


    ######################################Build the Model and Call the Sequence Method #################################

    # the "result" array from undersampling method
    # Shape of arrayOfDatasets = (1000, 4) -> Array of the 1000 subsampled datasets
    # Shape of arrayOfDatasets[0] = (4, varying) -> array of the split-up datasets (train, validation, test)
    # Shape of arrayOfDatasets[0][0] = (81, 2) -> array of xTrain, the training data for this subsampled dataset
    # Shape of arrayOfDatasets[0][1] = (81,) -> Array of yTrain, the classification for the training data
    # Shape of arrayOfDatasets[0][2] = (27, 2) -> Array of xValid, the validation data for this subsampled dataset
    # Shape of arrayOfDatasets[0][3] = (27,) -> Array of yValid, the classification for the validation data
    arrayOfDatasets, testDatasetArray = undersampling(nonMasers, masers, maserCount)
    print("Shape of arrayOfDatasets[0][0] = ", arrayOfDatasets[0][0].shape)
    print("Length of arrayOfDatasets = ", len(arrayOfDatasets))
    print("Type of arrayOfDatasets = ", type(arrayOfDatasets))
    print("arrayOfDatasets[0][0][0] = ", arrayOfDatasets[0][0][0])
    print("arrayOfDatasets[0][1][0] = ", arrayOfDatasets[0][1][0])
    print("Length of testDatasetArray = ", len(testDatasetArray))
    print("TestDataSetArray: ", testDatasetArray)
    print("TestDatasetArray Type = ", type(testDatasetArray))
    print("Shape of TestDataSetArray[0]", testDatasetArray[0].shape)
    print("TestDataSetArray[0] = ", testDatasetArray[0])

    xTrain = []
    yTrain = []
    xValid = []
    yValid = []

    count = 0
    for value in arrayOfDatasets:
        xTrain.append(arrayOfDatasets[count][0])
        yTrain.append(arrayOfDatasets[count][1])
        xValid.append(arrayOfDatasets[count][2])
        yValid.append(arrayOfDatasets[count][3])
        count += 1

    batchSize = 81
    num_epochs = 1

    # This sequence receives all xTrain and yTrain values for every subset.
    sequence = CIFAR10Sequence(np.array(xTrain), np.array(yTrain), batchSize)

    # # 3-Layer Model
    # model = Sequential()
    # model.add(Dense(5, input_dim=2, activation='relu'))
    # model.add(Dense(4, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 4-Layer Model
    model = Sequential()
    model.add(Dense(6, input_dim=2, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # The error occurs here, stating:
    # "ValueError: Error when checking input: expected dense_1_input to have 2 dimensions,
    # but got array with shape (1000, 27, 2)"
    # If you look at the comments above the declaration of 'arrayOfDatasets' you will see the only array that has
    # shape of (27, 2) is the validation set. I assume this means the error comes from the
    # 'validation_data' parameter, but I am not sure what shape it wishes to have instead.
    # The documentation on the fit() method does not say to exclude the validation_data parameter when using
    # a Sequence().
    model.fit(sequence, epochs=num_epochs, verbose=1)

    # print("Shape of XTest Data = ", testDatasetArray.shape)
    # yValPred = model.predict_classes(np.array(X_val))
    yTestPred = model.predict_classes(np.array(testDatasetArray[0]))

    # Compute the accuracy of the predicted values
    sklearn_acc = metrics.accuracy_score(testDatasetArray[1], yTestPred)
    print('accuracy from sklearn is:', sklearn_acc)

    # # Compute the f1 score of the predicted values
    f1 = metrics.f1_score(testDatasetArray[1], yTestPred)
    print('f1 is:', f1)

    # # Write the output of the accuracies to file, as well as the details of the 3Layer model used
    # f = open('./NNDataSelectionV2/3LayerModel/modelStatistics.txt', 'w')
    # f.write("arrayOfDatasetsREADME:\n")
    # f.write("X[0] is all of the X datasets; (1000, 81, 2) shape\n")
    # f.write("X[1] is all of the classifications for each X set; (1000, 81, 1) shape\n")
    # f.write("X[2] is all validation X datasets\n")
    # f.write("X[3] is all Validation classifications for each X validation set\n")
    # f.write("\n")
    # f.write("Test Accuracy = %.3f\n" % sklearn_acc)
    # f.write("Test F1 Score = %.3f\n" % f1)
    # f.write("\n")
    # f.write("Model layer Description:\n")
    # f.write("Layer 1: ReLU; 5 nodes\n")
    # f.write("Layer 2: ReLU; 4 nodes\n")
    # f.write("Layer 3: Sigmoid; 1 node")
    # f.close()

    # Write the output of the accuracies to file, as well as the details of the 4Layer model used
    f = open('./NNDataSelectionV2/4LayerModel/modelStatistics.txt', 'w')
    f.write("arrayOfDatasetsREADME:\n")
    f.write("X[0] is all of the X datasets; (1000, 81, 2) shape\n")
    f.write("X[1] is all of the classifications for each X set; (1000, 81, 1) shape\n")
    f.write("X[2] is all validation X datasets\n")
    f.write("X[3] is all Validation classifications for each X validation set\n")
    f.write("\n")
    f.write("Test Accuracy = %.3f\n" % sklearn_acc)
    f.write("Test F1 Score = %.3f\n" % f1)
    f.write("\n")
    f.write("Model layer Description:\n")
    f.write("Layer 1: ReLU; 6 nodes\n")
    f.write("Layer 2: ReLU; 4 nodes\n")
    f.write("Layer 3: ReLU; 2 nodes\n")
    f.write("Layer 4: Sigmoid; 1 node")
    f.close()

    # # Save the 3Layer model
    # model.save('./NNDataSelectionV2/3LayerModel/3LayerNNModel.h5')
    # dataSetArrayStr = './NNDataSelectionV2/3LayerModel/arrayOfDatasets'
    # testDataSetXStr = './NNDataSelectionV2/3LayerModel/arrayOfTestDatasetX'
    # testDataSetYStr = './NNDataSelectionV2/3LayerModel/arrayOfTestDatasetY'
    # np.save(dataSetArrayStr, arrayOfDatasets)
    # np.save(testDataSetXStr, testDatasetArray[0])
    # np.save(testDataSetYStr, testDatasetArray[1])

    # Save the 4Layer model
    model.save('./NNDataSelectionV2/4LayerModel/4LayerNNModel.h5')
    dataSetArrayStr = './NNDataSelectionV2/4LayerModel/arrayOfDatasets'
    testDataSetXStr = './NNDataSelectionV2/4LayerModel/arrayOfTestDatasetX'
    testDataSetYStr = './NNDataSelectionV2/4LayerModel/arrayOfTestDatasetY'
    np.save(dataSetArrayStr, arrayOfDatasets)
    np.save(testDataSetXStr, testDatasetArray[0])
    np.save(testDataSetYStr, testDatasetArray[1])

def undersampling(nonMasers, masers, maserCount):
    # Returns a sequence (nd-array) of 1,000 undersampled datasets, as well as a test dataset,
    # to be fed to the neural network; one new set for each epoch to train on

    ####################################################################################################################
    ######################################## Perform Undersampling of NonMaser Data ####################################
    # Create a random sampling of test data from the nonMaser list
    # Creates a data range the size of the nonMaser dataset for undersampling purposes
    numTestNonMasers = 14
    numTestMasers = 14
    totalNonMaserLength = len(nonMasers)
    totalMaserLength = len(masers)
    print("Length of NonMasers = ", totalNonMaserLength, "; Length of Test Nonmasers = ", numTestNonMasers, "\n")

    # Two lists of 14 randomly chosen numbers used to specify the index of the chosen nonmasers and masers
    chosenNonMasers = random.sample(range(0, totalNonMaserLength), k=numTestNonMasers)
    chosenMasers = random.sample(range(0, totalMaserLength), k=numTestMasers)

    # Make 4 lists:
    # Maser Training Dataset
    # Maser Test Dataset
    # Nonmaser Training Dataset
    # Nonmaser Test Dataset

    maserTraining = []
    maserTest = []
    nonMaserTraining = []
    nonMaserTest = []

    count = 0
    for value in nonMasers:
        # Add to the list of test nonMasers
        if count in chosenNonMasers:
            nonMaserTest.append(np.array(nonMasers[count]))
        else:
            nonMaserTraining.append(nonMasers[count])
        count += 1
    print("Length of nonMaserTest = ", len(nonMaserTest))
    print("Length of nonMaser Training = ", len(nonMaserTraining))

    count = 0
    for value in masers:
        # Add to the list of test masers
        if count in chosenMasers:
            maserTest.append(np.array(masers[count]))
        # Add to the list of training masers
        else:
            maserTraining.append(masers[count])
        count += 1
    print("Length of maserTest = ", len(maserTest))
    print("Length of maserTraining = ", len(maserTraining))

    # At this point, there are lists of the test masers/nonmasers and training masers/nonmasers

    # Create the test X and y datasets by alternating the addition of masers/nonmasers from the testMaser and
    # testNonMaser datasets
    XTest = []
    yTest = []
    count = 0
    for value in maserTest:
        XTest.append(nonMaserTest[count])
        yTest.append(0)
        XTest.append(maserTest[count])
        yTest.append(1)
        count += 1
    print("Length of XTest = ", len(XTest))
    print("Shape of XTest = ", np.array(XTest).shape)
    print("Length of yTest = ", len(yTest))

    testDataSets = [np.array(XTest), np.array(yTest)]

    ################################### Choosing 1,000 undersampled data sets ##########################################

    # Run a loop to select 1,000 randomly undersampled datasets
    numOfLoops = range(0, 1000)

    # The list of dataset lists
    # Will contain 1000 rows, each with 6 columns
    # The columns will be: [XTrain, YTrain, XVal, YVal, XTest, y_test] datasets
    result = []

    for value in numOfLoops:

        # Choose a random selection of len(maserTraining) number of nonMasers from nonMaserTraining
        # Choose new random numbers each loop
        chosen = random.sample(range(0, len(nonMaserTraining)), k=len(maserTraining))
        # print("Length of Chosen = ", len(chosen))

        # Build the X training dataset for use in the neural network training based on the randomly selected nonMaser galaxies
        # ALTERNATE adding maser, non-maser to X data set
        XTrain = []
        yTrain = []
        # Create the class value list to go with the data set for accuracy testing
        Class = []
        count = 0
        for value in chosen:
            XTrain.append(nonMaserTraining[value])
            yTrain.append(0)
            XTrain.append((maserTraining[count]))
            yTrain.append(1)
            count += 1
        # print("Length of X = ", len(XTrain))
        # print("Length of Class = ", len(yTrain))

        # Implements Stratified Test Set Validation to test accuracy of NN model
        # Creates a random selection of Training and Validation data
        # Validation is 25% of Train data, which is 20% of the total data
        # Train set is now 60% of total dataset
        randNum = random.randint(0, 100)
        XTrain, XVal, yTrain, yVal = train_test_split(XTrain, yTrain, test_size=0.25, random_state=randNum)

        result.append([np.array(XTrain), np.array(yTrain), np.array(XVal), np.array(yVal)])

    # Returns an array of the 1,000 sets of training/validation data AND an array of the single test data set
    return np.array(result), testDataSets
    #####################

if __name__ == '__main__':
    main()

    #### Play a fun sound to alert me to it being done
    playsound('collect_coin_8bit.mp3')
    playsound('collect_coin_8bit.mp3')
    playsound('collect_coin_8bit.mp3')
