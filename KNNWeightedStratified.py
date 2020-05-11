import numpy as np
import sys
import random
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn import metrics

############################################ Read In Data From File ####################################################
# Read in column 0 from Table1 for the name of the galaxy
galaxyName = genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=0)

# Read in Column 6 from Table1 (Maser Classification)
maserType = genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=6)

# Read in L12 from Table1
L12 = genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=7)

# Read in Lobs from Table2
Lobs = genfromtxt(sys.argv[2], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=4)

########################################## Normalize the Data ##########################################################
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

########################################## Reshape the Data Matrix #####################################################
# Currently, the shape of the data matrix is flipped
# Reshape the data matrix to have 2 columns, one for each attribute, and as many rows as there are examples (galaxies)
data = []
count = 0
for value in L12:
    data.append([L12[count], Lobs[count]])
    count += 1

print("Length of data = ", len(data))
print("Length of Data[0]", len(data[0]))
print("Length of Data[1] = ", len(data[1]))
print("Length of MaserType[] = ", len(maserType))

########################################## Sort the Masers from the Non-Masers #########################################
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

print("Number of Masers = ", maserCount)
print("Length of Masers[] = ", len(masers))
print("Length of NonMasers[] = ", len(nonMasers))

########################################################################################################################
######################################## Perform Undersampling of NonMaser Data ########################################
# Create a random sampling of training data from the nonMaser list
# Creates a data range the size of the nonMaser dataset for undersampling purposes
upperBound = len(nonMasers)
dataRange = range(0, upperBound)


######################################## Outer Loop: Choosing Random Data ##############################################
# Chooses random data numOfIterations times to perform KNN analysis and Stratified Validation

kAverages = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # Used to graph accuracy of each k value; k = 1-15
f1Averages = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # Used to graph f1 score of each k value; k = 1-15
numOfIterations = 100
dataIterations = range(0, numOfIterations)

bestKAcc = 0
bestF1Score = 0
bestTrainSetX = []
bestTrainSetY = []
bestTestSetX = []
bestTestSetY = []
bestKVal = 0

# plt.subplot(2, 1, 1)
for num in dataIterations:
    # Choose n number of random nonMaser galaxies where n = number of Maser galaxies
    chosen = random.sample(dataRange, k=maserCount)

    # Build the X dataset for use in KNN based on the randomly selected nonMaser galaxies
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

    #################################### Inner Loop to Test Multiple K Values of KNN ###################################
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
    trainMaserListX= []
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

    # Add masers from training set to the validation maser list
    # Add nonmasers from the training set to the validation nonmaser list
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

    # Plot the masers and nonmasers from the training and validation sets and distinguish them as unique symbols
    # plt.scatter(trainMaserListX, trainMaserListY, c='orange', marker='s')
    # plt.scatter(trainNonMaserListX, trainNonMaserListY, c='cyan', marker = '^')
    # plt.scatter(valMaserListX, valMaserListY, c='purple', marker='s')
    # plt.scatter(valNonMaserListX, valNonMaserListY, c='green', marker='^')
    # plt.legend()
    # plt.savefig("MappingOfDataTest")
    # plt.show()

    kRange = range(2, 16)  # Used to test K=2 -> k=15
    kScores = []  # Usesd to keep track of the accuracy scores of k
    countK = 0

    for k in kRange:
        # Create the KNN Classifier implementing distance as a weight
        model = KNeighborsClassifier(n_neighbors=k, metric='euclidean', weights='distance')

        # Fit the model on the training set and predict the labels of the X-validation set
        y_pred = model.fit(X_train, y_train).predict(X_val)

        # print("y-pred = ", y_pred)
        # print("y-vali = ", y_val)
        # print("\n")

        # Compute the accuracy of the predicted values
        sklearn_acc = metrics.accuracy_score(y_val, y_pred)
        # print('accuracy from sklearn is:', sklearn_acc)

        # Compute the f1 score of the predicted values
        f1 = metrics.f1_score(y_val, y_pred)
        # print('f1 is:', f1)

        # Keep track of the accuracy and f1 score for each K value
        kAverages[countK] = kAverages[countK] + sklearn_acc
        f1Averages[countK] = f1Averages[countK] + f1

        # If the F1 score of this k value and training set is better than any other previous f1 value,
        # store the accuracy, f1 score, training set, test set, and k value for use later
        # Will be used to rebuild this exact model to test the test data on
        if f1 > bestF1Score:
            bestF1Score = f1
            bestKAcc = sklearn_acc
            bestTrainSetX = X_train
            bestTrainSetY = y_train
            bestTestSetX = X_test
            bestTestSetY = y_test
            bestKVal = k

        # Increase the K value to build the model on
        countK += 1

bestTrainSetX = np.load('DataTrainingXKNNUnweighted.npy')
print("bestTrainSetX[13] = ", bestTrainSetX[13])
bestTrainSetY = np.load('DataTrainingYKNNUnweighted.npy')
bestTestSetX = np.load('DataTestXKNNUnweighted.npy')
bestTestSetY = np.load('DataTestYKNNUnweighted.npy')

# Rebuild the model with the most accurate training set and k value found from above, using distance as a weight
model = KNeighborsClassifier(n_neighbors=bestKVal, metric='euclidean', weights='distance')
y_pred = model.fit(bestTrainSetX, bestTrainSetY).predict(bestTestSetX)
testAcc = metrics.accuracy_score(bestTestSetY, y_pred)
testF1 = metrics.f1_score(bestTestSetY, y_pred)

print("Number of Iterations = ", numOfIterations)
print("Best K Val = ", bestKVal)
print("Best Training Accuracy = ", bestKAcc)
print("Best Training F1 Score = ", bestF1Score)
print("Test Accuracy = ", testAcc)
print("Test F1 Score = ", testF1)

# Text to be used in the Heat Map
t = "BestKVal = {}\nBest Training F1 = {}\nTest F1 Score = {}".format(bestKVal, round(bestF1Score, 3), round(testF1, 3))

k = 0
# texts = []
for slot in kAverages:
    kAverages[k] = kAverages[k]/numOfIterations
    f1Averages[k] = f1Averages[k]/numOfIterations
    kAverages[k] = round(kAverages[k], 3)
    f1Averages[k] = round(f1Averages[k], 3)
    # if k == 6:
    #     texts.append(plt.text(k, kAverages[k], kAverages[k]))
    k += 1
print("kAverageAccuracy Per K Value:")
print(kAverages)
print("F1AverageScore Per K Value:")
print(f1Averages)

#################################### Plot the data points on a scatter plot ############################################
# plt.legend()
# plt.savefig("MappingOfDataTest")
# plt.show()

########################### Plot the average accuracy and f1 score of each k value #####################################
# plt.subplot(2, 1, 1)
# plt.plot(kRange, kAverages, label = 'Average Accuracy')
# plt.plot(kRange, f1Averages, label = 'Average F1')
# plt.savefig("BestKValue10Value")
# plt.show()

# plt.xlabel("Value of K")
# plt.ylabel("Accuracy of KNN Model")
# plt.title("Accuracy of KNN Model Over 100 Random Data Series")
# plt.legend()
# plt.savefig("KNNStratTest")
# plt.show()

######################### Plot a heat map of Accuracy of Predicting Maser Values at Given Points #######################
# Plot the test values
plt.subplot(2, 1, 1)

bestMaserListX= []
bestMaserListY = []
bestNonMaserListX = []
bestNonMaserListY = []

count = 0
for value in bestTrainSetY:
    if value == 0:
        for val in bestTrainSetX[count]:
                bestNonMaserListX.append(bestTrainSetX[count][0])
                bestNonMaserListY.append((bestTrainSetX[count][1]))
    else:
        for val in bestTrainSetX[count]:
                bestMaserListX.append(bestTrainSetX[count][0])
                bestMaserListY.append((bestTrainSetX[count][1]))
    count += 1

plt.scatter(bestMaserListX, bestMaserListY, c='orange', marker='s', label='maser')
plt.scatter(bestNonMaserListX, bestNonMaserListY, c='cyan', marker = '^', label='nonMaser')
plt.legend()

# Assign to second suplot
# This will be the colored probability mapping the probability of finding a maser at the location on the graph
plt.subplot(2, 1, 2)

# Create the x and y axis values (0 - 1 stepping by .1)
# xAxis = np.linspace(0, 1, num=11)
# yAxis = np.linspace(0, 1, num=11)

# Create the x and y axis values (0 - 1 stepping by .01)
xAxis = np.linspace(0, 1, num=101)
yAxis = np.linspace(0, 1, num=101)

# The X data set to populate and predict probability
predX = []
for x in xAxis:
    for y in yAxis:
        predX.append([x, y])

model = KNeighborsClassifier(n_neighbors=bestKVal, metric='euclidean', weights='distance')
model.fit(bestTrainSetX, bestTrainSetY)
predProb = model.predict_proba(predX)
predMaser = predProb[:,1]
# print("One Dimension: ", predMaser)
# predMaser = predMaser.reshape(11, 11)
predMaser = predMaser.reshape(101, 101)
predMaser = predMaser.transpose()
# print("After reshape to 2d: ", predMaser)
# print(type(predMaser))
# print(predMaser.shape)
# print("Predicted probabilites = ", predProb)
# print("length of predProb = ", len(predProb))
# predMaser = np.flip(predMaser, 1)
# print("After flip: ", predMaser)

plt.imshow(predMaser, origin='lower')
plt.colorbar()
plt.text(-50, 50, t, family='serif', ha='right', wrap=True)
plt.savefig('100IterWeightedProbMap100x100SameDataset')
plt.show()