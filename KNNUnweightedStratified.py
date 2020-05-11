import numpy as np
import sys
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from adjustText import adjust_text
from sklearn.neighbors import KNeighborsClassifier
from numpy import genfromtxt
from sklearn.model_selection import cross_val_score
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

if len(data) == 642 and len(data[0]) == 2 and len(maserType) == 642:
    print("Data loaded properly")
else:
    exit("Data loaded improperly")

print("Length of data = ", len(data))
print("Length of Data[0]", len(data[0]))
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

if len(masers) == 68 and len(nonMasers) == 574:
     print("Total Masers and NonMasers Separated Correctly")
     print("Number of Total Maser Galaxies = ", len(masers))
     print("Number of Total NonMaser Galaxies = ", len(nonMasers))
else:
    exit("Maser and NonMaser Separation Error")

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
numOfIterations = 1000
print("Number of Iterations This Run = ", numOfIterations)
dataIterations = range(0, numOfIterations)

bestKAcc = 0
bestF1Score = 0
bestTrainSetX = []
bestTrainSetY = []
bestValidationSetX = []
bestValidationSetY = []
bestTestSetX = []
bestTestSetY = []
bestKVal = 0

# plt.subplot(2, 1, 1)
for num in dataIterations:
    if (num % (numOfIterations/10)) == 0:
        print("Iteration Number ", num)
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

    kRange = range(2, 16) # Used to test K=2 -> k=15
    kScores = [] # Usesd to keep track of the accuracy scores of k
    countK = 0
    for k in kRange:
        # Create the KNN Classifier
        model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')

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

# print("bestTrainSetX = ", bestTrainSetX)

# Rebuild the model with the most accurate training set and k value found from above
model = KNeighborsClassifier(n_neighbors=bestKVal, metric='euclidean')
y_pred = model.fit(bestTrainSetX, bestTrainSetY).predict(bestTestSetX)
testAcc = metrics.accuracy_score(bestTestSetY, y_pred)
testF1 = metrics.f1_score(bestTestSetY, y_pred)

# # Rebuild the model by loading in the data from file
# bestTrainSetX2 = np.load('DataTrainingXKNNUnweighted.npy')
# bestTrainSetY2 = np.load('DataTrainingYKNNUnweighted.npy')
# bestTestSetX2 = np.load('DataTestXKNNUnweighted.npy')
# bestTestSetY2 = np.load('DataTestYKNNUnweighted.npy')
# model = KNeighborsClassifier(n_neighbors=3, metric='euclidian')
# y_pred2 = model.fit(bestTrainSetX2, bestTrainSetY2).predict(bestTestSetX2)
# testF12 = metrics.f1_score(bestTestSetY2, y_pred2)

# unwTestF1Scores = []
# for kval in kRange:
#     model = KNeighborsClassifier(n_neighbors=kval, metric='euclidian')
#     forLoopUnwY_pred = model.fit(bestTrainSetX, bestTrainSetY).predict(bestTestSetX)
#     forLoopUnwTestF1 = metrics.d1_score(bestTestSetY, forLoopUnwY_pred)
#     unwTestF1Scores.append(forLoopUnwTestF1)

# strBestKValue = 'Best K-Value = ', bestKVal
# strBestUnwTrainF1 = ' Best Unweighted Training F1 Score = {} '.format(round(bestF1Score, 3))
# strBestUnwTestF1 = 'Unweighted Test F1 Score = {}'.format(round(testF1, 3))
f = open('bestUnwDataStats.txt', 'w')
f.write("Best K-Value = %i\n" % bestKVal)
f.write("Best Unweighted Accuracy = %.3f\n" % bestKAcc)
f.write("Best Unweighted Training F1 Score = %.3f\n" % bestF1Score)
f.write("Best Unweighted Test Accuracy = %.3f\n" % testAcc)
f.write("Best Unweighted Test F1 Score = %.3f" % testF1)
f.close()
# text_file = open("bestUnwDataStats.txt", "w")
# n = text_file.write('Best K-Value = ', bestKVal)
# m = text_file.write(strBestUnwTrainF1)
# l = text_file.write(strBestUnwTestF1)
# text_file.close()
# bestDataUnweightedArray = np.array([strBestKValue, strBestUnwTrainF1, strBestUnwTestF1])
# print(bestDataUnweightedArray)

# Constant names for the long string names of the saved .npy files
xUnwDataTrain = 'DataTrainingXKNNUnweighted'
yUnwDataTrain = 'DataTrainingYKNNUnweighted'
xUnwDataVal = 'DataValidationXKNNUnweighted'
yUnwDataVal = 'DataValidationYKNNUnweighted'
xUnwDataTest = 'DataTestXKNNUnweighted'
yUnwDataTest = 'DataTestYKNNUnweighted'
bestUnwDataStats = 'BestUnwDataSetStats'
bestWeightDataStats = 'BestWeiDataSetStats'

np.save(xUnwDataTrain, bestTrainSetX)
np.save(yUnwDataTrain, bestTrainSetY)
np.save(xUnwDataVal, X_val)
np.save(yUnwDataVal, y_val)
np.save(xUnwDataTest, bestTestSetX)
np.save(yUnwDataTest, bestTestSetY)

print("Number of Iterations = ", numOfIterations)
print("Best K Val = ", bestKVal)
print("Best Training Accuracy = ", bestKAcc)
print("Best Training F1 Score = ", bestF1Score)
print("Test Accuracy = ", testAcc)
print("Test F1 Score = ", testF1)

# Text to be used in the Heat Map
t = "BestKVal = {}\nBest Training F1 = {}\nTest F1 Score = {}".format(bestKVal, round(bestF1Score, 3), round(testF1, 3))

k = 0
texts = []
for slot in kAverages:
    kAverages[k] = kAverages[k]/numOfIterations
    f1Averages[k] = f1Averages[k]/numOfIterations
    kAverages[k] = round(kAverages[k], 3)
    f1Averages[k] = round(f1Averages[k], 3)
    if k == bestKVal:
        texts.append(plt.text(k, kAverages[k], kAverages[k]))
    k += 1
print("kAverageAccuracy Per K Value:")
print(kAverages)
print("F1AverageScore Per K Value:")
print(f1Averages)

########################### Plot the average unweighted f1 score of each k value #####################################
plt.figure(figsize=(6.4, 4.8))
plt.plot(kRange, kAverages, label = 'Average Accuracy')
plt.plot(kRange, f1Averages, label = 'Average F1')
plt.xlabel("Value of K")
plt.ylabel("Accuracy of KNN Model")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title("Average Accuracy and F1 Score of KNN For Each K, Over 10,000 Random Data Series")
plt.legend()
plt.savefig("KNNAccUnwStrat1000.pdf", dpi=400, bbox_inches='tight', pad_inches=0.05)
plt.show()
plt.clf()
plt.close()

######################### Plot a heat map of Accuracy of Predicting Maser Values at Given Points #######################
# Plot the test values
# plt.subplot(2, 1, 1)

bestMaserListX= []
bestMaserListY = []
bestNonMaserListX = []
bestNonMaserListY = []

# # Values used to make scatter plot
# # Currently Using training set... Should it be the test set? Or validation set?
# # Purpose of scatter plot is...
# count = 0
# for value in bestTrainSetY:
#     if value == 0:
#         for val in bestTrainSetX[count]:
#                 bestNonMaserListX.append(bestTrainSetX[count][0])
#                 bestNonMaserListY.append((bestTrainSetX[count][1]))
#     else:
#         for val in bestTrainSetX[count]:
#                 bestMaserListX.append(bestTrainSetX[count][0])
#                 bestMaserListY.append((bestTrainSetX[count][1]))
#     count += 1
#
# plt.figure(figsize=(6.4, 4.8))
# plt.scatter(bestMaserListX, bestMaserListY, c='orange', marker='s', label='maser')
# plt.scatter(bestNonMaserListX, bestNonMaserListY, c='cyan', marker = '^', label='nonMaser')
# plt.legend()
# plt.show()
# plt.savefig('1000IterUnweightedScatterPlot.pdf', dpi=400, bbox_inches='tight', pad_inches=0.05)
# plt.clf()
# plt.close()

# Create a scatter plot of all galaxies in the data set, showing masers and non masers as unique shapes/colors
maserL12 = []
maserLobs = []
nonMaserL12 = []
nonMaserLobs = []
count = 0
for value in data:
    if maserType[count] > 0:
        maserL12.append(data[count][0])
        maserLobs.append(data[count][1])
        count += 1
    else:
        nonMaserL12.append(data[count][0])
        nonMaserLobs.append(data[count][1])
        count+=1
print('Length of MaserL12 = ', len(maserL12))
print('Length of MaserLobs = ', len(maserLobs))
print('Length of NonMaserL12 = ', len(nonMaserL12))
print('Length of NonMaserLobs = ', len(nonMaserLobs))
plt.figure(figsize=(6.4, 4.8))
plt.scatter(maserL12, maserLobs, c='orange', marker='s', label='maser')
plt.scatter(nonMaserL12, nonMaserLobs, c='cyan', marker='^', label='nonMaser', alpha=.2)
plt.xlabel('L12',fontsize=16)
plt.ylabel('Lx',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend()
plt.savefig('AllDataScatterPlot.pdf', dpi=400, bbox_inches='tight', pad_inches=0.05)
plt.show()
plt.clf()
plt.close()

######################## Heat Map Creation Now Done in UnwAndWeiKNNFromFile ############################################

# Create the x and y axis values (0 - 1 stepping by .1)
# xAxis = np.linspace(0, 1, num=11)
# yAxis = np.linspace(0, 1, num=11)

# # Create the x and y axis values (0 - 1 stepping by .01)
# xAxis = np.linspace(0, 1, num=101)
# yAxis = np.linspace(0, 1, num=101)
#
# # The X data set to populate and predict probability
# predX = []
# for x in xAxis:
#     for y in yAxis:
#         predX.append([x, y])
#
# model = KNeighborsClassifier(n_neighbors=bestKVal, metric='euclidean')
# model.fit(bestTrainSetX, bestTrainSetY)
# predProb = model.predict_proba(predX)
# predMaser = predProb[:,1]
# predMaser = predMaser.reshape(101, 101)
# predMaser = predMaser.transpose()
#
# # Plot the heat map for the Unweighted KNN model
# plt.figure(figsize=(6.4, 4.8))
# plt.imshow(predMaser, origin='lower', extent=[0,1,0,1])
# plt.xticks(np.arange(.2, 1.1, step=0.2))  # Set label locations
# cbar = plt.colorbar()
# cbar.ax.tick_params(labelsize=16)
# cbar.set_label('Unweighted Predicted Prob of Maser', fontsize=16)
# cbar.set_clim(0,1)
# plt.xlabel('L12',fontsize=16)
# plt.ylabel('Lx',fontsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# # save as a PDF
# plt.savefig('1000IterUnweightedProbMap100x100SameDataset.pdf', dpi=400, bbox_inches='tight', pad_inches=0.05)
# plt.show()
# plt.clf()
# plt.close()



############################# Below Here is plot to created Unweighted KNN Map after loading .npy files



############################# TEMP Below here is plot to create Weighted KNN MAP after loading .npy files ##############
bestTrainSetX1 = np.load('DataTrainingXKNNUnweighted.npy')
bestTrainSetY1 = np.load('DataTrainingYKNNUnweighted.npy')
bestTestSetX1 = np.load('DataTestXKNNUnweighted.npy')
bestTestSetY1 = np.load('DataTestYKNNUnweighted.npy')

# Rebuild the model with the most accurate training set and k value found from above, using distance as a weight
model = KNeighborsClassifier(n_neighbors=bestKVal, metric='euclidean', weights='distance')
y_pred = model.fit(bestTrainSetX1, bestTrainSetY1).predict(bestTestSetX1)
testAcc1 = metrics.accuracy_score(bestTestSetY1, y_pred)
testF11 = metrics.f1_score(bestTestSetY1, y_pred)

# weightedTestF1Scores = []
# for kval in kRange:
#     model = KNeighborsClassifier(n_neighbors=kval, metric='euclidean', weights='distance')
#     forLoopWeiY_pred = model.fit(bestTrainSetX1, bestTrainSetY1).predict(bestTestSetX1)
#     forLoopUnwTestF11 = metrics.d1_score(bestTestSetY1, forLoopWeiY_pred)
#     weightedTestF1Scores.append(forLoopUnwTestF11)

# strBestKValue = 'Best K-Value = ', bestKVal
# strBestWeiTestF1 = ' Weighted Test F1 Score = {}'.format(round(testF11, 3))
f = open('bestWeightDataStats.txt', 'w')
f.write("Best K-Value = %i\n" % (bestKVal))
f.write("Best Weighted Test Accuracy = %.3f\n" % testAcc1)
f.write("Best Weighted Test F1 Score = %.3f" % testF11)
f.close()
# text_file = open("bestWeightDataStats.txt", "w")
# n = text_file.write(strBestKValue)
# m = text_file.write(strBestWeiTestF1)
# text_file.close()
# bestDataWeightedArray = np.array([strBestKValue, strBestWeiTestF1])
# print(bestDataWeightedArray)
# np.savetxt(bestWeightDataStats, bestDataWeightedArray)

print("Number of Iterations = ", numOfIterations)
print("Best K Val = ", bestKVal)
# print("Best Training Accuracy = ", bestKAcc)
# print("Best Training F1 Score = ", bestF1Score)
print("Test Accuracy1 = ", testAcc1)
print("Test F1 Score1 = ", testF11)

# Text to be used in the Heat Map
t = "BestKVal = {}\nTest F1 Score = {}".format(bestKVal, round(testF11, 3))

# k = 0
# # texts = []
# for slot in kAverages:
#     kAverages[k] = kAverages[k]/numOfIterations
#     f1Averages[k] = f1Averages[k]/numOfIterations
#     kAverages[k] = round(kAverages[k], 3)
#     f1Averages[k] = round(f1Averages[k], 3)
#     # if k == 6:
#     #     texts.append(plt.text(k, kAverages[k], kAverages[k]))
#     k += 1
# print("kAverageAccuracy Per K Value:")
# print(kAverages)
# print("F1AverageScore Per K Value:")
# print(f1Averages)

########################### Plot the unweighted and weighted f1 scores of each k value #################################
# plt.figure(figsize=(6.4, 4.8))
# plt.plot(kRange, unwTestF1Scores, label = 'Unweighted F1 Test Scores Per K Value')
# plt.plot(kRange, weightedTestF1Scores, label = 'Weighted F1 Test Scores Per K Value')
# plt.xlabel("Value of K")
# plt.ylabel("F1 Score of KNN Model")
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.title("Comparison Of F1 Scores Between Weighted and Unweighted Models for Values of K: 2-15")
# plt.legend()
# plt.savefig("UnwAndWeiTestF1ScoreValuesEveryK.pdf", dpi=400, bbox_inches='tight', pad_inches=0.05)
# plt.show()
# plt.clf()
# plt.close()

######################### Plot a heat map of Accuracy of Predicting Maser Values at Given Points #######################
# Plot the test values
# plt.subplot(2, 1, 1)

# bestMaserListX1 = []
# bestMaserListY1 = []
# bestNonMaserListX1 = []
# bestNonMaserListY1 = []
#
# count = 0
# for value in bestTrainSetY1:
#     if value == 0:
#         for val in bestTrainSetX1[count]:
#                 bestNonMaserListX1.append(bestTrainSetX1[count][0])
#                 bestNonMaserListY1.append((bestTrainSetX1[count][1]))
#     else:
#         for val in bestTrainSetX1[count]:
#                 bestMaserListX1.append(bestTrainSetX1[count][0])
#                 bestMaserListY1.append((bestTrainSetX1[count][1]))
#     count += 1

# #Scatterplot of Training Set used
# plt.figure(figsize=(6.4, 4.8))
# plt.scatter(bestMaserListX1, bestMaserListY1, c='orange', marker='s', label='maser')
# plt.scatter(bestNonMaserListX1, bestNonMaserListY1, c='cyan', marker = '^', label='nonMaser')
# plt.legend()
# plt.show()
# #plt.savefig('1000IterWeightedScatterPlot.pdf', dpi=400, bbox_inches='tight', pad_inches=0.05)
# plt.clf()
# plt.close()

################################ Weighted Heat Map Creation moved to UnwAndWaiKNNFromFile ##############################

# Create the x and y axis values (0 - 1 stepping by .1)
# xAxis = np.linspace(0, 1, num=11)
# yAxis = np.linspace(0, 1, num=11)

# # Create the x and y axis values (0 - 1 stepping by .01)
# xAxis = np.linspace(0, 1, num=101)
# yAxis = np.linspace(0, 1, num=101)
#
# # The X data set to populate and predict probability
# predX = []
# for x in xAxis:
#     for y in yAxis:
#         predX.append([x, y])
#
# model = KNeighborsClassifier(n_neighbors=bestKVal, metric='euclidean', weights='distance')
# model.fit(bestTrainSetX1, bestTrainSetY1)
# predProb = model.predict_proba(predX)
# predMaser = predProb[:,1]
# # print("One Dimension: ", predMaser)
# # predMaser = predMaser.reshape(11, 11)
# predMaser = predMaser.reshape(101, 101)
# predMaser = predMaser.transpose()
# # print("After reshape to 2d: ", predMaser)
# # print(type(predMaser))
# # print(predMaser.shape)
# # print("Predicted probabilites = ", predProb)
# # print("length of predProb = ", len(predProb))
# # predMaser = np.flip(predMaser, 1)
# # print("After flip: ", predMaser)
#
# plt.figure(figsize=(6.4, 4.8))
# plt.imshow(predMaser, origin='lower', extent=[0,1,0,1])
# plt.xticks(np.arange(.2, 1.1, step=0.2))  # Set label locations
# cbar = plt.colorbar()
# cbar.ax.tick_params(labelsize=16)
# cbar.set_label('Weighted Predicted Prob of Maser', fontsize=16)
# cbar.set_clim(0,1)
# # plt.text(-10, 50, t, family='serif', ha='right', wrap=True)
# plt.xlabel('L12',fontsize=16)
# plt.ylabel('Lx',fontsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# # Save as a PDF
# plt.savefig('1000IterWeightedProbMap100x100SameDataset.pdf', dpi=400, bbox_inches='tight', pad_inches=0.05)
# plt.show()
# plt.clf()
# plt.close()