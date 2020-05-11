import numpy as np
import sys
import random
import matplotlib.pyplot as plt
from adjustText import adjust_text
from sklearn.neighbors import KNeighborsClassifier
from numpy import genfromtxt
from sklearn.model_selection import cross_val_score

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

######################################## Perform Undersampling of NonMaser Data ########################################
# Create a random sampling of training data from the nonMaser list
# Creates a data range the size of the nonMaser dataset for undersampling purposes
upperBound = len(nonMasers)
dataRange = range(0, upperBound)


######################################## Outer Loop: Choosing Random Data ##############################################
# Chooses random data numOfIterations times to perform KNN analysis and K-Fold Cross Validation on

kAverages = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # Used to graph accuracy of each k value
numOfIterations = 10000
dataIterations = range(0, numOfIterations)
for num in dataIterations:
    # Choose n number of random nonMaser galaxies where n = number of Maser galaxies
    chosen = random.sample(dataRange, k=maserCount)
    # print(chosen[1])

    # # Build the X dataset for use in KNN based on the randomly selected nonMaser galaxies
    # X = []
    # # Create the class value list to go with the data set for accuracy testing
    # Class = []
    # for value in chosen:
    #     X.append(nonMasers[value])
    #     Class.append(0)
    #
    # count = 0
    # for value in range(0, int(len(masers) / 2)):
    #     X.append(masers[count])
    #     Class.append(1)
    #     count += 1

    # Build the X dataset for use in KNN based on the randomly selected nonMaser galaxies
    ####################### ALTERNATE adding maser, non-maser to X data set
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

    #################################### Inner Loop to Test Multiple K Values of KNN ###################################
    # Implements K-Fold Cross Validation to test accuracy of KNN model

    kRange = range(1,16)
    kScores = []



    countK = 0
    for k in kRange:
        # Create the KNN Classifier
        model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        # KFold Value of 5
        scores = cross_val_score(model, X, Class, cv=5, scoring='accuracy') # validate classifier using cross validation
        # print('k:', k, ' mean:',scores.mean(), ' std:', scores.std())
        kScores.append(scores.mean())
        kAverages[countK] = kAverages[countK] + scores.mean()
        countK += 1

    # label = 'series ', num
    # plt.plot(kRange, kScores, label='series ' + str(num))

k = 0
texts = []
for slot in kAverages:
    kAverages[k] = kAverages[k]/numOfIterations
    kAverages[k] = round(kAverages[k], 3)
    if k == 10:
        texts.append(plt.text(k, kAverages[k], kAverages[k]))
    k += 1
print(kAverages)


plt.plot(kRange, kAverages, label = 'Average Accuracy')
# plt.savefig("BestKValue10Value")
# plt.show()

plt.xlabel("Value of K")
plt.ylabel("Accuracy of KNN Model")
plt.title("Accuracy of KNN Model Over 10,000 Random Data Series")
plt.legend()
plt.savefig("KNNAcc_10000Iterations")
plt.show()