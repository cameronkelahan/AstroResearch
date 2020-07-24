import numpy as np
import matplotlib.pyplot as plt
import math
from statistics import mean
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, make_scorer
from numpy import genfromtxt
from scipy.stats import norm

# List 1 and 2 here should be the nonmaser and maser list pertaining to the same parameter (i.e. L12, Lx, etc.)
# List1name should be masers___ where ___ is the parameter (i.e. nonmasersL12) Used for the label parameter of plot
# xLalel should be string called 'Normalized ____', where ____ is the passed paramter (i.e. L12, Lx, etc.)
def plot(parentList, childList, parentListName, childListName, xLabel, saveName, title):
    ######################## Plot a Univariate Gaussian curve for 2 parameters

    # Calculate the Mu, variance, and sigma of parent list masers
    mu = np.mean(parentList)
    variance = np.var(parentList)
    sigma = math.sqrt(variance)

    # Create an x-axis based on the mu and sigma values
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)

    # Plot the gaussian curve of the parent
    plt.plot(x, norm.pdf(x, mu, sigma), label=parentListName, linestyle='solid')

    # calculate the mu, sigma, x values, and variance of the child list
    mu = np.mean(childList)
    variance = np.var(childList)
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)

    # Plot the gaussian curve of the child on top of the parent gaussian plot
    plt.plot(x, norm.pdf(x, mu, sigma), label=childListName, linestyle='dashed')

    # Use Sturge's Rule to calculate correct number of bins based on data size of child data set and parent data set
    # Cast to int for a whole number
    numBinsChild = int(1 + 3.322 * math.log10(len(childList)))
    numBinsParent = int(1 + 3.322 * math.log10(len(parentList)))

    # Create a Histogram of the child and parent sets' attribute; plot over gaussians
    # Plot over the gaussian curves too
    # Alpha of 0.5 for translucence
    plt.hist(childList, numBinsChild, density=True, alpha=0.5, edgecolor='black', linewidth=1.5,label='ChildHist')
    plt.hist(parentList, numBinsParent, density=True, alpha=0.5, edgecolor='red', linewidth=1.5, label='ParentHist',
             linestyle='dashed')

    plt.xlabel(xLabel)
    plt.title(title)
    # Unicode for Delta symbol
    plt.ylabel('$dN/N_{total}$')
    plt.legend()

    plt.legend()
    plt.savefig(saveName, dpi=400, bbox_inches='tight', pad_inches=0.05)
    plt.show()

    plt.close()


def main():
    ############################################ Read In Data From File ####################################################
    # Read in column 0 from Table1 for the name of the galaxy
    galaxyName = genfromtxt('Paper2Table1.csv', delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=0)

    # Read in Column 6 from Table1 (Maser Classification)
    maserType = genfromtxt('Paper2Table1.csv', delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=6)

    # Read in L12 from Table1
    L12 = genfromtxt('Paper2Table1.csv', delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=7)

    # Read in Lobs from Table2
    Lobs = genfromtxt('Paper2Table2.csv', delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=4)

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

    ################################## Lists of Masers/Nonmasers per Attribute #############################################
    # Make lists of all L12 values for masers and nonmasers
    # Make lists of all LX values for masers and nonmasers
    masersL12 = []
    nonMasersL12 = []
    masersLX = []
    nonMasersLX = []

    # Add all L12 and LX values from list of masers to respective lists
    for value in masers:
        masersL12.append(value[0])
        masersLX.append(value[1])

    # Add all L12 and LX values from list of nonmasers to respective lists
    for value in nonMasers:
        nonMasersL12.append(value[0])
        nonMasersLX.append(value[1])

    ######################################## Randomly Select a Test and Training Data Set ##################################
    # The state of the split provides reproduceability
    # Number of splits is only 1, as we only want it to split into 1 training and 1 test set
    # .2 signifies 20% of the parent data set; used as the test data set
    # This splitting preserves the proportionality of masers to nonmasers from the parent data set
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=3)
    # random_state 3 was chosen to offer the most accurate distribution among the L12 and LX attributes

    # Extract the chosen data value indices for the training and test data sets form the split
    for train_index, test_index in sss.split(data, maserType):
        print("TRAIN:", train_index, "\nTEST:", test_index)
        # X_train, X_test = data[train_index], data[test_index]
        # y_train, y_test = maserType[train_index], maserType[test_index]
        # print("X_train: ", X_train)
        # print("Y_Train: ", y_train)

    X_train = []
    y_train = []

    L12_train_maser = []
    L12_train_nonmaser = []
    LX_train_nonmaser = []
    LX_train_maser = []

    X_test = []
    y_test = []

    L12_test_maser = []
    L12_test_nonmaser = []
    LX_test_maser = []
    LX_test_nonmaser = []


    # Add chosen data values to the training set, based on the train_index values
    # Also separate the chosen values for each data set into lists of L12 and LX values for maser and nonmaser
    # Used to plot the distribution of chosen L12 and LX values to compare with the parent data set's distribution
    count = 0
    for value in train_index:
        X_train.append(data[value])
        y_train.append(maserType[value])
        if maserType[value] == 0:
            L12_train_nonmaser.append(data[value][0])
            LX_train_nonmaser.append(data[value][1])
        else:
            L12_train_maser.append(data[value][0])
            LX_train_maser.append(data[value][1])

    for value in test_index:
        X_test.append(data[value])
        y_test.append(maserType[value])
        if maserType[value] == 0:
            L12_test_nonmaser.append(data[value][0])
            LX_test_nonmaser.append(data[value][1])
        else:
            L12_test_maser.append(data[value][0])
            LX_test_maser.append(data[value][1])

    # ################################# Plot the TRAINING Gaussian/Histogram Plots ###########################################
    # # MASER L12 TRAINING
    # # Compares parent's (masersL12) maser L12 values with child's (L12_train_maser) L12 maser values on plot
    # plot(parentList=masersL12, childList=L12_train_maser, parentListName='Parent L12 Maser',
    #      childListName='Child L12 Maser', xLabel='L12', saveName='ChosenTrainL12MaserParentChildGaussHist.pdf',
    #      title='Training L12 Maser Distribution: Parent vs. Child')
    #
    # # NONMASER L12 TRAINING
    # # Compares parent's (nonMasersL12) nonmaser L12 values with child's (L12_train_nonmaser) L12 nonmaser values on plot
    # plot(parentList=nonMasersL12, childList=L12_train_nonmaser, parentListName='Parent L12 Nonmaser',
    #      childListName='Child L12 Nonmaser', xLabel='L12', saveName='ChosenTrainL12NonMaserParentChildGaussHist.pdf',
    #      title='Training L12 Nonmaser Distribution: Parent vs. Child')
    #
    # # MASER LX TRAINING
    # # Compares parent's (masersLX) maser LX values with child's (LX_train_maser) LX maser values on plot
    # plot(parentList=masersLX, childList=LX_train_maser, parentListName='Parent LX Maser',
    #      childListName='Child LX Maser', xLabel='LX', saveName='ChosenTrainLXMaserParentChildGaussHist.pdf',
    #      title='Training LX Maser Distribution: Parent vs. Child')
    #
    # # NONMASER LX TRAINING
    # # Compares parent's (nonMasersLX) nonmaser LX values with child's (LX_train_nonmaser) LX nonmaser values on plot
    # plot(parentList=nonMasersLX, childList=LX_train_nonmaser, parentListName='Parent LX Nonmaser',
    #      childListName='Child LX Nonmaser', xLabel='LX', saveName='ChosenTrainLXNonMaserParentChildGaussHist.pdf',
    #      title='Training LX Nonmaser Distribution: Parent vs. Child')
    #
    # ################################# Plot the TEST Gaussian/Histogram Plots ###########################################
    # # MASER L12 TEST
    # # Compares parent's (masersL12) maser L12 values with child's (L12_test_maser) L12 maser values on plot
    # plot(parentList=masersL12, childList=L12_test_maser, parentListName='Parent L12 Maser',
    #      childListName='Child L12 Maser', xLabel='L12', saveName='ChosenTestL12MaserParentChildGaussHist.pdf',
    #      title='Test L12 Maser Distribution: Parent vs. Child')
    #
    # # NONMASER L12 TEST
    # # Compares parent's (nonMasersL12) nonmaser L12 values with child's (L12_test_nonmaser) L12 nonmaser values on plot
    # plot(parentList=nonMasersL12, childList=L12_test_nonmaser, parentListName='Parent L12 Nonmaser',
    #      childListName='Child L12 Nonmaser', xLabel='L12', saveName='ChosenTestL12NonMaserParentChildGaussHist.pdf',
    #      title='Test L12 Nonmaser Distribution: Parent vs. Child')
    #
    # # MASER LX TEST
    # # Compares parent's (masersLX) maser LX values with child's (LX_test_maser) LX maser values on plot
    # plot(parentList=masersLX, childList=LX_test_maser, parentListName='Parent LX Maser',
    #      childListName='Child LX Maser', xLabel='LX', saveName='ChosenTestLXMaserParentChildGaussHist.pdf',
    #      title='Test LX Maser Distribution: Parent vs. Child')
    #
    # # NONMASER LX TEST
    # # Compares parent's (nonMasersLX) nonmaser LX values with child's (LX_test_nonmaser) LX nonmaser values on plot
    # plot(parentList=nonMasersLX, childList=LX_test_nonmaser, parentListName='Parent LX Nonmaser',
    #      childListName='Child LX Nonmaser', xLabel='LX', saveName='ChosenTestLXNonMaserParentChildGaussHist.pdf',
    #      title='Test LX Nonmaser Distribution: Parent vs. Child')

    ############################## Build Unw KNN Model Testing Multiple Values of K ####################################
    # Tests K values 1-30 inclusively on the chosen data set
    # Uses stratified k-fold cross validation to ensure all items are used as both valdiation and training data
    # Splits up into as many data points that there are, making the validation data only 1 point
    f1List = []
    maxF1 = 0
    maxK = 0
    for k in range(1,31):
        model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        # Uses 'Leave-One-Out' methodology; will split into as many sections as there is data
        # Makes only one data point at a time the test data set
        # Cannot do that; number of splits cannot be greater than the number of members in each class; stick to 5
        f1 = cross_val_score(model, X_train, y_train, cv=5, scoring=make_scorer(f1_score),verbose=0)
        print("F1 List: ", f1)
        f1 = mean(f1)
        print("F1 Mean: ", f1)
        # Append F1 score to the f1List which keeps track of which K gets what F1 score
        f1List.append(f1)

        if maxF1 < f1:
            maxF1 = f1
            maxK = k

    print("F1 List: ", f1List)
    print('Max F1: ', maxF1, " | Max K: ", maxK)

if __name__ == '__main__':
    main()
