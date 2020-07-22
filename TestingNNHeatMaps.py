import numpy as np
import sys
from keras.callbacks import ModelCheckpoint, CSVLogger, LambdaCallback

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn import metrics
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm

# List 1 and 2 here should be the nonmaser and maser list pertaining to the same parameter (i.e. L12, Lx, etc.)
# List1name should be masers___ where ___ is the parameter (i.e. nonmasersL12) Used for the label parameter of plot
# xBalel should be string called 'Normalized ____', where ____ is the passed paramter (i.e. L12, Lx, etc.)
def plot(list1, list2, list1name, list2name, xLabel, saveName, title):
    ######################## Plot a Univariate Gaussian curve for 2 parameters

    # Calculate the Mu, variance, and sigma of parameter 1 nonmaser
    mu = np.mean(list1)
    variance = np.var(list1)
    sigma = math.sqrt(variance)

    # Create an x-axis based on the mu and sigma values
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)

    # Use Sturge's Rule to calculate correct number of bins based on data size of nonmaser dataset
    # Cast to int for a whole number
    # For this dataset, numBins = 10
    numBins = int(1 + 3.322 * math.log10(len(list2)))

    # Create a Histogram of nonMasers and Masers using their attributes
    # Plot over the gaussian curves too
    # Separate into 10 bins
    # Alpha of 0.5 for translucence
    plt.hist(list1, numBins, density=True, alpha=0.5, edgecolor='black', linewidth=1.5, label='NonMasers')
    plt.hist(list2, numBins, density=True, alpha=0.5, edgecolor='black', linewidth=1.5, label='Masers',
             linestyle='dashed')

    # Plot the curve
    plt.plot(x, norm.pdf(x, mu, sigma), label=list1name, linestyle='dotted')
    plt.xlabel(xLabel)
    plt.title(title)
    # Unicode for Delta symbol
    plt.ylabel('$\u0394N/N_{total}$')
    plt.legend()

    # Plot a Univariate Gaussian curve for parameter 1 maser on top of the nonMaser curve
    mu = np.mean(list2)
    variance = np.var(list2)
    sigma = math.sqrt(variance)

    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)

    # Plot the maser L12 curve on top of the nonmaser L12 curve
    plt.plot(x, norm.pdf(x, mu, sigma), label=list2name, linestyle='dashdot')
    plt.legend()
    plt.savefig(saveName, dpi=400, bbox_inches='tight', pad_inches=0.05)
    plt.show()

    plt.close()
###############

def splitData(dataSet, classSet, attributeNum):
    nonmasers = []
    masers = []

    count = 0
    for value in dataSet:
        if classSet[count] == 0:
            nonmasers.append(value[attributeNum])
        else:
            masers.append(value[attributeNum])
        count += 1
    print("Nonmasers loop: ", nonmasers)
    print("masers loop: ", masers)

    return nonmasers, masers

######################################### Model Training ###############################################################
# 3-Layer Model
model = Sequential()
model.add(Dense(5, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # 4-Layer Model
# model = Sequential()
# model.add(Dense(6, input_dim=2, activation='relu'))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(2, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

num_epochs = 1000
np.allow_pickle=True
# datasets3Layer = np.load('./NNDataSelectionV2/3LayerModelF182/arrayOfDatasets.npy')
# trainX = datasets3Layer[0][0][0]
# trainY = datasets3Layer[0][1][0]

trainX = np.load('./DataSetNN_3Layer/DataTrainingXNN.npy')
print("Shape of trainX: ", trainX.shape)
print("Train X L12 slot 0: ", trainX[0])
print("Train X L12 slot 1: ", trainX[1])
trainY = np.load('./DataSetNN_3Layer/DataTrainingYNN.npy')
print(trainY.shape)
valX = np.load('./DataSetNN_3Layer/DataValidationXNN.npy')
valY = np.load('./DataSetNN_3Layer/DataValidationYNN.npy')
testX = np.load('./DataSetNN_3Layer/DataTestXNN.npy')
testY = np.load('./DataSetNN_3Layer/DataTestYNN.npy')
model.fit(trainX, trainY, epochs=num_epochs, validation_data=(valX, valY), batch_size=100, verbose=0)
testPred = model.predict_classes(testX)
F1 = metrics.f1_score(testY, testPred)
print("F1 score = ", F1)

# model82F1_3Layer = load_model('./NNDataSelectionV2/3LayerModelF182/3LayerNNModel.h5')
# model84F1_4Layer = load_model('./NNDataSelectionV2/4LayerModelF184/4LayerNNModel.h5')

################# 3 Layer NN Heat Map
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

# print("Type of PredX: ", type(predX))
# print("Type of PredX[0]: ", type(predX[0]))
# print("PredX: ", np.array(predX).shape)

predProb = model.predict_proba(np.array(predX))
predClass = model.predict(np.array(predX))

# count = 0
# for value in predProb:
    # print("Pred Prob value = ", predProb[count])
    # print("Predicted Class = ", predClass[count])
    # count += 1
# print("PredProb, predClass: ", [predProb, predClass])

# # Unused with the NN predict class
# predMaser = predProb[:,1]

# predMaser = predMaser.reshape(11, 11)
predMaser = predProb.reshape(101, 101)
predMaser = predMaser.transpose()


plt.figure(figsize=(6.4, 4.8))
plt.imshow(predMaser, origin='lower', extent=[0,1,0,1])
plt.xticks(np.arange(.2, 1.1, step=0.2))  # Set label locations
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Weighted Predicted Prob of Maser', fontsize=16)
plt.clim(vmin=0, vmax=1)
cbar.set_clim(0,1)
# plt.text(-10, 50, t, family='serif', ha='right', wrap=True)
# plt.title("3-Layer NN Maser Classification Probability Heat Map")
plt.xlabel('L12',fontsize=16)
plt.ylabel('Lx',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# Save as a PDF
# plt.savefig('NNHeatMapsTest/heatMap3LayerV6_.pdf', dpi=400, bbox_inches='tight', pad_inches=0.05)
plt.show()
plt.clf()
plt.close()

########################################################################################################################

# Plot the gaussian distributions and histograms over each other for the subsets used for
# training, validation, and testing

# Create lists for nonMasers and masers in their attributes (L12, LX) for the trianing, test, and validation sets
# Used to plot the gaussian and histogram distributions of the data
nonMasersL12Train, masersL12Train = splitData(trainX, trainY, 0)
nonMasersLXTrain, masersLXTrain = splitData(trainX, trainY, 1)

nonMasersL12Val, masersL12Val = splitData(valX, valY, 0)
nonMasersLXVal, masersLXVal = splitData(valX, valY, 1)

nonMasersL12Test, masersL12Test = splitData(testX, testY, 0)
nonMasersLXTest, masersLXTest = splitData(testX, testY, 1)

########
# Plot the gaussian and histogram figures for each subset of data to see how the distribution of values compares
# with the entire data set

# Plot a Univariate Gaussian curve on top of a histogram plot for L12
# comparing masers and nonmasers for the training dataset
plot(nonMasersL12Train, masersL12Train, 'nonMasersL12', 'masersL12', 'Normalized $L_{12}$',
     'MasersAndNonMasersGaussianHistoL12Train.pdf', 'Training Set L12 Distribution')

# Plot a Univariate Gaussian curve on top of a histogram plot for LX
# comparing masers and nonmasers for the training dataset
plot(nonMasersLXTrain, masersLXTrain, 'nonMasersLX', 'masersLX', 'Normalized $L_{X}$',
     'MasersAndNonMasersGaussianHistoLXTrain.pdf', 'Training Set LX Distribution')

# Plot a Univariate Gaussian curve on top of a histogram plot for L12
# comparing masers and nonmasers for the validation dataset
plot(nonMasersL12Val, masersL12Val, 'nonMasersL12', 'masersL12', 'Normalized $L_{12}$',
     'MasersAndNonMasersGaussianHistoL12Val.pdf', 'Validation Set L12 Distribution')

# Plot a Univariate Gaussian curve on top of a histogram plot for LX
# comparing masers and nonmasers for the validation dataset
plot(nonMasersLXVal, masersLXVal, 'nonMasersLX', 'masersLX', 'Normalized $L_{X}$',
     'MasersAndNonMasersGaussianHistoLXVal.pdf', 'Validation Set LX Distribution')

# Plot a Univariate Gaussian curve on top of a histogram plot for L12
# comparing masers and nonmasers for the test dataset
plot(nonMasersL12Test, masersL12Test, 'nonMasersL12', 'masersL12', 'Normalized $L_{12}$',
     'MasersAndNonMasersGaussianHistoL12Test.pdf', 'Test Set L12 Distribution')

# Plot a Univariate Gaussian curve on top of a histogram plot for LX
# comparing masers and nonmasers for the test dataset
plot(nonMasersLXTest, masersLXTest, 'nonMasersLX', 'masersLX', 'Normalized $L_{X}$',
     'MasersAndNonMasersGaussianHistoLXTest.pdf', 'Test Set LX Distribution')
