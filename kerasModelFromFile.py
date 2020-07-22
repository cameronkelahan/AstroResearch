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
import matplotlib.pyplot as plt
import time

# Removes truncation of arrays
np.set_printoptions(threshold=np.inf)


################ LOAD TEST DATASETS FROM RESPECTIVE UNW AND NN SUBSAMPLED DATASETS

# # Load test dataset from unw dataset1 (80 Acc)
# testUnwX80 = np.load('./DataSetUnwKNN80+Acc/DataTestXKNNUnweighted.npy')
# testUnwY80 = np.load('./DataSetUnwKNN80+Acc/DataTestYKNNUnweighted.npy')
#
# # Load test dataset from unw dataset2 (50-60 Acc)
# testUnwX50 = np.load('./DataSetUnwKNN50-60Acc/DataTestXKNNUnweighted.npy')
# testUnwY50 = np.load('./DataSetUnwKNN50-60Acc/DataTestYKNNUnweighted.npy')
#
# # Load test dataset from unw dataset3 (70 Acc)
# testUnwX70 = np.load('./DataSetUnwKNN70Acc/DataTestXKNNUnweighted.npy')
# testUnwY70 = np.load('./DataSetUnwKNN70Acc/DataTestYKNNUnweighted.npy')

#####

# Load test dataset from NN dataset1 (84 F1)
testNNX84 = np.load('./DataSetNN84F1/DataTestXNN.npy')
testNNY84 = np.load('./DataSetNN84F1/DataTestYNN.npy')

# Load test dataset and total X from NN dataset2 (57F1)
testNNX57 = np.load('./DataSetNN57F1/DataTestXNN.npy')
testNNY57 = np.load('./DataSetNN57F1/DataTestYNN.npy')
# nnTotalX50 = np.load
# nnTotalY50 = np.load

# Load test dataset and total X from NN dataset3 (79 F1)
testNNX79 = np.load('./DataSetNN79F1/DataTestXNN.npy')
testNNY79 = np.load('./DataSetNN79F1/DataTestYNN.npy')
# nnTotalX70 = np.load
# nnTotalY70 = np.load


###################### Load the models which were trained using the data from Unw KNN Dataset 1
# model80Acc3Layer = load_model('./DataSetUnwKNN80+Acc/NeuralNetworkInfo/kerasModel3Layer_12_8_1.h5')
# model80Acc4Layer = load_model('./DataSetUnwKNN80+Acc/NeuralNetworkInfo/kerasModel4Layer_8_12_4_1.h5')
# model80Acc6Layer = load_model('./DataSetUnwKNN80+Acc/NeuralNetworkInfo/kerasModel6LayerEpoch550_10_10_20_50_17_1.h5')
# model80Acc10Layer = load_model('./DataSetUnwKNN80+Acc/NeuralNetworkInfo/kerasModel10LayerEpoch550_10_10_20_25_30_50_17_12_8_1.h5')
######################

###################### Train the Neural Network models based on NN Dataset 1 (84 F1)
# 4Layer Model
model84F1_4Layer = Sequential()
model84F1_4Layer.add(Dense(8, input_dim=2, activation='relu'))
model84F1_4Layer.add(Dense(12, activation='relu'))
model84F1_4Layer.add(Dense(4, activation='relu'))
model84F1_4Layer.add(Dense(1, activation='sigmoid'))
model84F1_4Layer.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model84F1_4Layer.save('./DataSetNN84F1/kerasModel4Layer_8_12_4_1.h5')

# 6Layer Model; load from file; created while searching for the datasets
model84F1_6Layer = load_model('./DataSetNN84F1/best6LayerNNModel.h5')
model84F1_6Layer.save('./DataSetNN84F1/kerasModel6Layer_10_10_20_50_17_1.h5')

# 10Layer Model
model84F1_10Layer = Sequential()
model84F1_10Layer.add(Dense(10, input_dim=2, activation='relu'))
model84F1_10Layer.add(Dense(10, activation='relu'))
model84F1_10Layer.add(Dense(20, activation='relu'))
model84F1_10Layer.add(Dense(25, activation='relu'))
model84F1_10Layer.add(Dense(30, activation='relu'))
model84F1_10Layer.add(Dense(50, activation='relu'))
model84F1_10Layer.add(Dense(17, activation='relu'))
model84F1_10Layer.add(Dense(12, activation='relu'))
model84F1_10Layer.add(Dense(8, activation='relu'))
model84F1_10Layer.add(Dense(1, activation='sigmoid'))
model84F1_10Layer.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model84F1_10Layer.save('./DataSetNN84F1/kerasModel10Layer_10_10_20_25_30_50_17_12_8_1.h5')
#######################

####################### Calculate the predicted Class values for each unweighted test dataset from data set 1, 2, and 3
####################### for each Neural Network model
# threeLayerDataset1TestYPred = model80Acc3Layer.predict_classes(testUnwX80)
# fourLayerDataset1TestYPred = model80Acc4Layer.predict_classes(testUnwX80)
# sixLayerDataset1TestYPred = model80Acc6Layer.predict_classes(testUnwX80)
# tenLayerDataset1TestYPred = model80Acc10Layer.predict_classes(testUnwX80)
#
# threeLayerDataset2TestYPred = model80Acc3Layer.predict_classes(testUnwX50)
# fourLayerDataset2TestYPred = model80Acc4Layer.predict_classes(testUnwX50)
# sixLayerDataset2TestYPred = model80Acc6Layer.predict_classes(testUnwX50)
# tenLayerDataset2TestYPred = model80Acc10Layer.predict_classes(testUnwX50)
#
# threeLayerDataset3TestYPred = model80Acc3Layer.predict_classes(testUnwX70)
# fourLayerDataset3TestYPred = model80Acc4Layer.predict_classes(testUnwX70)
# sixLayerDataset3TestYPred = model80Acc6Layer.predict_classes(testUnwX70)
# tenLayerDataset3TestYPred = model80Acc10Layer.predict_classes(testUnwX70)
#######################

####################### Calculate the predicted Class from each neural network model based on:
####################### the test dataset from Dataset1 (because the model was trained on this dataset),
####################### and then all of (unsplit X) datasets 2 and 3
nn4LayerDataset1TestYPred = model84F1_4Layer.predict_classes(testNNX84)
nn6LayerDataset1TestYPred = model84F1_6Layer.predict_classes(testNNX84)
nn10LayerDataset1TestYPred = model84F1_10Layer.predict_classes(testNNX84)

# Use the testX set for now, replace later with entire X
nn4LayerDataset2TestYPred = model84F1_4Layer.predict_classes(testNNX57)
nn6LayerDataset2TestYPred = model84F1_6Layer.predict_classes(testNNX57)
nn10LayerDataset2TestYPred = model84F1_10Layer.predict_classes(testNNX57)

# Use the testX set for now, replace later with entire X
nn4LayerDataset3TestYPred = model84F1_4Layer.predict_classes(testNNX79)
nn6LayerDataset3TestYPred = model84F1_6Layer.predict_classes(testNNX79)
nn10LayerDataset3TestYPred = model84F1_10Layer.predict_classes(testNNX79)
#######################

# print('##########################################################################')
# print("Actual Y")
# print(testY80)
# print('##########################################################################')
# print("Predicted Y")
# print(threeLayerDataset1TestYPred)

####################### Calculate F1 and Accuracies of each model's prediction for each Unweighted dataset
# # Calculate accuracy and F1 Score of dataset1 for the 3Layer model
# dataset1Test3LayerF1 = metrics.f1_score(testUnwY80, threeLayerDataset1TestYPred)
# print('3 Layer Dataset 1 Test F1 Score: %.3f' % dataset1Test3LayerF1)
# _, dataset1Test3LayerAccuracy = model80Acc3Layer.evaluate(testUnwX80, testUnwY80)
# print('3 Layer Dataset 1 Test Accuracy: %.2f' % (dataset1Test3LayerAccuracy * 100))
#
# # Calculate accuracy and F1 Score of dataset2 for the 3Layer model
# dataset2Test3LayerF1 = metrics.f1_score(testUnwY50, threeLayerDataset2TestYPred)
# print('3 Layer Dataset 2 Test F1 Score: %.3f' % dataset2Test3LayerF1)
# _, dataset2Test3LayerAccuracy = model80Acc3Layer.evaluate(testUnwX50, testUnwY50)
# print('3 Layer Dataset 2 Test Accuracy: %.2f' % (dataset2Test3LayerAccuracy * 100))
#
# # Calculate accuracy and F1 Score of dataset3 for the 3Layer model
# dataset3Test3LayerF1 = metrics.f1_score(testUnwY70, threeLayerDataset3TestYPred)
# print('3 Layer Dataset 3 Test F1 Score: %.3f' % dataset3Test3LayerF1)
# _, dataset3Test3LayerAccuracy = model80Acc3Layer.evaluate(testUnwX70, testUnwY70)
# print('3 Layer Dataset 3 Test Accuracy: %.2f' % (dataset3Test3LayerAccuracy * 100))
#
# ####
#
# # Calculate accuracy and F1 Score of dataset1 for the 4Layer Model
# dataset1Test4LayerF1 = metrics.f1_score(testUnwY80, fourLayerDataset1TestYPred)
# print('4Layer Dataset 1 Test F1 Score: %.3f' % dataset1Test4LayerF1)
# _, dataset1Test4LayerAccuracy = model80Acc4Layer.evaluate(testUnwX80, testUnwY80)
# print('4 Layer Dataset 1 Test Accuracy: %.2f' % (dataset1Test4LayerAccuracy * 100))
#
# # Calculate accuracy and F1 Score of dataset2 for the 4Layer Model
# dataset2Test4LayerF1 = metrics.f1_score(testUnwY50, fourLayerDataset2TestYPred)
# print('4 Layer Dataset 2 Test F1 Score: %.3f' % dataset2Test4LayerF1)
# _, dataset2Test4LayerAccuracy = model80Acc4Layer.evaluate(testUnwX50, testUnwY50)
# print('4 Layer Dataset 2 Test Accuracy: %.2f' % (dataset2Test4LayerAccuracy * 100))
#
# # Calculate accuracy and F1 Score of dataset3 for the 4Layer Model
# dataset3Test4LayerF1 = metrics.f1_score(testUnwY70, fourLayerDataset3TestYPred)
# print('4 Layer Dataset 3 Test F1 Score: %.3f' % dataset3Test4LayerF1)
# _, dataset3Test4LayerAccuracy = model80Acc4Layer.evaluate(testUnwX70, testUnwY70)
# print('4 Layer Dataset 3 Test Accuracy: %.2f' % (dataset3Test4LayerAccuracy * 100))
#
# ####
#
# # Calculate accuracy and F1 Score of dataset1 for the 6Layer Model
# dataset1Test6LayerF1 = metrics.f1_score(testUnwY80, sixLayerDataset1TestYPred)
# print('6Layer Dataset 1 Test F1 Score: %.3f' % dataset1Test6LayerF1)
# _, dataset1Test6LayerAccuracy = model80Acc6Layer.evaluate(testUnwX80, testUnwY80)
# print('6 Layer Dataset 1 Test Accuracy: %.2f' % (dataset1Test6LayerAccuracy * 100))
#
# # Calculate accuracy and F1 Score of dataset2 for the 6Layer Model
# dataset2Test6LayerF1 = metrics.f1_score(testUnwY50, sixLayerDataset2TestYPred)
# print('6 Layer Dataset 2 Test F1 Score: %.3f' % dataset2Test6LayerF1)
# _, dataset2Test6LayerAccuracy = model80Acc6Layer.evaluate(testUnwX50, testUnwY50)
# print('6 Layer Dataset 2 Test Accuracy: %.2f' % (dataset2Test6LayerAccuracy * 100))
#
# # Calculate accuracy and F1 Score of dataset3 for the 6Layer Model
# dataset3Test6LayerF1 = metrics.f1_score(testUnwY70, sixLayerDataset3TestYPred)
# print('6 Layer Dataset 3 Test F1 Score: %.3f' % dataset3Test6LayerF1)
# _, dataset3Test6LayerAccuracy = model80Acc6Layer.evaluate(testUnwX70, testUnwY70)
# print('6 Layer Dataset 3 Test Accuracy: %.2f' % (dataset3Test6LayerAccuracy * 100))
#
# ####
#
# # Calculate accuracy and F1 Score of dataset1 for the 10Layer Model
# dataset1Test10LayerF1 = metrics.f1_score(testUnwY80, tenLayerDataset1TestYPred)
# print('10Layer Dataset 1 Test F1 Score: %.3f' % dataset1Test10LayerF1)
# _, dataset1Test10LayerAccuracy = model80Acc10Layer.evaluate(testUnwX80, testUnwY80)
# print('10 Layer Dataset 1 Test Accuracy: %.2f' % (dataset1Test10LayerAccuracy * 100))
#
# # Calculate accuracy and F1 Score of dataset2 for the 10Layer Model
# dataset2Test10LayerF1 = metrics.f1_score(testUnwY50, tenLayerDataset2TestYPred)
# print('10 Layer Dataset 2 Test F1 Score: %.3f' % dataset2Test10LayerF1)
# _, dataset2Test10LayerAccuracy = model80Acc10Layer.evaluate(testUnwX50, testUnwY50)
# print('10 Layer Dataset 2 Test Accuracy: %.2f' % (dataset2Test10LayerAccuracy * 100))
#
# # Calculate accuracy and F1 Score of dataset3 for the 10Layer Model
# dataset3Test10LayerF1 = metrics.f1_score(testUnwY70, tenLayerDataset3TestYPred)
# print('10 Layer Dataset 3 Test F1 Score: %.3f' % dataset3Test10LayerF1)
# _, dataset3Test10LayerAccuracy = model80Acc10Layer.evaluate(testUnwX70, testUnwY70)
# print('10 Layer Dataset 3 Test Accuracy: %.2f' % (dataset3Test10LayerAccuracy * 100))
#######################

####################### Calculate F1 and Accuracies of each model's prediction for each NN dataset

# Calculate accuracy and F1 Score of dataset1 for the 4Layer Model
nnDataset1Test4LayerF1 = metrics.f1_score(testNNY84, nn4LayerDataset1TestYPred)
print('4Layer Dataset 1 Test F1 Score: %.3f' % nnDataset1Test4LayerF1)
_, nnDataset1Test4LayerAccuracy = model84F1_4Layer.evaluate(testNNX84, testNNY84)
print('4 Layer Dataset 1 Test Accuracy: %.2f' % (nnDataset1Test4LayerAccuracy * 100))

# Calculate accuracy and F1 Score of dataset2 for the 4Layer Model
nnDataset2Test4LayerF1 = metrics.f1_score(testNNY57, nn4LayerDataset2TestYPred)
print('4 Layer Dataset 2 Test F1 Score: %.3f' % nnDataset2Test4LayerF1)
_, nnDataset2Test4LayerAccuracy = model84F1_4Layer.evaluate(testNNX57, testNNY57)
print('4 Layer Dataset 2 Test Accuracy: %.2f' % (nnDataset2Test4LayerAccuracy * 100))

# Calculate accuracy and F1 Score of dataset3 for the 4Layer Model
nnDataset3Test4LayerF1 = metrics.f1_score(testNNY79, nn4LayerDataset3TestYPred)
print('4 Layer Dataset 3 Test F1 Score: %.3f' % nnDataset3Test4LayerF1)
_, nnDataset3Test4LayerAccuracy = model84F1_4Layer.evaluate(testNNX79, testNNY79)
print('4 Layer Dataset 3 Test Accuracy: %.2f' % (nnDataset3Test4LayerAccuracy * 100))

####

# Calculate accuracy and F1 Score of dataset1 for the 6Layer Model
nnDataset1Test6LayerF1 = metrics.f1_score(testNNY84, nn6LayerDataset1TestYPred)
print('6Layer Dataset 1 Test F1 Score: %.3f' % nnDataset1Test6LayerF1)
_, nnDataset1Test6LayerAccuracy = model84F1_6Layer.evaluate(testNNX84, testNNY84)
print('6 Layer Dataset 1 Test Accuracy: %.2f' % (nnDataset1Test6LayerAccuracy * 100))

# Calculate accuracy and F1 Score of dataset2 for the 6Layer Model
nnDataset2Test6LayerF1 = metrics.f1_score(testNNY57, nn6LayerDataset2TestYPred)
print('6 Layer Dataset 2 Test F1 Score: %.3f' % nnDataset2Test6LayerF1)
_, nnDataset2Test6LayerAccuracy = model84F1_6Layer.evaluate(testNNX57, testNNY57)
print('6 Layer Dataset 2 Test Accuracy: %.2f' % (nnDataset2Test6LayerAccuracy * 100))

# Calculate accuracy and F1 Score of dataset3 for the 6Layer Model
nnDataset3Test6LayerF1 = metrics.f1_score(testNNY79, nn6LayerDataset3TestYPred)
print('6 Layer Dataset 3 Test F1 Score: %.3f' % nnDataset3Test6LayerF1)
_, nnDataset3Test6LayerAccuracy = model84F1_6Layer.evaluate(testNNX79, testNNY79)
print('6 Layer Dataset 3 Test Accuracy: %.2f' % (nnDataset3Test6LayerAccuracy * 100))

####

# Calculate accuracy and F1 Score of dataset1 for the 10Layer Model
nnDataset1Test10LayerF1 = metrics.f1_score(testNNY84, nn10LayerDataset1TestYPred)
print('10Layer Dataset 1 Test F1 Score: %.3f' % nnDataset1Test10LayerF1)
_, nnDataset1Test10LayerAccuracy = model84F1_10Layer.evaluate(testNNX84, testNNY84)
print('10 Layer Dataset 1 Test Accuracy: %.2f' % (nnDataset1Test10LayerAccuracy * 100))

# Calculate accuracy and F1 Score of dataset2 for the 10Layer Model
nnDataset2Test10LayerF1 = metrics.f1_score(testNNY57, nn10LayerDataset2TestYPred)
print('10 Layer Dataset 2 Test F1 Score: %.3f' % nnDataset2Test10LayerF1)
_, nnDataset2Test10LayerAccuracy = model84F1_10Layer.evaluate(testNNX57, testNNY57)
print('10 Layer Dataset 2 Test Accuracy: %.2f' % (nnDataset2Test10LayerAccuracy * 100))

# Calculate accuracy and F1 Score of dataset3 for the 10Layer Model
nnDataset3Test10LayerF1 = metrics.f1_score(testNNY79, nn10LayerDataset3TestYPred)
print('10 Layer Dataset 3 Test F1 Score: %.3f' % nnDataset3Test10LayerF1)
_, nnDataset3Test10LayerAccuracy = model84F1_10Layer.evaluate(testNNX79, testNNY79)
print('10 Layer Dataset 3 Test Accuracy: %.2f' % (nnDataset3Test10LayerAccuracy * 100))
#######################
