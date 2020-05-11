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

# Load test dataset from dataset1 (80 Acc)
testX80 = np.load('./DataSet80+Acc/DataTestXKNNUnweighted.npy')
testY80 = np.load('./DataSet80+Acc/DataTestYKNNUnweighted.npy')

# Load test dataset from dataset2 (50-60 Acc)
testX50 = np.load('./DataSet50-60Acc/DataTestXKNNUnweighted.npy')
testY50 = np.load('./DataSet50-60Acc/DataTestYKNNUnweighted.npy')

# Load test dataset from dataset3 (70 Acc)
testX70 = np.load('./DataSet70Acc/DataTestXKNNUnweighted.npy')
testY70 = np.load('./DataSet70Acc/DataTestYKNNUnweighted.npy')

# Test the models from dataset1 on the test data from datasets 2 and 3
model80Acc3Layer = load_model('./DataSet80+Acc/kerasModel3Layer_12_8_1.h5')
model80Acc4Layer = load_model('./DataSet80+Acc/kerasModel4Layer_8_12_4_1.h5')

threeLayerDataset1TestYPred = model80Acc3Layer.predict_classes(testX80)
fourLayerDataset1TestYPred = model80Acc4Layer.predict_classes(testX80)

threeLayerDataset2TestYPred = model80Acc3Layer.predict_classes(testX50)
fourLayerDataset2TestYPred = model80Acc4Layer.predict_classes(testX50)

threeLayerDataset3TestYPred = model80Acc3Layer.predict_classes(testX70)
fourLayerDataset3TestYPred = model80Acc4Layer.predict_classes(testX70)

print('##########################################################################')
print("Actual Y")
print(testY80)
print('##########################################################################')
print("Predicted Y")
print(threeLayerDataset1TestYPred)

# Calculate accuracy and F1 Score of dataset1 for the 3Layer model
dataset1Test3LayerF1 = metrics.f1_score(testY80, threeLayerDataset1TestYPred)
print('3 Layer Dataset 1 Test F1 Score: %.3f' % dataset1Test3LayerF1)
_, dataset1Test3LayerAccuracy = model80Acc3Layer.evaluate(testX80, testY80)
print('3 Layer Dataset 1 Test Accuracy: %.2f' % (dataset1Test3LayerAccuracy * 100))

# Calculate accuracy and F1 Score of dataset2 for the 3Layer model
dataset2Test3LayerF1 = metrics.f1_score(testY50, threeLayerDataset2TestYPred)
print('3 Layer Dataset 2 Test F1 Score: %.3f' % dataset2Test3LayerF1)
_, dataset2Test3LayerAccuracy = model80Acc3Layer.evaluate(testX50, testY50)
print('3 Layer Dataset 2 Test Accuracy: %.2f' % (dataset2Test3LayerAccuracy * 100))

# Calculate accuracy and F1 Score of dataset3 for the 3Layer model
dataset3Test3LayerF1 = metrics.f1_score(testY70, threeLayerDataset3TestYPred)
print('3 Layer Dataset 3 Test F1 Score: %.3f' % dataset3Test3LayerF1)
_, dataset3Test3LayerAccuracy = model80Acc3Layer.evaluate(testX70, testY70)
print('3 Layer Dataset 3 Test Accuracy: %.2f' % (dataset3Test3LayerAccuracy * 100))

####

# Calculate accuracy and F1 Score of dataset1 for the 4Layer Model
dataset1Test4LayerF1 = metrics.f1_score(testY80, fourLayerDataset1TestYPred)
print('4Layer Dataset 1 Test F1 Score: %.3f' % dataset1Test4LayerF1)
_, dataset1Test4LayerAccuracy = model80Acc4Layer.evaluate(testX80, testY80)
print('4 Layer Dataset 1 Test Accuracy: %.2f' % (dataset1Test4LayerAccuracy * 100))

# Calculate accuracy and F1 Score of dataset2 for the 4Layer Model
dataset2Test4LayerF1 = metrics.f1_score(testY50, fourLayerDataset2TestYPred)
print('4 Layer Dataset 2 Test F1 Score: %.3f' % dataset2Test4LayerF1)
_, dataset2Test4LayerAccuracy = model80Acc4Layer.evaluate(testX50, testY50)
print('4 Layer Dataset 2 Test Accuracy: %.2f' % (dataset2Test4LayerAccuracy * 100))

# Calculate accuracy and F1 Score of dataset3 for the 4Layer Model
dataset3Test4LayerF1 = metrics.f1_score(testY70, fourLayerDataset3TestYPred)
print('4 Layer Dataset 3 Test F1 Score: %.3f' % dataset3Test4LayerF1)
_, dataset3Test4LayerAccuracy = model80Acc4Layer.evaluate(testX70, testY70)
print('4 Layer Dataset 3 Test Accuracy: %.2f' % (dataset3Test4LayerAccuracy * 100))