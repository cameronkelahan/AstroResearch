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

def heatMap(model, title, saveName):
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

    predProb = model.predict_proba(np.array(predX))

    # predMaser = predMaser.reshape(11, 11)
    predMaser = predProb.reshape(101, 101)
    predMaser = predMaser.transpose()

    plt.figure(figsize=(6.4, 4.8))
    plt.imshow(predMaser, origin='lower', extent=[0, 1, 0, 1])
    plt.xticks(np.arange(.2, 1.1, step=0.2))  # Set label locations
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Predicted Prob of Maser', fontsize=16)
    plt.clim(vmin=0, vmax=1)
    cbar.set_clim(0, 1)
    # plt.text(-10, 50, t, family='serif', ha='right', wrap=True)
    plt.title(title)
    plt.xlabel('L12', fontsize=16)
    plt.ylabel('Lx', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # Save as a PDF
    # plt.savefig(saveName, dpi=400, bbox_inches='tight', pad_inches=0.05)
    plt.show()
    plt.clf()
    plt.close()

def main():
    # Load test dataset from NN dataset1 (84 F1)
    X_train = np.load('./EntireDataSetTrainingTest/EntireDataSetXTrain.npy')
    y_train = np.load('./EntireDataSetTrainingTest/EntireDataSetYTrain.npy')

    # Load test dataset and total X from NN dataset2 (57F1)
    X_test = np.load('./EntireDataSetTrainingTest/EntireDataSetXTest.npy')
    y_test = np.load('./EntireDataSetTrainingTest/EntireDataSetYTest.npy')

    ################ Create the 3 and 4 layer NN models
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

    # # 8-Layer Model
    # model = Sequential()
    # model.add(Dense(10, input_dim=2, activation='relu'))
    # model.add(Dense(9, activation='relu'))
    # model.add(Dense(8, activation='relu'))
    # model.add(Dense(7, activation='relu'))
    # model.add(Dense(6, activation='relu'))
    # model.add(Dense(4, activation='relu'))
    # model.add(Dense(2, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 3Layer
    num_epochs = 400

    # # 4Layer
    # num_epochs = 500

    # # 8Layer
    # num_epochs = 500

    model.fit(X_train, y_train, epochs=num_epochs, batch_size=10, verbose=1)
    # validation_data=(np.array(X_val), y_val),
    testYPred = model.predict_classes(np.array(X_test))
    testAcc = metrics.accuracy_score(y_test, testYPred)
    testF1 = metrics.f1_score(y_test, testYPred)

    print("Test Acc = ", testAcc)
    print("Test F1 = ", testF1)

    heatMap(model, '3-Layer NN Maser Probability Heat Map', '3LayerHeatMapEntireDataSetTraining.pdf')
    # heatMap(model, '4-Layer NN Maser Probability Heat Map', '4LayerHeatMapEntireDataSetTraining.pdf')
    # heatMap(model, '8-Layer NN Maser Probability Heat Map', '8LayerHeatMapEntireDataSetTraining.pdf')

if __name__ == '__main__':
    main()

# The accuracy being produced by the NN models always = .8947
# 89.47% of the training data set is nonmasers (only getting these right)
# F1 score is 0
