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

kRange = range(2, 16)
bestKValue = 3

# Rebuild the models by loading in the data from file
bestTrainSetX = np.load('DataTrainingXKNNUnweighted.npy')
bestTrainSetY = np.load('DataTrainingYKNNUnweighted.npy')
bestTestSetX = np.load('DataTestXKNNUnweighted.npy')
bestTestSetY = np.load('DataTestYKNNUnweighted.npy')

# Unweighted KNN Model
modelUnw = KNeighborsClassifier(n_neighbors=bestKValue, metric='euclidean')
y_predUnw = modelUnw.fit(bestTrainSetX, bestTrainSetY).predict(bestTestSetX)
testF1Unw = metrics.f1_score(bestTestSetY, y_predUnw)

# Rebuild the model with the most accurate training set and k value found from above, using distance as a weight
modelWei = KNeighborsClassifier(n_neighbors=bestKValue, metric='euclidean', weights='distance')
y_predWei = modelWei.fit(bestTrainSetX, bestTrainSetY).predict(bestTestSetX)
testF1Wei = metrics.f1_score(bestTestSetY, y_predWei)

# Find the Unweighted Test F1 Scores for each K using the Unweighted KNN Model. Used for a graph later.
unwTestF1Scores = []
for kval in kRange:
    model = KNeighborsClassifier(n_neighbors=kval, metric='euclidean')
    forLoopUnwY_pred = model.fit(bestTrainSetX, bestTrainSetY).predict(bestTestSetX)
    forLoopUnwTestF1 = metrics.f1_score(bestTestSetY, forLoopUnwY_pred)
    unwTestF1Scores.append(forLoopUnwTestF1)

# Find the Weighted Test F1 Scores for each K using the Weighted KNN Model. Used for a graph later
weightedTestF1Scores = []
for kval in kRange:
    model = KNeighborsClassifier(n_neighbors=kval, metric='euclidean', weights='distance')
    forLoopWeiY_pred = model.fit(bestTrainSetX, bestTrainSetY).predict(bestTestSetX)
    forLoopWeiTestF1 = metrics.f1_score(bestTestSetY, forLoopWeiY_pred)
    weightedTestF1Scores.append(forLoopWeiTestF1)

# Plot the Unweighted and Weighted F1 Scores for each value of K to compare
plt.figure(figsize=(6.4, 4.8))
plt.plot(kRange, unwTestF1Scores, label = 'Unweighted F1 Test Scores Per K Value')
plt.plot(kRange, weightedTestF1Scores, label = 'Weighted F1 Test Scores Per K Value')
plt.xlabel("Value of K")
plt.ylabel("F1 Score of KNN Model")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title("Comparison Of Test F1 Scores: Weighted and Unweighted Models")
plt.legend()
plt.savefig("UnwAndWeiTestF1ScoreValuesEveryK.pdf", dpi=400, bbox_inches='tight', pad_inches=0.05)
plt.show()
plt.clf()
plt.close()

# Create the x and y axis values (0 - 1 stepping by .01) as fabricated galaxies
xAxis = np.linspace(0, 1, num=101)
yAxis = np.linspace(0, 1, num=101)

# The X data set to populate and predict probability
predX = []
for x in xAxis:
    for y in yAxis:
        predX.append([x, y])

# Create Unweighted Heat Map of Data based on the classification of fabricated galaxies
model = KNeighborsClassifier(n_neighbors=bestKValue, metric='euclidean')
model.fit(bestTrainSetX, bestTrainSetY)
predProb = model.predict_proba(predX)
predMaser = predProb[:,1]
predMaser = predMaser.reshape(101, 101)
predMaser = predMaser.transpose()

plt.figure(figsize=(6.4, 4.8))
plt.imshow(predMaser, origin='lower', extent=[0,1,0,1])
plt.xticks(np.arange(.2, 1.1, step=0.2))  # Set label locations
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Unweighted Predicted Prob of Maser', fontsize=16)
cbar.set_clim(0,1)
plt.xlabel('L12',fontsize=16)
plt.ylabel('Lx',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# save as a PDF
plt.savefig('1000IterUnweightedProbMap100x100SameDataset.pdf', dpi=400, bbox_inches='tight', pad_inches=0.05)
plt.show()
plt.clf()
plt.close()

# Create the Weighted Heat Map of Data based on the classification of fabricated galaxies
model = KNeighborsClassifier(n_neighbors=bestKValue, metric='euclidean', weights='distance')
model.fit(bestTrainSetX, bestTrainSetY)
predProb = model.predict_proba(predX)
predMaser = predProb[:,1]
predMaser = predMaser.reshape(101, 101)
predMaser = predMaser.transpose()

plt.figure(figsize=(6.4, 4.8))
plt.imshow(predMaser, origin='lower', extent=[0,1,0,1])
plt.xticks(np.arange(.2, 1.1, step=0.2))  # Set label locations
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Weighted Predicted Prob of Maser', fontsize=16)
cbar.set_clim(0,1)
plt.xlabel('L12',fontsize=16)
plt.ylabel('Lx',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# Save as a PDF
plt.savefig('1000IterWeightedProbMap100x100SameDataset.pdf', dpi=400, bbox_inches='tight', pad_inches=0.05)
plt.show()
plt.clf()
plt.close()