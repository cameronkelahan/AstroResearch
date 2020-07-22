import numpy as np
import random
from numpy import genfromtxt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


############################################ Read In Data From File ####################################################
# Read in column 0 from Table1 for the name of the galaxy
galaxyName = genfromtxt('Paper2Table1.csv', delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=0)

# Read in Column 6 from Table1 (Maser Classification)
maserType = genfromtxt('Paper2Table1.csv', delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=6)

# Read in L12 from Table1
L12 = genfromtxt('Paper2Table1.csv', delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=7)

# Read in Lobs from Table2
LX = genfromtxt('Paper2Table2.csv', delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=4)

# Read in the NH Values
NH = genfromtxt('Paper2Table2.csv', delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=6)

########################################## Reshape the Data Matrix #####################################################
# Currently, the shape of the data matrix is flipped
# Reshape the data matrix to have 3 columns, one for each attribute, and as many rows as there are examples (galaxies)
data = []
count = 0
for value in L12:
    data.append([L12[count], LX[count], NH[count]])
    count += 1

###################################### Sort the galaxies by NH values ##################################################
gt22NHL12nonMaser = []
gt22NHLXnonMaser = []
gt22NHL12maser = []
gt22NHLXmaser = []

gt23NHL12 = []
gt23NHLX = []
gt23Class = []
gt23NHL12nonMaser = []
gt23NHLXnonMaser = []
gt23NHL12maser = []
gt23NHLXmaser = []

gt24NHL12 = []
gt24NHLX = []
gt24Class = []
gt24NHL12nonMaser = []
gt24NHLXnonMaser = []
gt24NHL12maser = []
gt24NHLXmaser = []

gt243NHL12 = []
gt243NHLX = []
gt243Class = []
gt243NHL12nonMaser = []
gt243NHLXnonMaser = []
gt243NHL12maser = []
gt243NHLXmaser = []

lt243NHL12nonMaser = []
lt243NHLXnonMaser = []
lt243NHL12maser = []
lt243NHLXmaser = []

# Need to compare the x,y values of the NH lines with the x,y values of each galaxy

def NHLines(X_val, change):
    NH_yVal = (1.60567 - change) + (0.956333 * X_val)
    # print(NH_yVal)
    return NH_yVal

# Define the change-in-y-intercept values for the different NH value ranges
change0 = 0
change22 = 0.0414
change23 = .2810
change24 = 1.3556
change243 = 2.29
# change25 = 3.47857

# value[0] is L12 value, value[1] is LX value
count = 0
for value in data:
    # If y value of galaxy is greater than the y value of the line at the given X value of the galaxy
    # append it to the list of galaxies above that line
    if value[1] > NHLines(value[0], change22):
        if maserType[count] == 0:
            gt22NHL12nonMaser.append(data[count][0])
            gt22NHLXnonMaser.append(data[count][1])
        else:
            gt22NHL12maser.append(data[count][0])
            gt22NHLXmaser.append(data[count][1])
    # If y value of galaxy is less than the y value of the previous line and greater then the y value of the next line
    # at the given X value of the galaxy, append it to the list of galaxies above that line
    elif value[1] <= NHLines(value[0], change22) and value[1] > NHLines(value[0], change23):
        gt23NHL12.append(data[count][0])
        gt23NHLX.append(data[count][1])
        if maserType[count] == 0:
            gt23Class.append(0)
            gt23NHL12nonMaser.append(data[count][0])
            gt23NHLXnonMaser.append(data[count][1])
        else:
            gt23Class.append(1)
            gt23NHL12maser.append(data[count][0])
            gt23NHLXmaser.append(data[count][1])
    # If y value of galaxy is less than the y value of the previous line and greater then the y value of the next line
    # at the given X value of the galaxy, append it to the list of galaxies above that line
    elif value[1] <= NHLines(value[0], change23) and value[1] > NHLines(value[0], change24):
        gt24NHL12.append(data[count][0])
        gt24NHLX.append(data[count][1])
        if maserType[count] == 0:
            gt24Class.append(0)
            gt24NHL12nonMaser.append(data[count][0])
            gt24NHLXnonMaser.append(data[count][1])
        else:
            gt24Class.append(1)
            gt24NHL12maser.append(data[count][0])
            gt24NHLXmaser.append(data[count][1])
    # If y value of galaxy is less than the y value of the previous line and greater then the y value of the next line
    # at the given X value of the galaxy, append it to the list of galaxies above that line
    elif value[1] <= NHLines(value[0], change24) and value[1] > NHLines(value[0], change243):
        gt243NHL12.append(data[count][0])
        gt243NHLX.append(data[count][1])
        if maserType[count] == 0:
            gt243Class.append(0)
            gt243NHL12nonMaser.append(data[count][0])
            gt243NHLXnonMaser.append(data[count][1])
        else:
            gt243Class.append(1)
            gt243NHL12maser.append(data[count][0])
            gt243NHLXmaser.append(data[count][1])
    # At this point the y value of the galaxy must be lower than all lines
    else:
        if maserType[count] == 0:
            lt243NHL12nonMaser.append(data[count][0])
            lt243NHLXnonMaser.append(data[count][1])
        else:
            lt243NHL12maser.append(data[count][0])
            lt243NHLXmaser.append(data[count][1])
    count += 1

######################################
# REFERENCE
# # Normalize L12
# maxValL12 = np.amax(L12)
# minValL12 = np.amin(L12)
# countL12 = 0
# for value in L12:
#     L12[countL12] = (value - minValL12) / (maxValL12 - minValL12)
#     countL12 += 1
#
# # Normalize Lobs
# maxValLobs = np.amax(Lobs)
# minValLobs = np.amin(Lobs)
# countLobs = 0
# for value in Lobs:
#     Lobs[countLobs] = (value - minValLobs) / (maxValLobs - minValLobs)
#     countLobs += 1

###################################### Normalize based on entire dataset ###############################################
# Used for entire set normalization
minLX = int(min(LX))
maxLX = int(max(LX))

# Used for entire set normalization
minL12 = int(min(L12))
maxL12 = int(max(L12))

# NH values gt 22 separated by maser/nonmaser
count = 0
for value in  gt22NHL12maser:
    gt22NHL12maser[count] = (value - minL12) / (maxL12 - minL12)
    count += 1

count = 0
for value in gt22NHL12nonMaser:
    gt22NHL12nonMaser[count] = (value - minL12) / (maxL12 - minL12)
    count += 1

count = 0
for value in gt22NHLXmaser:
    gt22NHLXmaser[count] = (value - minLX) / (maxLX - minLX)
    count += 1

count = 0
for value in gt22NHLXnonMaser:
    gt22NHLXnonMaser[count] = (value - minLX) / (maxLX - minLX)
    count += 1

# NH value gt 23 separated by maser/nonmaser
count = 0
for value in gt23NHL12maser:
    gt23NHL12maser[count] = (value - minL12) / (maxL12 - minL12)
    count += 1

count = 0
for value in gt23NHL12nonMaser:
    gt23NHL12nonMaser[count] = (value - minL12) / (maxL12 - minL12)
    count += 1

count = 0
for value in gt23NHLXmaser:
    gt23NHLXmaser[count] = (value - minLX) / (maxLX - minLX)
    count += 1

count = 0
for value in gt23NHLXnonMaser:
    gt23NHLXnonMaser[count] = (value - minLX) / (maxLX - minLX)
    count += 1

# NH Values gt 24 separated by maser/nonmaser
count = 0
for value in gt24NHL12maser:
    gt24NHL12maser[count] = (value - minL12) / (maxL12 - minL12)
    count += 1

count = 0
for value in gt24NHL12nonMaser:
    gt24NHL12nonMaser[count] = (value - minL12) / (maxL12 - minL12)
    count += 1

count = 0
for value in gt24NHLXmaser:
    gt24NHLXmaser[count] = (value - minLX) / (maxLX - minLX)
    count += 1

count = 0
for value in gt24NHLXnonMaser:
    gt24NHLXnonMaser[count] = (value - minLX) / (maxLX - minLX)
    count += 1

# NH Values gt 24.3 separated by maser/nonmaser
count = 0
for value in gt243NHL12maser:
    gt24NHL12maser[count] = (value - minL12) / (maxL12 - minL12)
    count += 1

count = 0
for value in gt243NHL12nonMaser:
    gt24NHL12nonMaser[count] = (value - minL12) / (maxL12 - minL12)
    count += 1

count = 0
for value in gt243NHLXmaser:
    gt243NHLXmaser[count] = (value - minLX) / (maxLX - minLX)
    count += 1

count = 0
for value in gt243NHLXnonMaser:
    gt243NHLXnonMaser[count] = (value - minLX) / (maxLX - minLX)

# All galaxies with NH Values gt 23
count = 0
for value in gt23NHL12:
    gt23NHL12[count] = (value - minL12) / (maxL12 - minL12)
    count += 1

count = 0
for value in gt23NHLX:
    gt23NHLX[count] = (value - minLX) / (maxLX - minLX)
    count += 1

# All galaxies with NH Values gt 24
count = 0
for value in gt24NHL12:
    gt24NHL12[count] = (value - minL12) / (maxL12 - minL12)
    count += 1

count = 0
for value in gt24NHLX:
    gt24NHLX[count] = (value - minLX) / (maxLX - minLX)
    count += 1

# All galaxies with NH Values gt 24.3
count = 0
for value in gt243NHL12:
    gt243NHL12[count] = (value - minL12) / (maxL12 - minL12)
    count += 1

count = 0
for value in gt243NHLX:
    gt243NHLX[count] = (value - minLX) / (maxLX - minLX)
    count += 1

############################################# Heat Map Code Section ####################################################
# Create the x and y axis values (0 - 1 stepping by .01) as fabricated galaxies
xAxis = np.linspace(0, 1, num=101)
yAxis = np.linspace(0, 1, num=101)

# The X data set to populate the heat maps and predict probability
predX = []
for x in xAxis:
    for y in yAxis:
        predX.append([x, y])

## Lists used to build KNN Models
# Ordered list of L12 data ex: gt23NHL12[0]
# Ordered list of LX data ex: gt23NHLX[0]
# Ordered list of class data ex: gt23Class[0]

Xgt23 = []
count = 0
for value in gt23NHL12:
    Xgt23.append([gt23NHL12[count], gt23NHLX[count]])
    count += 1

Xgt24 = []
count = 0
for value in gt24NHL12:
    Xgt24.append([gt24NHL12[count], gt24NHLX[count]])
    count += 1

Xgt243 = []
count = 0;
for value in gt243NHL12:
    Xgt243.append([gt243NHL12[count], gt243NHLX[count]])
    count += 1

################################ Unw Modeling for NH values > 23
kRange = range(1, 16)

# Creates a random selection of Train and Test data
# Test data is 20%, Train data is 80%
randNum = random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(Xgt23, gt23Class, test_size=0.2, random_state=randNum)
# print("XTrain = ", X_train)
# print("length of Xtrain = ", len(X_train))
# print("length of Ytrain = ", len(y_train))
# print("Length of Xtest = ", len(X_test))
# print("Length of Ytest = ", len(y_test))

bestF1gt23 = 0
bestKValgt23 = 0
for value in kRange:
    modelUnwTestgt23 = KNeighborsClassifier(n_neighbors=value, metric='euclidean')
    y_predUnwTestgt23 = modelUnwTestgt23.fit(X_train, y_train).predict(X_test)

    # print("Y_Pred: ", y_predUnwTestgt23)
    # print("Y_Test: ", y_test)

    tempF1Score = metrics.f1_score(y_test, y_predUnwTestgt23, zero_division=True)
    # print("Temp F1 Score = ", tempF1Score)

    if tempF1Score >= bestF1gt23:
        bestF1gt23 = tempF1Score
        bestKValgt23 = value
    # print("Curr K Val = ", value)
    # print("Best K = ", bestKValgt23)
    # print("Best F1 = ", bestF1gt23)


# Create Unweighted Heat Map of Data based on the classification of fabricated galaxies
modelUnwgt23 = KNeighborsClassifier(n_neighbors=bestKValgt23, metric='euclidean')
modelUnwgt23.fit(X_train, y_train)
predProb = modelUnwgt23.predict_proba(predX)

predMasergt23 = predProb[:,1]
predMasergt23 = predMasergt23.reshape(101, 101)
predMaser = predMasergt23.transpose()
# print(predMasergt23)

plt.figure(figsize=(6.4, 4.8))
plt.imshow(predMaser, origin='lower', extent=[0, 1, 0, 1])
plt.xticks(np.arange(.2, 1.1, step=0.2))  # Set label locations
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Unweighted Predicted Prob of Maser', fontsize=16)
plt.clim(vmin=0, vmax=1)
cbar.set_clim(vmin=0, vmax=1)
plt.xlabel('$L_{12}$',fontsize=16)
plt.ylabel('$L_X$',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# save as a PDF
# plt.savefig('HeatMapUnwNormEntireSetgt23.pdf', dpi=400, bbox_inches='tight', pad_inches=0.05)
# plt.show()
plt.clf()
plt.close()

################################ Unw Modeling for NH values > 24
kRange = range(1, 16)

# Creates a random selection of Train and Test data
# Test data is 20%, Train data is 80%
randNum = random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(Xgt24, gt24Class, test_size=0.2, random_state=randNum)
# print("XTrain = ", X_train)
# print("length of Xtrain = ", len(X_train))
# print("length of Ytrain = ", len(y_train))
# print("Length of Xtest = ", len(X_test))
# print("Length of Ytest = ", len(y_test))

bestF1gt24 = 0
bestKValgt24 = 0
for value in kRange:
    modelUnwTestgt24 = KNeighborsClassifier(n_neighbors=value, metric='euclidean')
    y_predUnwTestgt24 = modelUnwTestgt24.fit(X_train, y_train).predict(X_test)

    # print("Y_Pred: ", y_predUnwTestgt24)
    # print("Y_Test: ", y_test)

    tempF1Score = metrics.f1_score(y_test, y_predUnwTestgt24, zero_division=True)
    # print("Temp F1 Score = ", tempF1Score)

    if tempF1Score >= bestF1gt24:
        bestF1gt24 = tempF1Score
        bestKValgt24 = value
    # print("Curr K Val = ", value)
    # print("Best K = ", bestKValgt24)
    # print("Best F1 = ", bestF1gt24)


# Create Unweighted Heat Map of Data based on the classification of fabricated galaxies
modelUnwgt24 = KNeighborsClassifier(n_neighbors=bestKValgt24, metric='euclidean')
modelUnwgt24.fit(X_train, y_train)
predProb = modelUnwgt24.predict_proba(predX)

predMasergt24 = predProb[:,1]
predMasergt24 = predMasergt24.reshape(101, 101)
predMaser = predMasergt24.transpose()
# print(predMasergt24)

plt.figure(figsize=(6.4, 4.8))
plt.imshow(predMaser, origin='lower', extent=[0, 1, 0, 1])
plt.xticks(np.arange(.2, 1.1, step=0.2))  # Set label locations
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Unweighted Predicted Prob of Maser', fontsize=16)
plt.clim(vmin=0, vmax=1)
cbar.set_clim(vmin=0, vmax=1)
plt.xlabel('$L_{12}$',fontsize=16)
plt.ylabel('$L_X$',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# save as a PDF
# plt.savefig('HeatMapUnwNormEntireSetgt24.pdf', dpi=400, bbox_inches='tight', pad_inches=0.05)
# plt.show()
plt.clf()
plt.close()

################################ Unw Modeling for NH values > 24.3
kRange = range(1, 16)

# Creates a random selection of Train and Test data
# Test data is 20%, Train data is 80%
randNum = random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(Xgt243, gt243Class, test_size=0.2, random_state=randNum)
# print("XTrain = ", X_train)
print("length of Xtrain = ", len(X_train))
print("length of Ytrain = ", len(y_train))
print("Length of Xtest = ", len(X_test))
print("Length of Ytest = ", len(y_test))

bestF1gt243 = 0
bestKValgt243 = 0
for value in kRange:
    modelUnwTestgt243 = KNeighborsClassifier(n_neighbors=value, metric='euclidean')
    y_predUnwTestgt243 = modelUnwTestgt243.fit(X_train, y_train).predict(X_test)

    print("Y_Pred: ", y_predUnwTestgt243)
    print("Y_Test: ", y_test)

    tempF1Score = metrics.f1_score(y_test, y_predUnwTestgt243, zero_division=True)
    print("Temp F1 Score = ", tempF1Score)

    if tempF1Score >= bestF1gt243:
        bestF1gt243 = tempF1Score
        bestKValgt243 = value
    print("Curr K Val = ", value)
    print("Best K = ", bestKValgt243)
    print("Best F1 = ", bestF1gt243)


# Create Unweighted Heat Map of Data based on the classification of fabricated galaxies
modelUnwgt243 = KNeighborsClassifier(n_neighbors=bestKValgt243, metric='euclidean')
modelUnwgt243.fit(X_train, y_train)
predProb = modelUnwgt243.predict_proba(predX)

predMasergt243 = predProb[:,1]
predMasergt243 = predMasergt243.reshape(101, 101)
predMaser = predMasergt243.transpose()
# print(predMasergt24)

plt.figure(figsize=(6.4, 4.8))
plt.imshow(predMaser, origin='lower', extent=[0, 1, 0, 1])
plt.xticks(np.arange(.2, 1.1, step=0.2))  # Set label locations
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Unweighted Predicted Prob of Maser', fontsize=16)
plt.clim(vmin=0, vmax=1)
cbar.set_clim(vmin=0, vmax=1)
plt.xlabel('$L_{12}$',fontsize=16)
plt.ylabel('$L_X$',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# save as a PDF
plt.savefig('HeatMapUnwNormEntireSetgt243.pdf', dpi=400, bbox_inches='tight', pad_inches=0.05)
plt.show()
plt.clf()
plt.close()