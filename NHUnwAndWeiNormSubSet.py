import numpy as np
from numpy import genfromtxt
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

gt23NHL12nonMaser = []
gt23NHLXnonMaser = []
gt23NHL12maser = []
gt23NHLXmaser = []

gt24NHL12nonMaser = []
gt24NHLXnonMaser = []
gt24NHL12maser = []
gt24NHLXmaser = []

gt243NHL12nonMaser = []
gt243NHLXnonMaser = []
gt243NHL12maser = []
gt243NHLXmaser = []

lt243NHL12nonMaser = []
lt243NHLXnonMaser = []
lt243NHL12maser = []
lt243NHLXmaser = []


# Need to compare the x,y values of the NH lines with the x,y values of each galaxy
def NHLines(x_val, change):
    NH_yVal = (1.60567 - change) + (0.956333 * x_val)
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
        if maserType[count] == 0:
            gt23NHL12nonMaser.append(data[count][0])
            gt23NHLXnonMaser.append(data[count][1])
        else:
            gt23NHL12maser.append(data[count][0])
            gt23NHLXmaser.append(data[count][1])
    # If y value of galaxy is less than the y value of the previous line and greater then the y value of the next line
    # at the given X value of the galaxy, append it to the list of galaxies above that line
    elif value[1] <= NHLines(value[0], change23) and value[1] > NHLines(value[0], change24):
        if maserType[count] == 0:
            gt24NHL12nonMaser.append(data[count][0])
            gt24NHLXnonMaser.append(data[count][1])
        else:
            gt24NHL12maser.append(data[count][0])
            gt24NHLXmaser.append(data[count][1])
    # If y value of galaxy is less than the y value of the previous line and greater then the y value of the next line
    # at the given X value of the galaxy, append it to the list of galaxies above that line
    elif value[1] <= NHLines(value[0], change24) and value[1] > NHLines(value[0], change243):
        if maserType[count] == 0:
            gt243NHL12nonMaser.append(data[count][0])
            gt243NHLXnonMaser.append(data[count][1])
        else:
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
# Used for entire gt22 normalization
gt22minLX = int(min(min(gt22NHLXmaser), min(gt22NHLXnonMaser)))
gt22maxLX = int(max(max(gt22NHLXmaser), max(gt22NHLXnonMaser)))
gt22minL12 = int(min(min(gt22NHL12maser), min(gt22NHL12nonMaser)))
gt22maxL12 = int(max(max(gt22NHL12maser), max(gt22NHL12nonMaser)))

# Used for entire gt23 normalization
gt23minLX = int(min(min(gt23NHLXmaser), min(gt23NHLXnonMaser)))
gt23maxLX = int(max(max(gt23NHLXmaser), max(gt23NHLXnonMaser)))
gt23minL12 = int(min(min(gt23NHL12maser), min(gt23NHL12nonMaser)))
gt23maxL12 = int(max(max(gt23NHL12maser), max(gt23NHL12nonMaser)))

# Used for entire gt24 normalization
gt24minLX = int(min(min(gt24NHLXmaser), min(gt24NHLXnonMaser)))
gt24maxLX = int(max(max(gt24NHLXmaser), max(gt24NHLXnonMaser)))
gt24minL12 = int(min(min(gt24NHL12maser), min(gt24NHL12nonMaser)))
gt24maxL12 = int(max(max(gt24NHL12maser), max(gt24NHL12nonMaser)))

# Used for entire gt24.3 normalization
gt243minLX = int(min(min(gt243NHLXmaser), min(gt243NHLXnonMaser)))
gt243maxLX = int(max(max(gt243NHLXmaser), max(gt243NHLXnonMaser)))
gt243minL12 = int(min(min(gt243NHL12maser), min(gt243NHL12nonMaser)))
gt243maxL12 = int(max(max(gt243NHL12maser), max(gt243NHL12nonMaser)))

# Used for entire lt24.3 normalization
lt243minLX = int(min(min(lt243NHLXmaser), min(lt243NHLXnonMaser)))
lt243maxLX = int(max(max(lt243NHLXmaser), max(lt243NHLXnonMaser)))
lt243minL12 = int(min(min(lt243NHL12maser), min(lt243NHL12nonMaser)))
lt243maxL12 = int(max(max(lt243NHL12maser), max(lt243NHL12nonMaser)))

# # NH values gt 22
# count = 0
# for value in  gt22NHL12maser:
#     gt22NHL12maser[count] = (value - minL12) / (maxL12 - minL12)
#     count += 1
#
# count = 0
# for value in gt22NHL12nonMaser:
#     gt22NHL12nonMaser[count] = (value - minL12) / (maxL12 - minL12)
#     count += 1
#
# count = 0
# for value in gt22NHLXmaser:
#     gt22NHLXmaser[count] = (value - minLX) / (maxLX - minLX)
#     count += 1
#
# count = 0
# for value in gt22NHLXnonMaser:
#     gt22NHLXnonMaser[count] = (value - minLX) / (maxLX - minLX)
#     count += 1
#
# # NH value gt 23
# count = 0
# for value in gt23NHL12maser:
#     gt23NHL12maser[count] = (value - minL12) / (maxL12 - minL12)
#     count += 1
#
# count = 0
# for value in gt23NHL12nonMaser:
#     gt23NHL12nonMaser[count] = (value - minL12) / (maxL12 - minL12)
#     count += 1
#
# count = 0
# for value in gt23NHLXmaser:
#     gt23NHLXmaser[count] = (value - minLX) / (maxLX - minLX)
#     count += 1
#
# count = 0
# for value in gt23NHLXnonMaser:
#     gt23NHLXnonMaser[count] = (value - minLX) / (maxLX - minLX)
#     count += 1
#
# # NH Values gt 24
# count = 0
# for value in gt24NHL12maser:
#     gt24NHL12maser[count] = (value - minL12) / (maxL12 - minL12)
#     count += 1
#
# count = 0
# for value in gt24NHL12nonMaser:
#     gt24NHL12nonMaser[count] = (value - minL12) / (maxL12 - minL12)
#     count += 1
#
# count = 0
# for value in gt24NHLXmaser:
#     gt24NHLXmaser[count] = (value - minLX) / (maxLX - minLX)
#     count += 1
#
# count = 0
# for value in gt24NHLXnonMaser:
#     gt24NHLXnonMaser[count] = (value - minLX) / (maxLX - minLX)
#     count += 1
#
# # NH Values gt 24.3
# count = 0
# for value in gt243NHL12maser:
#     gt24NHL12maser[count] = (value - minL12) / (maxL12 - minL12)
#     count += 1
#
# count = 0
# for value in gt243NHL12nonMaser:
#     gt24NHL12nonMaser[count] = (value - minL12) / (maxL12 - minL12)
#     count += 1
#
# count = 0
# for value in gt243NHLXmaser:
#     gt243NHLXmaser[count] = (value - minLX) / (maxLX - minLX)
#     count += 1
#
# count = 0
# for value in gt243NHLXnonMaser:
#     gt243NHLXnonMaser[count] = (value - minLX) / (maxLX - minLX)
#
# #################################################
