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
# Used for the y-axis
minLX = int(min(LX))
maxLX = int(max(LX))
LX_range = range(minLX - 1, maxLX + 2)

# Used for the x-Axis
minL12 = int(min(L12))
maxL12 = int(max(L12))
L12_range = range(minL12 - 1, maxL12 + 2)

# Currently, the shape of the data matrix is flipped
# Reshape the data matrix to have 3 columns, one for each attribute, and as many rows as there are examples (galaxies)
data = []
count = 0
for value in L12:
    data.append([L12[count], LX[count], NH[count]])
    count += 1

# print("Length of data = ", len(data))
# print("Length of Data[0] = ", len(data[0]))
# print("Length of Data[1] = ", len(data[1]))
# print("Length of Data[2] = ", len(data[2]))
# print("Data[0][0] = ", data[0])
# print("Data[0][1] = ", data[1])
# print("Data[0][2] = ", data[2])
# print("Length of MaserType[] = ", len(maserType))

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

print("Length of gt22NHL12nonMaser = ", len(gt22NHL12nonMaser))
print("Length of gt22NHLXnonMaser = ", len(gt22NHLXnonMaser))
print("Length of gt22NHL12maseraser = ", len(gt22NHL12maser))
print("Length of gt22NHLXmaser = ", len(gt22NHLXmaser))

print("Length of gt23NHL12nonMaser = ", len(gt23NHL12nonMaser))
print("Length of gt23NHLXnonMaser = ", len(gt23NHLXnonMaser))
print("Length of gt23NHL12maser = ", len(gt23NHL12maser))
print("Length of gt23NHLXmaser = ", len(gt23NHLXmaser))

print("Length of gt24NHL12nonMaser = ", len(gt24NHL12nonMaser))
print("Length of gt24NHLXnonMaser = ", len(gt24NHLXnonMaser))
print("Length of gt24NHL12maser = ", len(gt24NHL12maser))
print("Length of gt24NHLXmaser = ", len(gt24NHLXmaser))

print("Length of gt24.3NHL12nonMaser = ", len(gt243NHL12nonMaser))
print("Length of gt24.3NHLXnonMaser = ", len(gt243NHLXnonMaser))
print("Length of gt24.3NHL12maser = ", len(gt243NHL12maser))
print("Length of gt24.3NHLXmaser = ", len(gt243NHLXmaser))

print("Length of lt24.3NHL12nonMaser = ", len(lt243NHL12nonMaser))
print("Length of lt24.3NHLXnonMaser = ", len(lt243NHLXnonMaser))
print("Length of lt24.3NHL12maser = ", len(lt243NHL12maser))
print("Length of lt24.3NHLXmaser = ", len(lt243NHLXmaser))

# Plot each subset of values with unique shapes/colors
# Maybe distinct by maser/non-maser?

LineValueX1 = np.linspace(37, 46, 9)

Line0ValueY1 = NHLines(LineValueX1, change0)

Line22ValueY1 = NHLines(LineValueX1, change22)

Line23ValueY1 = NHLines(LineValueX1, change23)

Line24ValueY1 = NHLines(LineValueX1, change24)

Line243ValueY1 = NHLines(LineValueX1, change243)

#### Plot all values between 22 and 23
plt.figure(figsize=(6.4, 4.8))
plt.plot(LineValueX1, Line22ValueY1, label='NH = 22')
plt.plot(LineValueX1, Line23ValueY1, '-r', label='NH = 23')
plt.scatter(gt23NHL12maser, gt23NHLXmaser, label='Maser Galaxies with \n[22 > NH values > 23]', marker='^', c='orange')
plt.scatter(gt23NHL12nonMaser, gt23NHLXnonMaser, label='NonMaser Galaxies with \n[22 > NH Values > NH 23]', marker='o', c='cyan', alpha=0.25)
# plt.xlim(minL12, maxL12)
# plt.ylim(minLX, maxLX)
plt.xlim(38, 46)
plt.ylim(38, 46)
plt.title('NH Values Between 22 and 23')
plt.xlabel("$L_{12}$")
plt.ylabel("$L_X$")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend()
plt.savefig('NH_22_23_scatterplot')
plt.show()
plt.clf()
plt.close()

#### Plot all values between 23 and 24
plt.figure(figsize=(6.4, 4.8))
plt.plot(LineValueX1, Line23ValueY1, label='NH = 23')
plt.plot(LineValueX1, Line24ValueY1, '-r', label='NH = 24')
plt.scatter(gt24NHL12maser, gt24NHLXmaser, label='Maser Galaxies with \n[23 > NH Values > 24]', marker='^', c='orange')
plt.scatter(gt24NHL12nonMaser, gt24NHLXnonMaser, label='NonMaser Galaxies with \n[23 > NH Values > 24', marker='o', c='cyan', alpha=0.25)
# plt.xlim(minL12, maxL12)
# plt.ylim(minLX, maxLX)
plt.xlim(38, 46)
plt.ylim(38, 46)
plt.title('NH Values Between 23 and 24')
plt.xlabel("$L_{12}$")
plt.ylabel("$L_X$")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend()
plt.savefig('NH_23_24_scatterplot')
plt.show()
plt.clf()
plt.close()

#### Plot all values between 24 and 24.3
plt.figure(figsize=(6.4, 4.8))
# plt.xlim(minL12, maxL12)
# plt.ylim(minLX, maxLX)
plt.xlim(38, 46)
plt.ylim(38, 46)
plt.plot(LineValueX1, Line24ValueY1, label='NH = 24')
plt.plot(LineValueX1, Line243ValueY1, '-r', label='NH = 24.3')
plt.scatter(gt243NHL12maser, gt243NHLXmaser, label='Maser Galaxies with \n[24 > NH Values > 24.3]', marker='^', c='orange')
plt.scatter(gt243NHL12nonMaser, gt243NHLXnonMaser, label='NonMaser Galaxies with \n[24 > NH Values > 24.3]', marker='o', c='cyan', alpha=0.25)
plt.title('NH Values Between 24 and 24.3')
plt.xlabel("$L_{12}$")
plt.ylabel("$L_X$")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend()
plt.savefig('NH_24_243_scatterplot')
plt.show()
plt.clf()
plt.close()

#### Plot all values and all lines between NH = 22 and NH = 24.3
plt.figure(figsize=(6.4, 4.8))
# plt.xlim(minL12, maxL12)
# plt.ylim(minLX, maxLX)
plt.xlim(38, 46)
plt.ylim(38, 46)
plt.plot(LineValueX1, Line22ValueY1, label='NH = 22')

plt.plot(LineValueX1, Line23ValueY1, 'orange', label='NH = 23')

plt.plot(LineValueX1, Line24ValueY1, '-g', label='NH = 24')

plt.plot(LineValueX1, Line243ValueY1, '-r', label='NH = 24.3')

plt.scatter(gt23NHL12maser, gt23NHLXmaser, label='Maser Galaxies', marker='s', c='red')
plt.scatter(gt23NHL12nonMaser, gt23NHLXnonMaser, label='NonMaser Galaxies', marker='o', c='green', alpha=0.25)

plt.scatter(gt24NHL12maser, gt24NHLXmaser, label='Maser Galaxies', marker='*', c='purple')
plt.scatter(gt24NHL12nonMaser, gt24NHLXnonMaser, label='NonMaser Galaxies', marker='o', c='black', alpha=0.25)

plt.scatter(gt243NHL12maser, gt243NHLXmaser, label='Maser Galaxies', marker='^', c='orange')
plt.scatter(gt243NHL12nonMaser, gt243NHLXnonMaser, label='NonMaser Galaxies', marker='o', c='cyan', alpha=0.25)
plt.title('NH Values Between 22 and 24.3')
plt.xlabel("$L_{12}$")
plt.ylabel("$L_X$")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend()
plt.savefig('NH_All_22_243_scatterplot')
plt.show()
plt.clf()
plt.close()
