import numpy as np
import sys
from keras.callbacks import ModelCheckpoint, CSVLogger, LambdaCallback

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn import metrics
from numpy import genfromtxt
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm

# List 1 and 2 here should be the nonmaser and maser list pertaining to the same parameter (i.e. L12, Lx, etc.)
# List1name should be masers___ where ___ is the parameter (i.e. nonmasersL12) Used for the label parameter of plot
# xBalel should be string called 'Normalized ____', where ____ is the passed paramter (i.e. L12, Lx, etc.)
def plot(parentListMasers, parentListNonMasers, childListMasers, childListNonMasers, parentListMasersName,
         parentListNonMasersName, childListMasersName, childListNonMasersName, xLabel, saveName, title):
    ######################## Plot a Univariate Gaussian curve for 2 parameters

    # Calculate the Mu, variance, and sigma of parent list masers
    mu = np.mean(parentListMasers)
    variance = np.var(parentListMasers)
    sigma = math.sqrt(variance)

    # Create an x-axis based on the mu and sigma values
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)

    # # Use Sturge's Rule to calculate correct number of bins based on data size of nonmaser dataset
    # # Cast to int for a whole number
    # # For this dataset, numBins = 10
    # numBins = int(1 + 3.322 * math.log10(len(list2)))
    #
    # # Create a Histogram of nonMasers and Masers using their attributes
    # # Plot over the gaussian curves too
    # # Separate into 10 bins
    # # Alpha of 0.5 for translucence
    # plt.hist(list1, numBins, density=True, alpha=0.5, edgecolor='black', linewidth=1.5, label='NonMasers')
    # plt.hist(list2, numBins, density=True, alpha=0.5, edgecolor='black', linewidth=1.5, label='Masers',
    #          linestyle='dashed')

    # Plot the curve
    plt.plot(x, norm.pdf(x, mu, sigma), label=parentListMasersName, linestyle='dotted')
    plt.xlabel(xLabel)
    plt.title(title)
    # Unicode for Delta symbol
    plt.ylabel('$\u0394N/N_{total}$')
    plt.legend()

    # Plot a Univariate Gaussian curve for parent list nonMaser on top of the parent Maser curve
    mu = np.mean(parentListNonMasers)
    variance = np.var(parentListNonMasers)
    sigma = math.sqrt(variance)

    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)

    # Plot the parent nonMaser L12 curve on top of the parent maser L12 curve
    plt.plot(x, norm.pdf(x, mu, sigma), label=parentListNonMasersName, linestyle='dashdot')

    # Plot a Univariate Gaussian curve for the child list masers on top of the parent curves
    mu = np.mean(childListMasers)
    variance = np.var(childListMasers)
    sigma = math.sqrt(variance)

    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)

    # Plot the child maser L12 curve on top of the other curves
    plt.plot(x, norm.pdf(x, mu, sigma), label=childListMasersName, linestyle='dashed')

    # Plot a Univariate Gaussian curve for the child list nonMasers on top of the parent curves and child maser curve
    mu = np.mean(childListNonMasers)
    variance = np.var(childListNonMasers)
    sigma = math.sqrt(variance)

    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)

    # Plot the child nonMaser L12 curve on top of the other curves
    plt.plot(x, norm.pdf(x, mu, sigma), label=childListNonMasersName)

    plt.legend()
    plt.savefig(saveName, dpi=400, bbox_inches='tight', pad_inches=0.05)
    plt.show()

    plt.close()
###############

# perform undersampling based on the entire dataset's gaussian distribution
def undersampling(parentSet, size):
    mu = np.mean(parentSet)
    variance = np.var(parentSet)
    sigma = math.sqrt(variance)

    selection = np.random.normal(mu, sigma, size)
    print(selection)

    return selection
###############

def main():
    # Reads in the 12 micrometer luminosity (L12) of Table1
    # Put table 1 in as the first sys arg[1]
    L12 = genfromtxt('Paper2Table1.csv', delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=7)

    # Reads in the X-ray Luminosity (LX) from Table2
    # Input table 2 as the second sys arg[2]
    LX = genfromtxt('Paper2Table2.csv', delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=4,)

    # Read in Column 6 from Table1 (Maser Classification)
    maserType = genfromtxt('Paper2Table1.csv', delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=6)

    # Read in W1 from Table1WISE
    W1 = genfromtxt('Paper2Table1WISE.csv', delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=11)

    # Read in W1 from Table1WISE
    W2 = genfromtxt('Paper2Table1WISE.csv', delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=13)

    # Read in W1 from Table1WISE
    W3 = genfromtxt('Paper2Table1WISE.csv', delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=15)

    # Read in W1 from Table1WISE
    W4 = genfromtxt('Paper2Table1WISE.csv', delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=17)

    # Calculate difference in W1 and W2 colors (W1 - W2)
    W1_W2List = []
    count = 0
    for value in W1:
        W1_W2List.append(value - W2[count])
        count += 1

    # Calculate difference in W2 and W3 colors (W2 - W3)
    W2_W3List = []
    count = 0
    for value in W2:
        W2_W3List.append(value - W3[count])
        count += 1

    # Calculate difference in W3 and W4 colors (W3 - W4)
    W1_W4List = []
    count = 0
    for value in W1:
        W1_W4List.append(value - W4[count])
        count += 1

    # Create a list of attributes for nonMasers and masers
    nonMasersL12 = []
    masersL12 = []
    nonMasersLX = []
    masersLX = []
    nonMasersW1_W2 = []
    masersW1_W2 = []
    nonMasersW2_W3 = []
    masersW2_W3 = []
    nonMasersW1_W4 = []
    masersW1_W4 = []

    # Separate galaxies into maser and nonmaser for L12 and LX attributes
    count = 0
    for value in maserType:
        if maserType[count] == 0:
            nonMasersL12.append(L12.item(count))
            nonMasersLX.append(LX.item(count))
            nonMasersW1_W2.append(W1_W2List[count])
            nonMasersW2_W3.append(W2_W3List[count])
            nonMasersW1_W4.append(W1_W4List[count])
        else:
            masersL12.append(L12.item(count))
            masersLX.append(LX.item(count))
            masersW1_W2.append(W1_W2List[count])
            masersW2_W3.append(W2_W3List[count])
            masersW1_W4.append(W1_W4List[count])
        count += 1

    # plot(masersL12, nonMasersL12, 'masersL12', 'nonMasersL12', 'log $L_{12}$ (erg $s^{-1}$)',
    #      'AllGalaxiesL12Histogram.pdf', 'Entire Data Set L12 Galaxies')
    # plot(masersLX, nonMasersLX, 'log $L_X$ (erg $s^{-1}$)', 'AllGalaxiesLXHistogram.pdf')
    # plot(masersW1_W2, nonMasersW1_W2, 'W1-W2', 'AllGalaxiesW1-W2Histogram.pdf')
    # plot(masersW2_W3, nonMasersW2_W3, 'W2-W3', 'AllGalaxiesW2-W3Histogram.pdf')
    # plot(masersW1_W4, nonMasersW1_W4, 'W1-W4', 'AllGalaxiesW1-W4Histogram.pdf')

    masersTestListL12 = undersampling(masersL12, 14)
    nonMasersTestListL12 = undersampling(nonMasersL12, 14)
    plot(masersL12, nonMasersL12, masersTestListL12, nonMasersTestListL12, 'parentMasersL12', 'parentNonMasersL12',
         'childMasersL12', 'childNonMasersL12', 'log $L_{12}$ (erg $s^{-1}$)',
         'ParentChildTestGalaxiesL12Gaussians.pdf', 'Parent and Child Sets Gaussian Curves for L12')



if __name__ == '__main__':
    main()