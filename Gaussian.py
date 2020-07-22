import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal
from numpy import genfromtxt

def normalizeValues(min, max, list):
    count = 0
    for value in list:
        list[count] = (value - min) / (max - min)
        count += 1
    return list

# List 1 and 2 here should be the nonmaser and maser list pertaining to the same parameter (i.e. L12, Lx, etc.)
# List1name should be masers___ where ___ is the parameter (i.e. nonmasersL12) Used for the label parameter of plot
# xBalel should be string called 'Normalized ____', where ____ is the passed paramter (i.e. L12, Lx, etc.)
def plot(list1, list2, list1name, list2name, xLabel, saveName):
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

def main():
    # Read in Column 6 from Table1 (Maser Classification)
    maserType = genfromtxt('Paper2Table1.csv', delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=6)

    # Read in L12 from Table1
    L12 = genfromtxt('Paper2Table1.csv', delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=7)

    # Read in LX from Table2
    LX = genfromtxt('Paper2Table2.csv', delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=4)

    # Read in W1 from Table1WISE
    W1 = genfromtxt('Paper2Table1WISE.csv', delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=11)

    # Read in W1 from Table1WISE
    W2 = genfromtxt('Paper2Table1WISE.csv', delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=13)

    # Read in W1 from Table1WISE
    W3 = genfromtxt('Paper2Table1WISE.csv', delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=15)

    # Read in W1 from Table1WISE
    W4 = genfromtxt('Paper2Table1WISE.csv', delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=17)

    print("Length of W1 = ", len(W1))
    print("Length of W2 = ", len(W2))
    print("Length of W3 = ", len(W3))
    print("Length of W4 = ", len(W4))

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

    # Normalize the L12 Data
    maxValL12 = np.amax(L12)
    minValL12 = np.amin(L12)
    L12 = normalizeValues(minValL12, maxValL12, L12)

    # Normalize the data in LX
    maxValLX = np.amax(LX)
    minValLX = np.amin(LX)
    LX = normalizeValues(minValLX, maxValLX, LX)

    # Normalize the data in W1_W2List
    maxValW1_W2 = np.amax(W1_W2List)
    minValW1_W2 = np.amin(W1_W2List)
    W1_W2List = normalizeValues(minValW1_W2, maxValW1_W2, W1_W2List)

    # Normalize the data in W2_W3List
    maxValW2_W3 = np.amax(W2_W3List)
    minValW2_W3 = np.amin(W2_W3List)
    W2_W3List = normalizeValues(minValW2_W3, maxValW2_W3, W2_W3List)

    # Normalize the data in W3_W4List
    maxValW1_W4 = np.amax(W1_W4List)
    minValW1_W4 = np.amin(W1_W4List)
    W1_W4List = normalizeValues(minValW1_W4, maxValW1_W4, W1_W4List)

    # Create lists for nonMasers and masers in their attributes: L12, LX, W1_W2, W2_W3, W1_W4
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

    # Sort the masers from non masers
    # Separate into LX and L12 lists
    count = 0
    for value in maserType:
        if maserType[count] == 0:
            nonMasersL12.append(L12.item(count))
            nonMasersLX.append(LX.item(count))
            nonMasersW1_W2.append(W1_W2List[count])
            nonMasersW2_W3.append(W2_W3List[count])
            nonMasersW1_W4.append((W1_W4List[count]))
        else:
            masersL12.append(L12.item(count))
            masersLX.append(LX.item(count))
            masersW1_W2.append(W1_W2List[count])
            masersW2_W3.append(W2_W3List[count])
            masersW1_W4.append(W1_W4List[count])
        count += 1

    ############################## Plot Univariate Gaussian Curves #########################################################

    # Plot a Univariate Gaussian curve for L12 comparing masers and nonmasers
    plot(nonMasersL12, masersL12, 'nonMasersL12', 'masersL12', 'Normalized $L_{12}$',
         'MasersAndNonMasersGaussianL12.pdf')

    # Plot a Univariate Gaussian curve for LX comparing masers and nonmasers
    plot(nonMasersLX, masersLX, 'nonMasersLX', 'masersLX', 'Normalized $L_{X}$',
         'MasersAndNonMasersGaussianLX.pdf')

    # Plot a Univariate Gaussian curve for W1-W2 comparing masers and nonmasers
    plot(nonMasersW1_W2, masersW1_W2, 'nonMasersW1-W2', 'masersW1-W2', 'Normalized $L_{W1-W2}$',
         'MasersAndNonMasersGaussianW1-W2.pdf')

    # Plot a Univariate Gaussian curve for W2-W3 comparing masers and nonmasers
    plot(nonMasersW2_W3, masersW2_W3, 'nonMasersW2-W3', 'masersW2-W3', 'Normalized $L_{W2-W3}$',
         'MasersAndNonMasersGaussianW2-W3.pdf')

    # Plot a Univariate Gaussian curve for W1-W4 comparing masers and nonmasers
    plot(nonMasersW1_W4, masersW1_W4, 'nonMasersW1-W4', 'masersW1-W4', 'Normalized $L_{W1-W4}$',
         'MasersAndNonMasersGaussianW1-W4.pdf')

    ######################################## Plot Multivariate Gaussian Curves #############################################
    # Unused in the paper
    # Unchecked code below

    # # Plot a multivariate Gaussian curve for NonMasers using L12 and Lobs
    # # X axis is L12
    # # Y axis is Lobs
    # muXNonMasers = np.mean(nonMasersL12)
    # muYNonMasers = np.mean(nonMasersLX)
    # varianceXNonMasers = np.var(nonMasersL12)
    # varianceYNonMasers = np.var(nonMasersLX)
    # covarianceNonMasers = np.cov(nonMasersL12, nonMasersLX)
    # sigmaXNonMasers = math.sqrt(varianceXNonMasers)
    # sigmaYNonMasers = math.sqrt(varianceYNonMasers)
    #
    # # Set the relative lower and upper X bounds of the equation
    # lowerBoundXNonMasers = muXNonMasers - 3*sigmaXNonMasers
    # upperBoundXNonMasers = muXNonMasers + 3*sigmaXNonMasers
    # rangeX = (upperBoundXNonMasers - lowerBoundXNonMasers) / 100
    #
    # # Set the relative lower and upper Y bounds of the equation
    # lowerBoundYNonMasers = muYNonMasers - 3*sigmaYNonMasers
    # upperBoundYNonMasers = muYNonMasers + 3*sigmaYNonMasers
    # rangeY = (upperBoundYNonMasers - lowerBoundYNonMasers) / 100
    #
    # x, y = np.mgrid[lowerBoundXNonMasers:upperBoundXNonMasers:rangeX,
    #                 lowerBoundYNonMasers:upperBoundYNonMasers:rangeY]
    # pos = np.dstack((x, y))
    # rv = multivariate_normal([muXNonMasers, muYNonMasers], covarianceNonMasers)
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111)
    # nonMaserPlot = ax2.contourf(x, y, rv.pdf(pos))
    #
    # # Plot a non-maser scatter plot over top of the Multivariate Gaussian plot
    # plt.scatter(nonMasersL12, nonMasersLX, c='orange')
    #
    # plt.savefig("MVGnonMaserScatter")
    #
    # # Close the previous plot to make a fresh one
    # plt.close()
    #
    # # Plot a multivariate Gaussian curve for Masers using L12 and Lobs
    # muXMasers = np.mean(masersL12)
    # muYMasers = np.mean(masersLX)
    # varianceXMasers = np.var(masersL12)
    # varianceYMasers = np.var(masersLX)
    # covarianceMasers = np.cov(masersL12, masersLX)
    # sigmaXMasers = math.sqrt(varianceXMasers)
    # sigmaYMasers = math.sqrt(varianceYMasers)
    #
    # # Set the relative lower and upper X bounds of the equation
    # lowerBoundXMasers = muXMasers - 3*sigmaXMasers
    # upperBoundXMasers = muXMasers + 3*sigmaXMasers
    # rangeX = (upperBoundXMasers - lowerBoundXMasers) / 100
    #
    # # Set the relative lower and upper Y bounds of the equation
    # lowerBoundYMasers = muYMasers - 3*sigmaYMasers
    # upperBoundYMasers = muYMasers + 3*sigmaYMasers
    # rangeY = (upperBoundYMasers - lowerBoundYMasers) / 100
    #
    # x, y = np.mgrid[lowerBoundXMasers:upperBoundXMasers:rangeX,
    #                 lowerBoundYMasers:upperBoundYMasers:rangeY]
    # pos = np.dstack((x, y))
    # rv = multivariate_normal([muXMasers, muYMasers], covarianceMasers)
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111)
    # maserPlot = ax2.contourf(x, y, rv.pdf(pos))
    #
    # # Plot a maser scatter plot over top of the Multivariate Gaussian plot
    # plt.scatter(masersL12, masersLX, c='orange')
    #
    # plt.savefig("MVGmaserScatter")

if __name__ == '__main__':
    main()
