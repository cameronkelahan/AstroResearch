import math
import matplotlib.pyplot as plt
from numpy import genfromtxt

# This file creates histograms to compare the probability density of masers and nonmasers
# for all attributes being examined: L12, LX, W1-W2, W2-W3, and W1-W4

# List1 is the maser list of the pattribute
# List 2 is the nonmaser list of the attribute
def plot(list1, list2, xLabel, saveName):
    # Plot a histogram of each galaxy attribute individually; map maser and nonmaser histograms on top of each other

    # Use Sturge's Rule to calculate correct number of bins based on data size of nonmaser dataset
    # Cast to int for a whole number
    # For this dataset, numBins = 10
    numBins = int(1 + 3.322 * math.log10(len(list2)))

    # Create a Histogram of nonMasers and Masers using their L12 attribute
    # Separate into 10 bins
    # Alpha of 0.5 for translucence
    plt.hist(list2, numBins, density=True, alpha=0.5, edgecolor='black', linewidth=1.5, label='NonMasers')
    plt.hist(list1, numBins, density=True, alpha=0.5, edgecolor='black', linewidth=1.5, label='Masers',
             linestyle='dashed')
    plt.xlabel(xLabel)
    # Unicode for Delta symbol
    plt.ylabel('$\u0394N/N_{total}$')
    # Unicode for Micro symbol
    plt.legend()
    plt.savefig(saveName, dpi=400, bbox_inches='tight', pad_inches=0.05)
    plt.show()

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

    plot(masersL12, nonMasersL12, 'log $L_{12}$ (erg $s^{-1}$)', 'AllGalaxiesL12Histogram.pdf')
    plot(masersLX, nonMasersLX, 'log $L_X$ (erg $s^{-1}$)', 'AllGalaxiesLXHistogram.pdf')
    plot(masersW1_W2, nonMasersW1_W2, 'W1-W2', 'AllGalaxiesW1-W2Histogram.pdf')
    plot(masersW2_W3, nonMasersW2_W3, 'W2-W3', 'AllGalaxiesW2-W3Histogram.pdf')
    plot(masersW1_W4, nonMasersW1_W4, 'W1-W4', 'AllGalaxiesW1-W4Histogram.pdf')

if __name__ == '__main__':
    main()
