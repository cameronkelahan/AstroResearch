import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

# This code reads in the 12 micrometer luminosity of Table1
# Put table 1 in as the first sys arg[1]
L12 = genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=7)

plt.hist(L12, 5, density=True, alpha = .5, lw=3, linestyle='dashed', linewidth=1.5, edgecolor='black', label='Table1')
plt.legend()
plt.savefig("L12With5Bins")

# print(type(L12))
# print(len(L12))

# Reads in the Lobs from Table2
# Input table 2 as the second sys arg[2]
Lobs = genfromtxt(sys.argv[2], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=4,)

plt.hist(Lobs, 5, lw=700, linestyle='dotted', linewidth=1.5, density=True, alpha = 0.5, edgecolor='black', label='Table2')
plt.legend()
plt.savefig("LobsWith5Bins")

# print(len(Lobs))

# Read in Column 6 from Table1 (Maser Classification)
maserType = genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=6)

# Create a list of nonMasers and masers
nonMasers = []
masers = []

count = 0
for value in maserType:
    if maserType[count] == 0:
        nonMasers.append(L12.item(count))
        nonMasers.append(Lobs.item(count))
    else:
        masers.append(L12.item(count))
        masers.append(Lobs.item(count))
    count += 1


# Create a combined list of both the X component and Y components
LTotal = np.concatenate([L12, Lobs])

plt.hist(LTotal, 5, density=True, alpha = 0.5, edgecolor='black', linewidth=1.5, label='CombinedTable')
plt.legend()
plt.savefig("LTotalWith5Bins")


print(len(LTotal))

# Create a Histogram of nonMasers
plt.close()
plt.hist(nonMasers, 15, density=True, alpha=0.5, edgecolor='black', linewidth=1.5, label='NonMasers', linestyle='dotted')
plt.legend()
plt.savefig('NonMasersHistogram15Bins')

# Create a Histogram of nonMasers and Masers
plt.hist(masers, 15, density=True, alpha=0.5, edgecolor='black', linewidth=1.5, label='Masers', linestyle='dashed')
plt.legend()
plt.savefig('AllNonMasersAndMasersHistogram15Bins')

# Create a Histogram of masers
plt.close()
plt.hist(nonMasers, 15, density=True, alpha=0.5, edgecolor='black', linewidth=1.5, label='Masers')
plt.legend()
plt.savefig('MasersHistogram15Bins')