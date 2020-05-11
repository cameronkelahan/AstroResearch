import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal
from numpy import genfromtxt

# Read in Column 6 from Table1 (Maser Classification)
maserType = genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=6)

# Read in L12 from Table1
L12 = genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=7)

# Read in Lobs from Table2
Lobs = genfromtxt(sys.argv[2], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=4)

# Normalize the L12 Data
maxValL12 = np.amax(L12)
minValL12 = np.amin(L12)
countL12 = 0
for value in L12:
    L12[countL12] = (value - minValL12) / (maxValL12 - minValL12)
    countL12 += 1

# Normalize the data in Lobs
maxValLobs = np.amax(Lobs)
minValLobs = np.amin(Lobs)
countLobs = 0
for value in Lobs:
    Lobs[countLobs] = (value - minValLobs) / (maxValLobs - minValLobs)
    countLobs += 1

nonMasersL12 = []
masersL12 = []

nonMasersLobs = []
masersLobs = []

# Sort the masers from non masers
count = 0
for value in maserType:
    if maserType[count] == 0:
        nonMasersL12.append(L12.item(count))
        nonMasersLobs.append(Lobs.item(count))
    else:
        masersL12.append(L12.item(count))
        masersLobs.append(Lobs.item(count))
    count += 1

# print(len(nonMasersL12))
# print(len(nonMasersLobs))
# print(len(masersL12))
# print(len(masersLobs))

############################## Plot Univariate Gaussian Curves #########################################################

# # Plot a Univariate Gaussian curve for nonMasers using L12
# upperLimit = np.amax(nonMasersL12)
# lowerLimit = np.amin(nonMasersL12)
# mu = np.mean(nonMasersL12)
# print("MU = ", mu)
# variance = np.var(nonMasersL12)
# print("Variance = ", variance)
# sigma = math.sqrt(variance)
# print("Sigma = ", sigma)
# x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
# print("X = ", x)
# plt.plot(x, norm.pdf(x, mu, sigma), label='nonMasersL12')
# plt.legend()
# plt.savefig("NonMasersGaussianL12NewGauss")
#
#
# #Plot a Univariate Gaussian curve for masers on top of the nonMaser curve using L12
# upperLimit = np.amax(masersL12)
# lowerLimit = np.amin(masersL12)
# mu = np.mean(masersL12)
# variance = np.var(masersL12)
# sigma = math.sqrt(variance)
# x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
# plt.plot(x, norm.pdf(x, mu, sigma), label='masersL12')
# plt.legend()
# plt.savefig("MasersAndNonMasersGaussianL12")
#
# plt.close()
#
# # Plot a Univariate Gaussian curve for nonMasers using Lobs
# upperLimit = np.amax(nonMasersLobs)
# lowerLimit = np.amin(nonMasersLobs)
# mu = np.mean(nonMasersLobs)
# variance = np.var(nonMasersLobs)
# sigma = math.sqrt(variance)
# x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
# plt.plot(x, norm.pdf(x, mu, sigma), label='nonMasersLobs')
# plt.legend()
# plt.savefig("NonMasersGaussianLobs")
#
# #Plot a Univariate Gaussian curve for masers on top of the nonMaser curve using L12
# upperLimit = np.amax(masersLobs)
# lowerLimit = np.amin(masersLobs)
# mu = np.mean(masersLobs)
# variance = np.var(masersLobs)
# sigma = math.sqrt(variance)
# x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
# plt.plot(x, norm.pdf(x, mu, sigma), label='masersLobs')
# plt.legend()
# plt.savefig("MasersAndNonMasersGaussianLobs")

######################################## Plot Multivariate Gaussian Curves #############################################
# Plot a multivariate Gaussian curve for NonMasers using L12 and Lobs
# X axis is L12
# Y axis is Lobs
muXNonMasers = np.mean(nonMasersL12)
muYNonMasers = np.mean(nonMasersLobs)
varianceXNonMasers = np.var(nonMasersL12)
varianceYNonMasers = np.var(nonMasersLobs)
covarianceNonMasers = np.cov(nonMasersL12, nonMasersLobs)
sigmaXNonMasers = math.sqrt(varianceXNonMasers)
sigmaYNonMasers = math.sqrt(varianceYNonMasers)

# Set the relative lower and upper X bounds of the equation
lowerBoundXNonMasers = muXNonMasers - 3*sigmaXNonMasers
upperBoundXNonMasers = muXNonMasers + 3*sigmaXNonMasers
rangeX = (upperBoundXNonMasers - lowerBoundXNonMasers) / 100

# Set the relative lower and upper Y bounds of the equation
lowerBoundYNonMasers = muYNonMasers - 3*sigmaYNonMasers
upperBoundYNonMasers = muYNonMasers + 3*sigmaYNonMasers
rangeY = (upperBoundYNonMasers - lowerBoundYNonMasers) / 100

x, y = np.mgrid[lowerBoundXNonMasers:upperBoundXNonMasers:rangeX,
                lowerBoundYNonMasers:upperBoundYNonMasers:rangeY]
pos = np.dstack((x, y))
rv = multivariate_normal([muXNonMasers, muYNonMasers], covarianceNonMasers)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
nonMaserPlot = ax2.contourf(x, y, rv.pdf(pos))

# Plot a non-maser scatter plot over top of the Multivariate Gaussian plot
plt.scatter(nonMasersL12, nonMasersLobs, c='orange')

plt.savefig("MVGnonMaserScatter")

# Close the previous plot to make a fresh one
plt.close()

# Plot a multivariate Gaussian curve for Masers using L12 and Lobs
muXMasers = np.mean(masersL12)
muYMasers = np.mean(masersLobs)
varianceXMasers = np.var(masersL12)
varianceYMasers = np.var(masersLobs)
covarianceMasers = np.cov(masersL12, masersLobs)
sigmaXMasers = math.sqrt(varianceXMasers)
sigmaYMasers = math.sqrt(varianceYMasers)

# Set the relative lower and upper X bounds of the equation
lowerBoundXMasers = muXMasers - 3*sigmaXMasers
upperBoundXMasers = muXMasers + 3*sigmaXMasers
rangeX = (upperBoundXMasers - lowerBoundXMasers) / 100

# Set the relative lower and upper Y bounds of the equation
lowerBoundYMasers = muYMasers - 3*sigmaYMasers
upperBoundYMasers = muYMasers + 3*sigmaYMasers
rangeY = (upperBoundYMasers - lowerBoundYMasers) / 100

x, y = np.mgrid[lowerBoundXMasers:upperBoundXMasers:rangeX,
                lowerBoundYMasers:upperBoundYMasers:rangeY]
pos = np.dstack((x, y))
rv = multivariate_normal([muXMasers, muYMasers], covarianceMasers)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
maserPlot = ax2.contourf(x, y, rv.pdf(pos))

# Plot a maser scatter plot over top of the Multivariate Gaussian plot
plt.scatter(masersL12, masersLobs, c='orange')

plt.savefig("MVGmaserScatter")