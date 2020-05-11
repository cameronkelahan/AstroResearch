import csv
import re
import sys
import numpy as np
from numpy import genfromtxt
from astropy import units as u
from astropy.coordinates import SkyCoord

#######################################################################################################################
# opens a .txt file and separates rows by tabs
# turns any whitespace values into a comma instead
# removes new lines

with open(sys.argv[1]) as file:
    content = file.readlines()
    rows = range(0, len(content))
    for row in rows:
        # print(content[row])
        content[row] = content[row][2:]
        content[row] = re.sub("\s+", ",", content[row])
        content[row] = content[row][:len(content[row]) - 1]
        # content[row] = re.sub("\n{1}", "", content[row])
        # print("after")
        # print(content[row])

    with open(sys.argv[2], mode='w') as writeFile:
        for row in rows:
            print(content[row])
            writeFile.write(content[row])
            writeFile.write("\n")

#######################################################################################################################

# Takes a csv file and produces a csv file
# This functionality coverts RA and Dec into degrees
# Used for maserGalaxyCatalog.csv
# count2019 = genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=0)
# name = genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=1)
# ra = genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=2)
# dec = genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=3)
# vSys = genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=4)
# lum = genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=5)
# classType = genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=6)
#
# with open(sys.argv[2], mode='w') as writeFile:
#     writeFile.write('count2019, Source_Name, RA_(J2000), Dec_(J2000), Vsys_(km/s), Lum, Class\n')
#
#     rows = range(ra.size)
#     for row in rows:
#         converted = SkyCoord(ra[row], dec[row], unit=(u.hourangle, u.deg))
#         # print(converted.ra.degree)
#         # print(converted.dec.degree)
#         ra[row] = converted.ra.degree
#         dec[row] = converted.dec.degree
#
#         string = "{},{},{},{},{},{},{}\n".format(count2019[row],name[row],ra[row],dec[row],vSys[row],lum[row],
#                      classType[row])
#         writeFile.write(string)

######################################################################################################################

# Took the huge galaxy dataset and turned it into a csv, adding question marks to blank spots
#
# with open(sys.argv[1], 'r') as file:
#     content = file.readlines()
#     rows = range(0, len(content))
#     count = 0;
#
#     # stringg = "        Hello      "
#     # print(stringg)
#     # stringg.lstrip()
#     # print(stringg)
#
#     content[0] = "Count2019" + content[0]
#
#     for row in rows:
#
#         content[row] = re.sub("\n{1}", "", content[row])
#         content[row] = content[row].strip()
#         content[row] = content[row].split()
#         if len(content[row]) == 14:
#             content[row].append('?')
#             content[row].append('?')
#             content[row].append('?')
#         content[row][1:4] = [' '.join(content[row][1:4])]
#         content[row][2:5] = [' '.join(content[row][2:5])]
#         content[row][1:3] = [' '.join(content[row][1:3])]
#         content[row] = ','.join(content[row])
#         content[row] = str(count) + "," + content[row]
#         print(content[row])
#
#         count += 1
#
#     content[0] = content[0].lstrip('0,')
#
#     with open(sys.argv[2], 'w') as writeFile:
#         for row in rows:
#             writeFile.write(content[row])
#             writeFile.write("\n")

#######################################################################################################################

# Take the huge galaxy dataset csv and convert the RA and DEC
# Line 5741 in TotalGalaxyCSV is only missing RMS2

# count2019 = genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=0)
# name = genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=1)
# radec = genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=2)
# velocity = genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=3)
# date_obs = genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=4)
# tsys = genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=5)
# intList = genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=6)
# rms1 = genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=7)
# vlo1 = genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=8)
# vhi1 = genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=9)
# rms2 = genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=10)
# vlo2 = genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=11)
# vhi2 = genfromtxt(sys.argv[1], delimiter=',', skip_header=1, dtype=None, encoding="utf8", usecols=12)
#
# with open(sys.argv[2], mode='w') as writeFile:
#     writeFile.write('count2019, Source_Name, RA, Dec, Velo, Date-Obs, Tsys, Int, Rms1, Vlo1, Vhi1, Rms2, Vlo2, Vhi2\n')
#
#     rows = range(radec.size)
#     for row in rows:
#         converted = SkyCoord(radec[row], unit=(u.hourangle, u.deg))
#         # print(converted.ra.degree)
#         # print(converted.dec.degree)
#         radec[row] = converted.to_string('decimal')
#         listRaDec = radec[row].split(' ')
#
#         print(listRaDec)
#
#         string = "{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(count2019[row],name[row],listRaDec[0],listRaDec[1],velocity[row],
#                                                                    date_obs[row],tsys[row],intList[row],rms1[row],
#                                                                    vlo1[row],vhi1[row],rms2[row],vlo2[row],vhi2[row])
#         writeFile.write(string)