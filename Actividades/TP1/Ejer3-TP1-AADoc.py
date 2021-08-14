# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 10:40:25 2021

@author: Lucas
"""

import os
import numpy as np

from downloadECG import loadParams
from utils import pasaBanda, notch, plotECG, peakDetection, plotSeveralECG, splitECG
from Ejers1y2_TP1_AADoc import findingClusters

#Load data
initialFolder = os.getcwd()
path = os.path.join(initialFolder,"mitdata")
os.chdir(path)

ecg = np.fromfile('100.dat', dtype = "byte")    
os.chdir(initialFolder) #vuelvo al directorio de inicio

params = loadParams("atributosECG-TP1.txt") #

#Starting point
fm = 360.

#filtering the signal
ecg = ecg - np.mean(ecg)
filteredECG = notch(ecg, removeFrec = 50., fm = fm)
filteredECG = notch(ecg, removeFrec = 20., fm = fm)
filteredECG = pasaBanda(filteredECG, lfrec = 0.1, hfrec = 45., order = 4, fm = fm)

#select a window
window = 1. #seg

#select time for cut the ECG signal
iniTime = 0. #secs
finalTime = 100. #secs

shortECG = filteredECG[int(iniTime*fm):int(finalTime*fm)]

# plot a piece of ECG
# plotECG(shortECG, fm = 360., iniTime = iniTime, finalTime = finalTime,
#         title = "ECG", savePlots = False, folder = "figs")

std = np.std(filteredECG)
threshold = 7*std #selecciono un umbral de 7 veces el std de la señal (¿por qué?, no hay por que :D )
#La selección del umbral fue realizada arbitrariamente.

peaks, indexes, avg = peakDetection(shortECG, window = 1.0, threshold = threshold,
                               fm = fm, startingIndex = int(iniTime*fm))   

#Plot the EEG with peak markers
plotECG(shortECG, fm = 360., iniTime = iniTime, finalTime = finalTime,
        markers = indexes, label = f"frecuencia {int(avg)}lpm",
        title = "ECG con picos encontrados", savePlots = True, folder = "figs")   


#Split the ECG data
splitWindow = 1 #secs
 
# plotECG(splittedecg[5][:], fm = 360., iniTime = 0, finalTime = 0,
#         title = "Splitted ECG", savePlots = False, folder = "figs")

splittedecg = splitECG(shortECG, peakIndexes = indexes, splitWindow = splitWindow, fm = fm)
#ecg, splitWindow, peakIndexes, fm = 360.

kClusters = 4
centersECG, labels = findingClusters(splittedecg, kClusters = kClusters, centers = None)

rows = int(np.ceil(kClusters/2))
# columns = int(np.ceil(len(np.unique(labels))//2))
if kClusters > 4 and kClusters % rows:
    columns = kClusters//rows+1
else:
    columns = kClusters//rows

plotSeveralECG(centersECG, labels = np.unique(labels),
                fm = 360., rows = rows, columns = columns,
                    iniTime = 0., finalTime = 0.,markers = [],
                    title = "Representative ECG signals for each cluster", savePlots = True,
                    folder = "figs")

plotSeveralECG(splittedecg, labels = labels,
                fm = 360., rows = rows, columns = columns,
                    iniTime = 0., finalTime = 0.,markers = [],
                    title = "Groups of ECG signals considering the cluster membership",
                    savePlots = True,
                    folder = "figs")

# kClusters = 4
# centersECG, labels = findingClusters(splittedecg, kClusters = kClusters, centers = None)

# rows = int(kClusters/2)
# columns = int(len(np.unique(labels))/2)

# plotSeveralECG(centersECG, labels = list(np.arange(0,kClusters)),
#                 fm = 360., rows = 4, columns = 2,
#                     iniTime = 0., finalTime = 0.,markers = [],
#                     title = "Representative ECG signals", savePlots = False,
#                     folder = "figs")

# plotSeveralECG(splittedecg, labels = labels,
#                 fm = 360., rows = 1, columns = len(np.unique(labels)),
#                     iniTime = 0., finalTime = 0.,markers = [],
#                     title = "Groups of ECG signals considering the cluster membership",
#                     savePlots = True,
#                     folder = "figs")
