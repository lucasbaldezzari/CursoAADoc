# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 12:05:38 2021

@author: Lucas
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from scipy.signal import butter, filtfilt, iirnotch

def pasaBanda(canal, lfrec, hfrec, order, fm = 360.0):
    '''
    Filtra la señal entre las frecuencias de corte lfrec (inferior) y hfrec (superior).
    Filtro del tipo "pasa banda"

    Argumentoss:
        - signal (numpy.ndarray): Arreglo con los datos de la señal
        - lfrec (float): frecuencia de corte baja (Hz).
        - hfrec (float): frecuencia de corte alta (Hz).
        - fm (float): frecuencia de muestreo (Hz).
        - orden (int): orden del filtro.

    Retorna:
        - canalFiltrado: canal filtrado en formato (numpy.ndarray)
        
        Info del filtro:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
    '''
    
    nyquist = 0.5 * fm
    low = lfrec / nyquist
    high = hfrec / nyquist
    b, a = butter(order, [low, high], btype='band') #obtengo los parámetros del filtro
    
    return filtfilt(b, a, canal) #aplicamos filtro con los parámetros obtenidos

def notch(canal, removeFrec, fm = 360.0):
    
    b, a = iirnotch(removeFrec, 20., fm)
    
    return filtfilt(b, a, canal)

def plotECG(ecg, fm = 360.,
            iniTime = 0., finalTime = 0.,
            markers = [],
            label = "",
            title = "ECG", savePlots = False,
            folder = "figs"):
    
    if finalTime == 0:
        finalTime = ecg.shape[0]/fm
    
    T = 1.0/fm #período de la señal
    totalLenght = ecg.shape[0]
    t = np.arange(int(iniTime*fm), int(finalTime*fm))*T
    
    plt.plot(t, ecg,'-o', c= "#403e7d", markevery = markers, mfc='#f37263', label = label)
    plt.xlabel("Tiempo [seg]")   # Inserta el título del eje X
    plt.ylabel("Voltaje [mV]")   # Inserta el título del eje Y
    plt.title(title)
    plt.legend(loc='lower center')
    
    if savePlots:
        pathACtual = os.getcwd()
        newPath = os.path.join(pathACtual, folder)
        os.chdir(newPath)
        plt.savefig(f'{title}.png', dpi = 600)
        os.chdir(pathACtual)
    
    plt.show()
    
def plotSeveralECG(ecg, labels = [], fm = 360., rows = 1, columns = 1,
                   iniTime = 0., finalTime = 0.,markers = [],
                   title = "ECG", savePlots = False,
                   folder = "figs"):
    
    chunksNummber = ecg.shape[0]
    ecgLenght = ecg[0].shape[0]
    
    T = 1.0/fm #período de la señal
    t = np.arange(0, ecgLenght)*T
        
    if columns == 1:
        columns = 2
    
    #genero la grilla para graficar
    fig, axes = plt.subplots(rows, columns, figsize=(40, 20),
                             gridspec_kw = dict(hspace=0.3, wspace=0.2))
    
    fig.suptitle(title, fontsize=28)

    if rows >= 2:
        axes = axes.reshape(-1)

    indexLabel = 0
    for chunk in range(chunksNummber):
        axes[int(labels[indexLabel])].plot(t, ecg[chunk], label=labels[indexLabel])
        axes[int(labels[indexLabel])].set_xlabel('Tiempo [seg]', fontsize=16) 
        axes[int(labels[indexLabel])].set_ylabel('Amplitud [uV]', fontsize=16)
        # axes[chunk].set_title(f'Sujeto {sujeto} - Blanco {blanco} - Canal {canal + 1}', fontsize=22)
        axes[int(labels[indexLabel])].yaxis.grid(True)
        indexLabel += 1
        
    if savePlots:
        pathACtual = os.getcwd()
        newPath = os.path.join(pathACtual, folder)
        os.chdir(newPath)
        plt.savefig(f'{title}.png', dpi = 200)
        os.chdir(pathACtual)
    
    plt.show()
    
    
def peakDetection(ecg, window, threshold, fm = 360., startingIndex = 0):
    
    ecg = ecg - np.mean(ecg)
    peaks = []
    indexesPeak = []
    initialIndex = 0
    finalIndex = int(window*fm)
    maxIndex = ecg.shape[0]
    
    while True:
        
        if finalIndex < maxIndex:
            
            # valuePeak.append([value for value in shortECG if value >= threshold])
            maxValue = max(ecg[initialIndex:finalIndex])
            if maxValue >= threshold:
                
                peaks.append(max(ecg[initialIndex:finalIndex]))    
                indexesPeak.append(np.argmax(ecg[initialIndex:finalIndex])+initialIndex)
                
            initialIndex += int(window*fm)
            finalIndex += int(window*fm)
            
        else:
            peaks = np.array(peaks)
            indexesPeak = np.array(indexesPeak)# + startingIndex
            avg = 0
            for i in range(len(indexesPeak)-1):
                avg += indexesPeak[i+1] - indexesPeak[i]
            
            avgFreqRate = avg/(indexesPeak.shape[0]-1)/fm*60
            break
        
    return peaks, indexesPeak, avgFreqRate

def splitECG(ecg, peakIndexes, splitWindow,  fm = 360.):
    
    splittedecg = []
    abSamples = int(splitWindow*fm/2) #after and before samples
    
    splittedecg = [ecg[index-abSamples : index+abSamples] for index in peakIndexes]
    
    return np.asarray(splittedecg)