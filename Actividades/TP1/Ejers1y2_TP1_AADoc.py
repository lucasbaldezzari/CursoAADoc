# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 11:39:44 2021

@author: Lucas B.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time

def plotScatter(values, centers, title = "Grafico Dispersión", labels = [],
                savePlots = False, folder = "figs"):
    
    """
    Args:
        - values: The dataset with form [NxM]
        - centers: The centers for my data
    """
    
    plt.grid(True, 'major', linestyle='-', linewidth =.9, zorder = 1.0)

    plt.scatter(values[: , 0], values[: , 1], c = labels,
             zorder = 2.0, alpha=0.65, cmap='tab10', s = 50)
    
    if centers.any(): #if centers is not empty
        plt.scatter(centers[: , 0], centers[: , 1],
                 zorder = 3.0, alpha=0.9, s = 50, marker = "*",
                 label = "Centroides", c="k")
    
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    # plt.legend(loc='best')

    if savePlots:
        pathACtual = os.getcwd()
        newPath = os.path.join(pathACtual, folder)
        os.chdir(newPath)
        plt.savefig(f'{title}.png', dpi = 600)
        os.chdir(pathACtual)
        
    plt.show()
    
def distance(X, centers):
    '''
    args:
        - X: Vector con datos para compararlo contra el dataset
        - dataset: set de datos de la forma [N x M]
    retorna:
        - La distancia de cada punto X respecto de cada centro en centers
        de la forma [1 x centers]
    '''
    
    numCenters = centers.shape[0]
    copias = np.tile(X, (numCenters,1)) #copio cada punto en "num_data" veces para luego compararlo con cada centro
    distance = np.sum((copias - centers)**2, axis = 1)**1/2
    
    return distance


def findingClusters(dataset, kClusters = 2, centers = None):
    """
    Nota: Agrego el argumento "centers" con la idea de que se puedan definir centros
    manualmente.
    Note: The "centers" argument is added for manual intialization
    """
    labels = np.zeros(dataset.shape[0])
    
    costSum = -1
    oldCostSum = 0
    
    #step 1
    #if we have not the centers, we choose them randomly
    if not centers:
        rand = np.random.RandomState()
        index = rand.permutation(dataset.shape[0])[:kClusters] #select kClusters centers randomly
        centers = dataset[index]
        centers2 = dataset[index]
        
    while oldCostSum != costSum:
        
        oldCostSum = costSum
        costSum = 0

        # step 2.1: assign data to closest center
        distances = np.apply_along_axis(distance, 1, dataset, centers)
        labels = distances.argmin(axis = 1)

        # Step 2.2: Update the centers and cost function
        for ic in range(kClusters):
            cluster = dataset[labels == ic]

            cost2 = np.apply_along_axis(distance, 1, cluster, cluster).sum(axis=0)
            centers[ic] = cluster[cost2.argmin()] #select the minimun
            costSum += cost2.min()
       
    return centers, labels

def main():
    
    initialFolder = os.getcwd() #directorio donde estamos actualmente. Debe contener el directorio dataset
    path = os.path.join(initialFolder,"data2cluster")
    files = os.listdir(path) #leo los archivos dentro del path.
    
    os.chdir(initialFolder) #vuelvo al directorio de inicio
    
    centers = np.array([])
    #grafico los datos originales
    for file in files:
        rawData = pd.read_csv(f"{path}/{file}") #cargo datos
        
        heads = rawData.keys() #nombre de columnas
        
        kGroups = rawData[heads[2]].nunique() #contabilizo la cantidad de grupos diferentes
        gruposOriginales = [f"Grupo {i}" for i in range(kGroups)]
        
        # mediaGrupos = rawData.groupby([heads[2]]).mean()
        
        values = np.asarray(rawData[[heads[0],heads[1]]])
        originalLabels = np.asarray(rawData[heads[2]])
        title = f"Grupos originales - {file}"
        
        # plotScatter(values, centers, title = title, savePlots = False, folder = "figs",
        #             labels = originalLabels)
    
    ks = [2]
    
    tiemposKs = dict()
    
    for k in ks:
        # tiemposKs[files[indexFile]] = dict()
        initialTime = time.time()
        tiempos = []
        for file in files:
            rawData = pd.read_csv(f"{path}/{file}") #cargo datos
            
            heads = rawData.keys() #nombre de columnas
            
            kGroups = rawData[heads[2]].nunique() #contabilizo la cantidad de grupos diferentes
            gruposOriginales = [f"Grupo {i}" for i in range(kGroups)]
            
                  
            values = np.asarray(rawData[[heads[0],heads[1]]])
            originalLabels = np.asarray(rawData[heads[2]])
        
            centers, labels = findingClusters(dataset = values, kClusters = k)
            
            title = f"Agrupamiento de findingClusters() con k = {k} - {file}"
            
            plotScatter(values, centers = centers,
                        title = title, savePlots = False, folder = "figs",
                        labels = labels)
            
            stopTime = time.time()#/1000
            tiempoTranscurrido = stopTime - initialTime
            # print(f"Tiempo transcurrido de implementación para {file}: {tiempoTranscurrido}")
            
            tiempos.append(tiempoTranscurrido)
            
            initialTime = time.time()
            
        tiemposKs[f"{k}"] = tiempos
            
    tiempos = np.zeros((len(ks),len(files))) 
    # tiempos = []
    i = 0
    for key in tiemposKs.keys():
        tiempos[i] = tiemposKs[key]
        i += 1
    
    title = "Tiempos totales para diferentes k"
    
    etiquetas = ["anisotropico","cumulos","variado","lunas","circulos"]
    
    for index in range(tiempos.shape[0]):
        plt.plot(etiquetas,tiempos[index,:], label = f"k = {ks[index]}",
                 marker='o', linestyle=':', alpha = 0.9)
        plt.xlabel("Archivos")   # Inserta el título del eje X
        plt.ylabel("Tiempo transcurrido [segundos]")   # Inserta el título del eje Y
        plt.title(title)
        plt.legend(loc = "upper right")
        
    savePlots = True
        
    if savePlots:
        pathACtual = os.getcwd()
        newPath = os.path.join(pathACtual, "figs")
        os.chdir(newPath)
        plt.savefig(f'{title}.png', dpi = 400)
        os.chdir(pathACtual)
        
    plt.show()
  

if __name__ == "__main__":
    main()
    