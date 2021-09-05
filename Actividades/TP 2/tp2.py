# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 10:35:18 2021

@author: Lucas Baldezzari
"""

"""
Preparación del dataset
Descargamos los datos del repositorio público y lo cargamos en la notebook

IMPORTANTE: Los comandos de la siguiente celda funcionan si se utiliza desde Colab.
Si usan la notebook de forma local puede no funcionar dependiendo de la configuración
de base. Se pueden llamar estos comandos desde la consola o directamente descargar el
archivo del link y guardarlo en la misma carpeta donde se encuentra la notebook
"""

#Debemos descargar el set de datos a través de 
#wget -q --show-progress https://sourceforge.net/projects/sourcesinc/files/mirdata/features/ath.zip/download

import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
import os



data = pd.read_csv("download/ath.csv")
print(data.head())

pd.set_option('display.float_format', lambda x: '%.5f' % x)

print(data.describe())

# data["CLASS"].value_counts()

casos = 1000

# Definimos una probabilidad de muestreo de forma tal que sea mucho mas probable
# muestrear un caso positivo que uno negativo
ratio = sum(data["CLASS"])/len(data)
data["weight"] = data["CLASS"].apply(lambda x: ratio if x==0 else 1)

data = data.sample(n = casos, weights=data["weight"])

data["CLASS"].value_counts()

"""
Comentario: Hemos cambiado la forma de mostrar mostrar los datos a solo 2 decimales
Por otro lado, el ratio calculado da aproximadamente 0.00022424, por lo tanto,
al aplicar 'display.float_format' estamos perdiendo información ya que 
para los casos x==0 veremos todos 0.00 en vez de ver el valor de ratio realmente.
De todos modos, a fines prácticos una probabilidad de 0.00224% podemos considerarla como cero,
pero no me queda claro el por qué aplicar lambda x: ratio if x==0 else 1 si ya tenemos valores cero en CLASS.
"""

X = data.drop(columns=["sequence_names", "CLASS", "weight"]).values
y = data["CLASS"].values
print(X.shape, y.shape)

# plt.figure(figsize=(6,3))
# plt.plot(X[:, 0], X[:, 76], "o")


"""
Haremos una validación cruzada de 5 particiones, las hacemos acá para que todas
las pruebas se corran con los mismo datos. Como los datos estan muy desbalanceados,
haremos una partición estratificada: cada partición tendrá la misma proporción de
clases que los datos originales
"""

from sklearn.model_selection import StratifiedKFold

generador_particiones = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)

particiones = list(generador_particiones.split(X, y))

# Ahora "particiones" es una lista de la forma
#[(train_ind1, test_ind1), (train_ind2, test_ind2), ...]


from sklearn.model_selection import train_test_split
# definimos la métrica para evaluar los modelos
from sklearn.metrics import f1_score

# Cargamos los datos de la primer partición de entrenamiento, para separarlos 
# en entrenamiento y optimizacion
train_ind, _ = particiones[0] # notar que "test_ind" es "_" porque no lo vamos a usar

# Ahora tendremos las particiones del esquema de arriba 
Xtrain, Xoptim, ytrain, yoptim = train_test_split(X[train_ind, :], y[train_ind],
                                                  test_size=.2, stratify=y[train_ind])

print("Datos de entrenamiento", Xtrain.shape,
      "| datos de optimización:", Xoptim.shape)
print(f"Entrenamiento: {len(ytrain)} ({np.sum(ytrain==1)} positivos)")
print(f"Optimización: {len(yoptim)} ({np.sum(yoptim==1)} positivos)")


"""Probando diferentes clasificadores"""
# TODO: Importar la función del clasificador a usar
from sklearn.svm import SVC #https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
from sklearn.metrics import accuracy_score

from utils import plot_decision_function

"""
Intuitively, the gamma parameter defines how far the influence of a single training example 
reaches, with low values meaning ‘far’ and high values meaning ‘close’. 
The gamma parameters can be seen as the inverse of the radius of influence of 
samples selected by the model as support vectors.

The C parameter trades off correct classification of training examples against 
maximization of the decision function’s margin. For larger values of C, a smaller 
margin will be accepted if the decision function is better at classifying all 
training points correctly. A lower C will encourage a larger margin, therefore 
a simpler decision function, at the cost of training accuracy. In other words C 
behaves as a regularization parameter in the SVM.

https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
"""

hiperParams = {"kernels": ["linear", "rbf"],
               "gammaValues": [1e-2, 1e-1, 1, 1e+1, 1e+2, "scale", "auto"],
               "CValues": [8e-1,9e-1, 1, 1e2, 1e3]}

clasificadoresSVM = {"linear": list(),
                  "rbf": list()
    }

rbfResults = np.zeros((len(hiperParams["gammaValues"]), len(hiperParams["CValues"])))
linearResults = list()

for i, kernel in enumerate(hiperParams["kernels"]):
    
    if kernel != "linear":
        for j, gamma in enumerate(hiperParams["gammaValues"]):
            
            for k, C in enumerate(hiperParams["CValues"]):
                #Instanciamos el modelo para los hipermarametros
                model = SVC(C = C, kernel = kernel, gamma = gamma)
                
                #entreno el modelo
                model.fit(Xtrain, ytrain)
                
                #predecimos con los datos en Xoptim
                pred = model.predict(Xoptim)
                accu = f1_score(yoptim, pred)
                
                rbfResults[j,k] = accu
                
                clasificadoresSVM[kernel].append((C, gamma, model, accu))
    else:
        for k, C in enumerate(hiperParams["CValues"]):
            
            #Instanciamos el modelo para los hipermarametros
            model = SVC(C = C, kernel = kernel)
            #entreno el modelo
            model.fit(Xtrain, ytrain)
            pred = model.predict(Xoptim)
            accu = f1_score(yoptim, pred)
            linearResults.append(accu)
            #predecimos con los datos en Xoptim
            
            clasificadoresSVM[kernel].append((C, model, accu))
            
plt.figure(figsize=(15,10))
plt.imshow(rbfResults)
plt.xlabel("Valor de C")
plt.xticks(np.arange(len(hiperParams["CValues"])), hiperParams["CValues"])
plt.ylabel("Valor de Gamma")
plt.yticks(np.arange(len(hiperParams["gammaValues"])), hiperParams["gammaValues"])
plt.colorbar();

for i in range(rbfResults.shape[0]):
    for j in range(rbfResults.shape[1]):
        plt.text(j, i, "{:.2f}".format(rbfResults[i, j]), va='center', ha='center')
plt.show()

num = np.arange(1,len(linearResults)+1)
plt.plot([str(C) for C in hiperParams["CValues"]], np.asarray(linearResults)*100)
plt.title("Accuracy para predicciones usando kernel 'linear'")
plt.xlabel("Valor de C") 
plt.ylabel("Accuracy (%)")
plt.show()

"""
Probando un Knn
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

- Hiperparametros del modelo Knn

print(modelo.get_params())

{'algorithm': 'auto', 'leaf_size': 30, 'metric': 'minkowski', 'metric_params': None,
 'n_jobs': None, 'n_neighbors': 5, 'p': 2, 'weights': 'uniform'}
"""

from sklearn.neighbors import KNeighborsClassifier

#Declaro algunos hiperparametros para utilizar posteriormente
hiperParams = {"algorithms": ["ball_tree", "kd_tree"],
               "metrics": ["euclidean","chebyshev","minkowski","manhattan"],
               "weights": ["uniform","distance"]}

# hiperParams = {"algorithms": ["ball_tree"],
#                "metrics": ["euclidean"],
#                "weights": ["uniform"]}

clasificadoresKnn = {"ball_tree": list(),
                     "kd_tree": list()}

# neighborsNum = {"5": dict(), "6": dict(), "7": dict(), "8": dict(),
                # "9": dict(), "10": dict(), "11": dict(), "12": dict()}

ball_treeResults = np.zeros((len(hiperParams["metrics"]), len(hiperParams["weights"])))
kd_treeResults = np.zeros((len(hiperParams["metrics"]), len(hiperParams["weights"])))


# accuForNeighNum = {"5": dict(), "6": dict(), "7": dict(), "8": dict(),
                   # "9": dict(), "10": dict(), "11": dict(), "12": dict()}

neighborsNum = list() #Utilizaré esta lista para almacenar los clasificadoresKnn
accuForNeighNum = list() #Para almacenar los accuracies

for n_neighbors in np.arange(5,11):#neighborsNum.keys():
    
    for i, algorithm in enumerate(hiperParams["algorithms"]):
        
        for j, metric in enumerate(hiperParams["metrics"]):
            
            for k, weight in enumerate(hiperParams["weights"]):
                #Instanciamos el modelo para los hipermarametros
                model = KNeighborsClassifier(n_neighbors = int(n_neighbors),
                                             weights = weight,
                                             algorithm = algorithm,
                                             metric = metric)
                #entreno el modelo
                model.fit(Xtrain, ytrain)
                
                #predecimos con los datos en Xoptim
                pred = model.predict(Xoptim)
                accu = f1_score(yoptim, pred)
                
                if algorithm == "ball_tree":
                    ball_treeResults[j,k] = accu
                    
                if algorithm == "kd_tree":
                    kd_treeResults[j,k] = accu
                
                clasificadoresKnn[algorithm].append((weight, metric,model,accu))

    # neighborsNum[n_neighbors] = clasificadoresKnn
    # accuForNeighNum[n_neighbors] = {"accu_ball_tree": ball_treeResults,
    #                                 "accu_kd_tree": kd_treeResults}
    
    neighborsNum.append(clasificadoresKnn)
    accuForNeighNum.append({"accu_ball_tree": ball_treeResults,
                            "accu_kd_tree": kd_treeResults})
    
    ball_treeResults = np.zeros((len(hiperParams["metrics"]), len(hiperParams["weights"])))
    kd_treeResults = np.zeros((len(hiperParams["metrics"]), len(hiperParams["weights"])))
    clasificadoresKnn = {"ball_tree": list(),
                     "kd_tree": list()}

#Grafico los accuracies para un modelo
title = "Accuracies - knn = 5 - accu_ball_tree"

plt.figure(figsize=(15,10))
plt.title(title)
plt.imshow(accuForNeighNum[0]["accu_ball_tree"], cmap = "Greens_r")
plt.xlabel("Weights")
plt.xticks(np.arange(len(hiperParams["weights"])), hiperParams["weights"])
plt.ylabel("Metric")
plt.yticks(np.arange(len(hiperParams["metrics"])), hiperParams["metrics"])
plt.colorbar();

for i in range(accuForNeighNum[0]["accu_ball_tree"].shape[0]):
    for j in range(accuForNeighNum[0]["accu_ball_tree"].shape[1]):
        plt.text(j, i, "{:.2f}".format(accuForNeighNum[0]["accu_ball_tree"][i, j]),
                 va='center', ha='center')
        
save = True

if save:
    pathACtual = os.getcwd()
    newPath = os.path.join(pathACtual, "figs")
    os.chdir(newPath)
    plt.savefig(title, dpi = 200)
    os.chdir(pathACtual)
    
plt.show()

# Grafico accuraccies para todos los modelos entrenados
fig, axes = plt.subplots(2, 6, figsize=(20, 15),
                         gridspec_kw = dict(hspace=0.1, wspace=0.2))
title = "Accuracies totales - accu_ball_tree (en verde) - accu_kd_tree (en azul)"

fig.suptitle(title, fontsize=28)

axes = axes.reshape(-1)
    
for n_neighbors in np.arange(0,6):
    axes[n_neighbors].set_xlabel('Weights') 
    axes[n_neighbors+6].set_xlabel('Weights')
    if n_neighbors == 0:
        axes[n_neighbors].set_ylabel('Metrics')
        axes[n_neighbors+6].set_ylabel('Metrics')
    
    axes[n_neighbors].set_title(f"Accuracies\n knn = {n_neighbors+5}\n accu_ball_tree")
    axes[n_neighbors+6].set_title(f"Accuracies\n knn = {n_neighbors+5}\n accu_kd_tree")
    
    axes[n_neighbors].imshow(accuForNeighNum[0]["accu_ball_tree"], cmap = "Greens_r")
    axes[n_neighbors+6].imshow(accuForNeighNum[0]["accu_kd_tree"], cmap = "Blues_r")

    axes[n_neighbors].set_xticks(np.arange(len(hiperParams["weights"])))
    axes[n_neighbors+6].set_xticks(np.arange(len(hiperParams["weights"])))
    
    axes[n_neighbors].set_xticklabels(hiperParams["weights"])
    axes[n_neighbors+6].set_xticklabels(hiperParams["weights"])
    
    if n_neighbors == 0:
        axes[n_neighbors].set_yticks(np.arange(len(hiperParams["metrics"])))
        axes[n_neighbors].set_yticklabels(hiperParams["metrics"])
        axes[n_neighbors+6].set_yticks(np.arange(len(hiperParams["metrics"])))
        axes[n_neighbors+6].set_yticklabels(hiperParams["metrics"])
    else:
        axes[n_neighbors].yaxis.set_visible(False)
        axes[n_neighbors+6].yaxis.set_visible(False)
        
    for i in range(accuForNeighNum[n_neighbors]["accu_ball_tree"].shape[0]):
        for j in range(accuForNeighNum[n_neighbors]["accu_ball_tree"].shape[1]):
            axes[n_neighbors].text(j, i, "{:.2f}".format(accuForNeighNum[n_neighbors]["accu_ball_tree"][i, j]),
                     va='center', ha='center') 
            
    for i in range(accuForNeighNum[n_neighbors]["accu_kd_tree"].shape[0]):
        for j in range(accuForNeighNum[n_neighbors]["accu_kd_tree"].shape[1]):
            axes[n_neighbors+6].text(j, i, "{:.2f}".format(accuForNeighNum[n_neighbors]["accu_kd_tree"][i, j]),
                     va='center', ha='center') 

if save:
    pathACtual = os.getcwd()
    newPath = os.path.join(pathACtual, "figs")
    os.chdir(newPath)
    plt.savefig(title, dpi = 200)
    os.chdir(pathACtual)

plt.show()
    
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
# for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
#     xs = X[:,1]
#     ys = X[:,31]
#     zs = X[:,41]
#     ax.scatter(xs, ys, zs, marker=m)

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()

"""
Comparando clasificadores
"""
# TQDM muestra una barra de avance a medida que pasan las iteraciones 
from tqdm import tqdm

# Vamos a analizar varias métricas
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

#Selecciono dos clasificadores SVM
modeloSVM1 = clasificadoresSVM["linear"][3][1] #modelo 3 con [Model = SVC(C=100.0, kernel='linear'), accu = 0.811]

gamma = 0.8
C = "scale"

for values in clasificadoresSVM["rbf"]:
    if values[0] == gamma and values[1] == C:
        modeloSVM2 = values[2] #modelo 2 es un SVM con kernel = rbf

#Selecciono dos clasificadores KNN
#Modelo KNN 1
n_neighbors = 9
algorithms = "ball_tree"
metric = "euclidean"
weight = "distance"

neighborsNum[n_neighbors-5][algorithms]

for values in neighborsNum[n_neighbors-5][algorithms]:
    if values[0] == weight and values[1] == metric:
        modeloKNN1 = values[2]
    
#Modelo KNN 2
n_neighbors = 9
algorithms = "kd_tree"
metric = "minkowski"
weight = "distance"

neighborsNum[n_neighbors-5][algorithms]

for values in neighborsNum[n_neighbors-5][algorithms]:
    if values[0] == weight and values[1] == metric:
        modeloKNN2 = values[2]

#Lista modelos
# modelos = [modeloSVM1, modeloSVM2, modeloKNN1, modeloKNN2]

modelos = {"SVM1": modeloSVM1,
           "SVM2": modeloSVM2,
           "KNN1": modeloKNN1,
           "KNN2": modeloKNN2}

generalResults = []

for train_ind, test_ind in tqdm(particiones): 
    for modelo in modelos: # TODO completar con todos los modelos a usar
            
        # if nombre_modelo == "modelo_1":
        #     # TODO: instanciar el modelo con los hiperparámetros que dieron mejor en el análisis previo
        #     modelo = # ...
        # if nombre_modelo == "modelo_2":
        #     # TODO: instanciar el modelo con los hiperparámetros que dieron mejor en el análisis previo
        #     modelo = # ...

        # # notar que ahora X[train_ind, :] contiene a Xtrain y Xoptim del punto 
        # # anterior
        modelos[modelo].fit(X[train_ind, :], y[train_ind])

        # La siguietne predicción se hace sobre datos que no fueron vistos antes
        pred = modelos[modelo].predict(X[test_ind, :])

        generalResults.append([modelo,
                               accuracy_score(y[test_ind], pred),
                               f1_score(y[test_ind], pred),
                               recall_score(y[test_ind], pred),
                               precision_score(y[test_ind], pred)])

generalResults = pd.DataFrame(generalResults, columns=["modelo", "acc", "f1", "pre", "rec"])
generalResults

import seaborn

fig, ax = plt.subplots(1, 4, figsize=(25, 10),
                       gridspec_kw = dict(hspace=0.1, wspace=0.2))

for k, metric in enumerate(["acc", "pre", "rec", "f1"]):
    seaborn.boxplot(data=generalResults, y=metric, x="modelo", ax=ax[k])
    ax[k].set_ylim([0, 1])
plt.tight_layout()
