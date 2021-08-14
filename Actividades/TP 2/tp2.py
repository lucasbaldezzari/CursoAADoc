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

print("Datos de entrenamiento", Xtrain.shape, "| datos de optimización:", Xoptim.shape)
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
    "CValues": [8e-1,9e-1, 1, 1e2, 1e3]
    }

clasificadores = {"linear": list(),
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
                
                clasificadores[kernel].append((C, gamma, model, accu))
    else:
        for k, C in enumerate(hiperParams["CValues"]):
            
            #Instanciamos el modelo para los hipermarametros
            model = SVC(C = C, kernel = kernel, gamma = gamma)
            #entreno el modelo
            model.fit(Xtrain, ytrain)
            pred = model.predict(Xoptim)
            accu = f1_score(yoptim, pred)
            linearResults.append(accu)
            #predecimos con los datos en Xoptim
            
            clasificadores[kernel].append((C, model, accu))
            
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