# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 09:51:48 2021

@author: Lucas

# TP 3

Autor: Baldezzari Lucas

Nota: Parte del código implementado fue otorgado por los docentes del curso.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sea

from copy import deepcopy

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from pandas_profiling import ProfileReport

#### Carga de datos

trn_filename = 'DERMATOLOGY_trn.csv'
tst_filename = 'DERMATOLOGY_tst.csv'


#---------------------
# Carga de datos
#---------------------
data_trn = pd.read_csv(trn_filename, index_col=False, header=0)
data_tst = pd.read_csv(tst_filename, index_col=False, header=0)

tiposData_trn = data_trn.dtypes
tiposdata_tst = data_tst.dtypes

desc_data_trn = data_trn.describe()
desc_data_tst = data_tst.describe()

sea.pairplot(data_trn.loc[:,["Polygonal_papules", "Hyperkeratosis","Vacuolisation",
                             "Saw-tooth_appearance","Band-like_infiltrate","Focal_hypergranulosis"]])


#---------------------
# Analizando datos
#---------------------

# profile_data_trn = ProfileReport(data_trn, title = "Reporte datos de entrenamiento")
# profile_data_trn.to_file(output_file="<reporteDatosEntrenamiento>.html")

#-----------------------------------
# Separamos patrones de etiquetas
#-----------------------------------
X_TRN = data_trn.iloc[:,:-1]  # Features - patrones training + validation
Y_TRN = data_trn.iloc[:,-1]  # Clases - patrones training + validation

X_tst = data_tst.iloc[:,:-1]  # Features - patrones test
y_tst = data_tst.iloc[:,-1]  # Clases - patrones test

### Exploración de datos

#Chequeo variables faltantes en mis datos de entrenamiento
faltantes  = [(faltante,X_TRN[faltante].isna().sum()) for faltante in X_TRN if X_TRN[faltante].isnull().sum()]
print(faltantes)

#¿Datos balanceados?
Y_TRN.value_counts()

##### Separo datos para entrenamiento y validación
#80% de los patrones para entrenamiento y 20% para validación.

X_trn, X_val, y_trn, y_val = train_test_split(X_TRN, Y_TRN, test_size=.2)

# faltantes  = [(faltante,X_trn[faltante].isna().sum()) for faltante in X_trn if X_trn[faltante].isnull().sum()]
# print(faltantes)

# from sklearn.impute import SimpleImputer

# imputador = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputador = imputador.fit(X_trn)
# X_trn = imputador.transform(X_trn)


#Relleno valores faltantes con la media de cada feature
for faltante in faltantes:
    X_trn[faltante[0]].fillna(int(X_trn[faltante[0]].mean()), inplace=True)
    X_val[faltante[0]].fillna(int(X_val[faltante[0]].mean()), inplace=True)

#Chequeo si aún tengo datos faltantes
# faltantes  = [(faltante,X_trn[faltante].isna().sum()) for faltante in X_TRN if X_TRN[faltante].isnull().sum()]
# print(faltantes)

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# PCA
pca = PCA(n_components = 2)
pca.fit(X_trn)

#proyecto los datos con el PCA
Xtrn_pca = pca.transform(X_trn)
X_val_pca = pca.transform(X_val)
X_tst_pca = pca.transform(X_tst)

#grafica
plt.figure(figsize=(8,6))
plt.scatter(Xtrn_pca[:,0], Xtrn_pca[:,1])
plt.title("Datos de entrenamiento proyectados mediante PCA")
plt.xlabel('Primera componente principal')
plt.ylabel('Segunda componente principal')

#varianza acumulada
print(pca.explained_variance_ratio_)
plt.figure(figsize=(8,6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title("Varianza acumulada de PCA")
plt.xlabel('Componentes')
plt.ylabel('Varianza acumulada')

#LDA
lda = LinearDiscriminantAnalysis(n_components = 2)
lda.fit(X_trn, y_trn)

#proyecto los datos con el PCA
Xtrn_lda = lda.transform(X_trn)
X_val_lda = lda.transform(X_val)
X_tst_lda = lda.transform(X_tst)

#Grafico de dispersión
plt.figure(figsize=(8,6))
plt.scatter(Xtrn_lda[:,0], Xtrn_lda[:,1])
plt.title("Datos de entrenamiento proyectados mediante LDA")
plt.xlabel('Primera componente principal')
plt.ylabel('Segunda componente principal')

print(lda.explained_variance_ratio_)
plt.figure(figsize=(8,6))
plt.plot(np.cumsum(lda.explained_variance_ratio_))
plt.title("Varianza acumulada de LDA")
plt.xlabel('Componentes')
plt.ylabel('Varianza acumulada')     

# METRICAS

#Estrcutura

METRICAS = {'modelo': {'trn': {'Pr': None, 'Rc': None, 'Acc': None, 'F1':None},
                       'val': {'Pr': None, 'Rc': None, 'Acc': None, 'F1':None},
                       'tst': {'Pr': None, 'Rc': None, 'Acc': None, 'F1':None}},
            'modelo+pca': {'trn': {'Pr': None, 'Rc': None, 'Acc': None, 'F1':None},
                           'val': {'Pr': None, 'Rc': None, 'Acc': None, 'F1':None},
                           'tst': {'Pr': None, 'Rc': None, 'Acc': None, 'F1':None}},
            'modelo+lda': {'trn': {'Pr': None, 'Rc': None, 'Acc': None, 'F1':None},
                           'val': {'Pr': None, 'Rc': None, 'Acc': None, 'F1':None},
                           'tst': {'Pr': None, 'Rc': None, 'Acc': None, 'F1':None}}}

# SVM
from sklearn.svm import SVC #https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
from sklearn.metrics import accuracy_score

modelo = SVC(C = 9e-1, kernel = "rbf", gamma = 1e-1)

model_full = deepcopy(modelo)
model_pca = deepcopy(modelo)
model_lda = deepcopy(modelo)

#Entrenamiento del modelo con todas las features

model_full.fit(X_trn, y_trn)

#predicciones modelo completo sobre datos de entrenamiento
y_pred = model_full.predict(X_trn)

# Cálculo de métricas para modelo completo
precision, recall, f1,_ = precision_recall_fscore_support(y_trn, y_pred, average='macro')
accuracy = accuracy_score(y_trn, y_pred)

# Guardo métricas
METRICAS['modelo']['trn']['Pr'] = precision
METRICAS['modelo']['trn']['Rc'] = recall
METRICAS['modelo']['trn']['Acc'] = accuracy
METRICAS['modelo']['trn']['F1'] = f1

#predicciones modelo completo sobre datos de validación
y_pred = model_full.predict(X_val)

# Cálculo de métricas
precision, recall, f1,_ = precision_recall_fscore_support(y_val, y_pred, average='macro')
accuracy = accuracy_score(y_val, y_pred)

# Guardo métricas
METRICAS['modelo']['val']['Pr'] = precision
METRICAS['modelo']['val']['Rc'] = recall
METRICAS['modelo']['val']['Acc'] = accuracy
METRICAS['modelo']['val']['F1'] = f1

#predicciones modelo completo sobre datos de test
y_pred = model_full.predict(X_tst)

# Cálculo de métricas
precision, recall, f1,_ = precision_recall_fscore_support(y_tst, y_pred, average='macro')
accuracy = accuracy_score(y_tst, y_pred)

# Guardo métricas
METRICAS['modelo']['tst']['Pr'] = precision
METRICAS['modelo']['tst']['Rc'] = recall
METRICAS['modelo']['tst']['Acc'] = accuracy
METRICAS['modelo']['tst']['F1'] = f1

# PCA

model_pca.fit(Xtrn_pca, y_trn)

# VALIDATION con PCA

#predicciones del modelo usando PCA
y_pred = model_pca.predict(Xtrn_pca)

# Calculo las métricas
precision, recall, f1,_ = precision_recall_fscore_support(y_trn, y_pred, average='macro')
accuracy = accuracy_score(y_trn, y_pred)

# Guardo las métricas
METRICAS['modelo+pca']['trn']['Pr'] = precision
METRICAS['modelo+pca']['trn']['Rc'] = recall
METRICAS['modelo+pca']['trn']['Acc'] = accuracy
METRICAS['modelo+pca']['trn']['F1'] = f1

#predicciones modelo + PCA sobre datos de validación
y_pred = model_pca.predict(X_val_pca)

# Cálculo de métricas
precision, recall, f1,_ = precision_recall_fscore_support(y_val, y_pred, average='macro')
accuracy = accuracy_score(y_val, y_pred)

# Guardo métricas
METRICAS['modelo+pca']['val']['Pr'] = precision
METRICAS['modelo+pca']['val']['Rc'] = recall
METRICAS['modelo+pca']['val']['Acc'] = accuracy
METRICAS['modelo+pca']['val']['F1'] = f1

#predicciones modelo + PCA sobre datos de test
y_pred = model_pca.predict(X_tst_pca)

# Cálculo de métricas
precision, recall, f1,_ = precision_recall_fscore_support(y_tst, y_pred, average='macro')
accuracy = accuracy_score(y_tst, y_pred)

# Guardo métricas
METRICAS['modelo+pca']['tst']['Pr'] = precision
METRICAS['modelo+pca']['tst']['Rc'] = recall
METRICAS['modelo+pca']['tst']['Acc'] = accuracy
METRICAS['modelo+pca']['tst']['F1'] = f1

#LDA

model_pca.fit(Xtrn_lda, y_trn)

# VALIDATION con PCA

#predicciones del modelo usando PCA
y_pred = model_pca.predict(Xtrn_lda)

# Calculo las métricas
precision, recall, f1,_ = precision_recall_fscore_support(y_trn, y_pred, average='macro')
accuracy = accuracy_score(y_trn, y_pred)

# Guardo las métricas
METRICAS['modelo+lda']['trn']['Pr'] = precision
METRICAS['modelo+lda']['trn']['Rc'] = recall
METRICAS['modelo+lda']['trn']['Acc'] = accuracy
METRICAS['modelo+lda']['trn']['F1'] = f1

#predicciones modelo + LDA sobre datos de validación
y_pred = model_pca.predict(X_val_lda)

# Cálculo de métricas
precision, recall, f1,_ = precision_recall_fscore_support(y_val, y_pred, average='macro')
accuracy = accuracy_score(y_val, y_pred)

# Guardo métricas
METRICAS['modelo+lda']['val']['Pr'] = precision
METRICAS['modelo+lda']['val']['Rc'] = recall
METRICAS['modelo+lda']['val']['Acc'] = accuracy
METRICAS['modelo+lda']['val']['F1'] = f1

#predicciones modelo + PCA sobre datos de test
y_pred = model_pca.predict(X_tst_lda)

# Cálculo de métricas
precision, recall, f1,_ = precision_recall_fscore_support(y_tst, y_pred, average='macro')
accuracy = accuracy_score(y_tst, y_pred)

# Guardo métricas
METRICAS['modelo+lda']['tst']['Pr'] = precision
METRICAS['modelo+lda']['tst']['Rc'] = recall
METRICAS['modelo+lda']['tst']['Acc'] = accuracy
METRICAS['modelo+lda']['tst']['F1'] = f1


# Resultados comparativos
print('\n------------------------------------------')
print(':::                TRAIN               :::')
print('------------------------------------------')
print('Features    Precision  Recall   Acc     F1')

m1 = METRICAS['modelo']['trn']
print(f"Full        {m1['Pr']:.4}     {m1['Rc']:.4}   {m1['Acc']:.4}  {m1['F1']:.4}")
m2 = METRICAS['modelo+pca']['trn']
print(f"PCA         {m2['Pr']:.4}     {m2['Rc']:.4}   {m2['Acc']:.4}  {m2['F1']:.4}")
m3 = METRICAS['modelo+lda']['trn']
print(f"LDA         {m3['Pr']:.4}     {m3['Rc']:.4}   {m3['Acc']:.4}  {m3['F1']:.4}")

#----------------------

print('\n-----------------------------------------------')
print(':::               VALIDATION                :::')
print('-----------------------------------------------')
print('Features    Precision  Recall   Acc     F1')

measures = METRICAS['modelo']['val']
print(f"Full        {measures['Pr']:.4}     {measures['Rc']:.4}   {measures['Acc']:.4}  {measures['F1']:.4}")
measures = METRICAS['modelo+pca']['val']
print(f"PCA         {measures['Pr']:.4}     {measures['Rc']:.4}   {measures['Acc']:.4}  {measures['F1']:.4}")
measures = METRICAS['modelo+lda']['val']
print(f"LDA         {measures['Pr']:.4}     {measures['Rc']:.4}   {measures['Acc']:.4}  {measures['F1']:.4}")