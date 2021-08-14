import numpy as np
from matplotlib import pyplot as plt 

def plot_decision_function(classifier, axis, title, X, y):
    """Grafica la distribución de puntajes por clase y vectores de soporte
    
    Parameters
    ----------
    classifier : SVC
        Clasificador entrenado.
    axis : Axis
        Ejes a graficar.
    title : str
        Título del gráfico
    X : numpy.ndarray
        Características
    y : numpy.ndarray
        Etiquetas
    """

    x1 = np.min(X[:, 0])
    x2 = np.max(X[:, 0])
    y1 = np.min(X[:, 1])
    y2 = np.max(X[:, 1])
    xx, yy = np.meshgrid(np.linspace(x1, x2, 200), np.linspace(y1, y2, 200))
    print(xx.shape)

    Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axis.contourf(xx, yy, Z, alpha=0.75, levels=4)
    axis.scatter(X[:, 0], X[:, 1], c=y, alpha=0.9,
                 cmap=plt.cm.bone, edgecolors='black')

    # Vectores de soporte
    axis.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
    
    axis.axis('off')
    axis.set_title(title)