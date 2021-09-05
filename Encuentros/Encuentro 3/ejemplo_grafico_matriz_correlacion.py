# SOURCE: https://stackoverflow.com/questions/34556180/how-can-i-plot-a-correlation-matrix-as-a-set-of-ellipses-similar-to-the-r-open

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import EllipseCollection

#=============================================================================
def plot_corr_ellipses(data, ax=None, **kwargs):

    M = np.array(data)
    if not M.ndim == 2:
        raise ValueError('data must be a 2D array')
    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw={'aspect':'equal'})
        ax.set_xlim(-0.5, M.shape[1] - 0.5)
        ax.set_ylim(-0.5, M.shape[0] - 0.5)

    # xy locations of each ellipse center
    xy = np.indices(M.shape)[::-1].reshape(2, -1).T

    # set the relative sizes of the major/minor axes according to the strength of
    # the positive/negative correlation
    w = np.ones_like(M).ravel()
    h = 1 - np.abs(M).ravel()
    a = 45 * np.sign(M).ravel()

    ec = EllipseCollection(widths=w,
                           heights=h,
                           angles=a,
                           units='x',
                           offsets=xy,
                           transOffset=ax.transData,
                           array=M.ravel(),
                           **kwargs)
    
    ax.add_collection(ec)

    # if data is a DataFrame, use the row/column names as tick labels
    if isinstance(data, pd.DataFrame):
        ax.set_xticks(np.arange(M.shape[1]))
        ax.set_xticklabels(data.columns, rotation=90)
        ax.set_yticks(np.arange(M.shape[0]))
        ax.set_yticklabels(data.index)

    return ec
#=============================================================================



#############
#  EJEMPLO  #
#############

#==================================
colnames= ['V1','V2','V3','V4','V5','V6','V7']
values = np.array([[0,0.00003,85.060,25.933,0,0,28.639],
                   [4.828,4.80720,90.918,29.17,27.449,17.5,141.850],
                   [7.7953,50.263,77.567,74.471,26.994,24.876,169.600],
                   [37.9610,17.996,94.188,45.892,46.734,23.461,186.990],
                   [57.929,25.723,106.27,36.699,58.988,58.118,211.78]])

DATA = {colnames[i]:values[:,i] for i in range(len(colnames))}

df = pd.DataFrame.from_dict(DATA)
#==================================


data = df.corr()
fig, ax = plt.subplots(1, 1)
plt.grid(True)
m = plot_corr_ellipses(data,
                       ax=ax,
                       cmap='jet',
                       clim=[-1, 1])

cb = fig.colorbar(m)
cb.set_label('Correlation coefficient')
ax.margins(0.1)



# Negative correlations can be plotted as ellipses with the opposite orientation:

fig2, ax2 = plt.subplots(1, 1)
data2 = np.linspace(-1, 1, 9).reshape(3, 3)
m2 = plot_corr_ellipses(data2,
                        ax=ax2,
                        cmap='seismic',
                        clim=[-1, 1])

cb2 = fig2.colorbar(m2)
ax2.margins(0.3)

plt.show()
