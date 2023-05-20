import math
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, preprocessing

from cpca import CPCA
from ccpca import CCPCA

dataset = datasets.load_wine()

X = dataset.data
y = dataset.target
X = preprocessing.scale(X)

cpca = CPCA()

# manual alpha selection
# cpca.fit(fg=X[y == 0], bg=X[y != 0], alpha=2.15)

# auto alpha selection
cpca.fit(fg=X[y == 0], bg=X[y != 0])
X_r = cpca.transform(X)

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], [0, 1, 2]):
    plt.scatter(
        X_r[y == i, 0],
        X_r[y == i, 1],
        color=color,
        alpha=.8,
        lw=lw,
        label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title(f'cPCA of IRIS dataset (alpha={cpca.get_best_alpha()})')
plt.show()

ccpca = CCPCA()

# apply fit and transform seaparately
# ccpca.fit(X[y == 0], X[y != 0], var_thres_ratio=0.5, max_log_alpha=0.5)
# X_r2 = ccpca.transform(X)

# apply fit and transform at the same time
X_r2 = ccpca.fit_transform(
    X[y == 0], X[y != 0], var_thres_ratio=0.5, max_log_alpha=0.5)

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], [0, 1, 2]):
    plt.scatter(
        X_r2[y == i, 0],
        X_r2[y == i, 1],
        color=color,
        alpha=.8,
        lw=lw,
        label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title(f'ccPCA of IRIS dataset (alpha ={ccpca.get_best_alpha()})')
plt.show()
