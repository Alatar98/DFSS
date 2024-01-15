

import numpy as np
import pandas as pd
from  statsmodels.formula.api import ols
from scipy.io import loadmat
import scipy.stats as stats
import matplotlib.pyplot as plt

import re

import TSST as TT


# Import and format data
SF_data = pd.read_csv('SystematischerMessfehler.csv', sep=',')
#print(SF_data)
ax1 = plt.figure(1, figsize=(6, 4)).subplots(1, 1)
weights = np.ones_like(SF_data) / len(SF_data)
ax1.hist(SF_data.values.reshape(-1), bins=10, weights=weights,
         label='Histogramm')
#ax1.axis([np.min(SF_data), np.min(SF_data), 0, 10])
ax1.set_xlabel(r'Systematischer Messfehler')
ax1.set_ylabel(r'Relative Häufigkeit')
