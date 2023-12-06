# -*- coding: utf-8 -*-
"""
TobisSixSigTools

Created on Wed Dec 06 15:48:43 2023

@author: freyt
"""

import numpy as np
import pandas as pd
from  statsmodels.formula.api import ols


def M_regression(X, Y, M=1):
    Ms = list(range(1,M+1))
    df = pd.DataFrame(np.array([X,Y]).T, columns=['X', 'Y'])
    while True:
        
        formula = ''.join(['Y ~ ']+["+I(X**%i)"%m for m in Ms])
        model = ols(formula,df)
        fit = model.fit()
        
        if (fit.pvalues < 0.05).all():
            return fit
        
        print("Removing X**%i wit highest p-Value of:%f"%(fit.pvalues.argmax(),fit.pvalues.max()))
        Ms.remove(fit.pvalues.argmax())


