# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 17:42:43 2023

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




tmp = np.linspace(20, 50, 100)

volt = 30*tmp + 0.0214*tmp**3 + np.random.normal(-100,100) +25

df = pd.DataFrame(np.array([volt,tmp]).T, columns=['volt', 'tmp'])



result = M_regression(df["tmp"],df["volt"],3)

print(result.summary())





    
    
