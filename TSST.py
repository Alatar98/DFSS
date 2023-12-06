# -*- coding: utf-8 -*-
"""
TobisSixSigTools

Created on Wed Dec 06 15:48:43 2023

@author: freyt
"""

import numpy as np
import pandas as pd
from  statsmodels.formula.api import ols

from sklearn.preprocessing import StandardScaler



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


def Mult_M_regression():
    return "not done"

class ExcludingScaler(StandardScaler):
    def __init__(self, exclude_col=None, **kwargs):
        self.exclude_col = exclude_col
        super().__init__(**kwargs)

    def fit(self, X):
        if self.exclude_col is not None and self.exclude_col in X.columns:
            X_to_fit = X.drop(self.exclude_col, axis=1)
            super().fit(X_to_fit)
        else:
            super().fit(X_to_fit)
        return self

    def std(self, X):
        if self.exclude_col is not None and self.exclude_col in X.columns:
            # Exclude the specified column
            X_to_scale = X.drop(self.exclude_col, axis=1)

            # Perform operations on the DataFrame without the excluded column
            # For example, you can apply standardization using the scaler
            X_transformed = pd.DataFrame(self.transform(X_to_scale), index=X_to_scale.index, columns=X_to_scale.columns)


            # Add the column back to the DataFrame
            X_transformed.insert(0, self.exclude_col, X[self.exclude_col])

        else:
            # No column to exclude, simply apply standardization
            X_to_scale = X
            X_transformed = pd.DataFrame(self.transform(X_to_scale), index=X_to_scale.index, columns=X_to_scale.columns)

        return X_transformed
    
    def rev(self, X):
        if self.exclude_col is not None and self.exclude_col in X.columns:
            # Exclude the specified column
            X_to_scale = X.drop(self.exclude_col, axis=1)
            # Perform operations on the DataFrame without the excluded column
            # For example, you can apply standardization using the scaler
            X_transformed = pd.DataFrame(self.inverse_transform(X_to_scale), index=X_to_scale.index, columns=X_to_scale.columns)

            # Add the column back to the DataFrame
            X_transformed.insert(0, self.exclude_col, X[self.exclude_col])
        else:
            # No column to exclude, simply apply standardization
            X_to_scale = X
            X_transformed = pd.DataFrame(self.inverse_transform(X_to_scale), index=X_to_scale.index, columns=X_to_scale.columns)
        
        return X_transformed



def stder(X, exclude_col=None):
    scaler = ExcludingScaler(exclude_col)
    fitted = scaler.fit(X)
    x_std = fitted.std(X)
    return x_std, fitted
