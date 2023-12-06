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


def word_iterator(string,words,M,comb):
    comb.append(string)
    words_copy=words.copy()
    for word in words:
        if M==1:
            comb.append(string+"*"+word)
        else: 
            word_iterator(string+"*"+word,words_copy,M-1,comb)
        words_copy.remove(word)

def word_multiplicator(words,M=2):
    comb=[]
    words_copy=words.copy()
    for word in words:
        word_iterator(word,words_copy,M-1,comb)
        words_copy.remove(word)
    return comb


def Mult_M_regression(df,y_name,M=2):
    words = [col for col in df.columns if col != y_name]
    comb = word_multiplicator(words,M)
    while True:
        
        formula = ''.join([y_name+' ~ ']+["+I(%s)"%c for c in comb])
        print("Regressing with: "+formula)
        model = ols(formula,df)
        fit = model.fit()
        
        if (fit.pvalues < 0.05).all():
            return fit
        
        print("Removing %s wit highest p-Value of:%f"%(comb[fit.pvalues.argmax()-1],fit.pvalues.max()))
        comb.remove(comb[fit.pvalues.argmax()-1])

class ExcludingScaler():
    def __init__(self, exclude_col=None, mean_val=None, std_val=None, ddof=1, **kwargs):
        self.exclude_col = exclude_col
        self.ddof=ddof
        self.mean_val_orig=mean_val
        self.std_val_orig=std_val


    def fit(self, X):
        
        if self.exclude_col is not None and self.exclude_col in X.columns:
            X_to_fit = X.drop(self.exclude_col, axis=1)
        else:
            X_to_fit = X.copy()

        self.mean_val = X_to_fit.mean()
        if self.mean_val_orig != None:
            mean_val = pd.concat([self.mean_val, self.mean_val_orig], ignore_index=True, sort=False)
            self.mean_val = mean_val.drop_duplicates(keep="last",  ignore_index=True)
        
        self.std_val = X_to_fit.std(ddof=self.ddof)
        if self.std_val_orig != None:
            std_val = pd.concat([self.std_val, self.std_val_orig], ignore_index=True, sort=False)
            self.std_val = std_val.drop_duplicates(keep="last",  ignore_index=True)

        return self

    def std(self, X):
        if self.exclude_col is not None and self.exclude_col in X.columns:
            # Exclude the specified column
            X_to_scale = X.drop(self.exclude_col, axis=1)
        
        else:
            # No column to exclude, simply apply standardization
            X_to_scale = X

        # Perform operations on the DataFrame without the excluded column
        # For example, you can apply standardization using the scaler

        X_transformed= (X_to_scale - self.mean_val)/self.std_val

        #X_transformed = pd.DataFrame(self.transform(X_to_scale), index=X_to_scale.index, columns=X_to_scale.columns)

        if self.exclude_col is not None and self.exclude_col in X.columns:
            # Add the column back to the DataFrame
            X_transformed.insert(0, self.exclude_col, X[self.exclude_col])

        return X_transformed
    
    def rev(self, X):
        if self.exclude_col is not None and self.exclude_col in X.columns:
            # Exclude the specified column
            X_to_scale = X.drop(self.exclude_col, axis=1)
        
        else:
            # No column to exclude, simply apply standardization
            X_to_scale = X

        # Perform operations on the DataFrame without the excluded column
        # For example, you can apply standardization using the scaler

        X_transformed= X_to_scale * self.std_val + self.mean_val

        #X_transformed = pd.DataFrame(self.transform(X_to_scale), index=X_to_scale.index, columns=X_to_scale.columns)

        if self.exclude_col is not None and self.exclude_col in X.columns:
            # Add the column back to the DataFrame
            X_transformed.insert(0, self.exclude_col, X[self.exclude_col])

        return X_transformed



def stder(X, exclude_col=None, ddof=1):
    scaler = ExcludingScaler(exclude_col,ddof=1)
    fitted = scaler.fit(X)
    x_std = fitted.std(X)
    return x_std, fitted
