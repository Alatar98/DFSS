# -*- coding: utf-8 -*-
"""
TobisSixSigTools

Created on Wed Dec 06 15:48:43 2023

@author: freyt
"""

import numpy as np
import pandas as pd
from  statsmodels.formula.api import ols
import re

from sklearn.preprocessing import StandardScaler


def word_multiplicator_backend(string,words,M,comb):
    'only used by word_multiplicator'
    comb.append(string)
    words_copy=words.copy()
    for word in words:
        if M==1:
            comb.append(string+"*"+word)
        else: 
            word_multiplicator_backend(string+"*"+word,words_copy,M-1,comb)
        words_copy.remove(word)


def word_multiplicator(words,M=2):
    'returns all possible products with words up to degree M.'
    comb=[]
    words_copy=words.copy()
    for word in words:
        word_multiplicator_backend(word,words_copy,M-1,comb)
        words_copy.remove(word)
    return comb


def word_interactor_backend(string,words,M,comb):
    'only used by word_interactor'
    comb.append(string)
    words_copy=words.copy()
    for word in words:
        if M==1:
            comb.append(string+"*"+word)
        else: 
            words_copy.remove(word)
            word_interactor_backend(string+"*"+word,words_copy,M-1,comb)
        


def word_interactor(words,M=2):
    'returns all possible interactions with words up to degree M.'
    #m must be smaller then len(words)
    if M>len(words):
        M=len(words)
    comb=[]
    words_copy=words.copy()
    for word in words:
        words_copy.remove(word)
        word_interactor_backend(word,words_copy,M-1,comb)
    return comb

def regression_fit(df, y_name, comb, p_val=0.05):
    '''
    \nreturns a fitted regression of the df using y_name as output and the formula of comb elements.
    \ncomponents with p-value > p_val are gradually discarded.
    '''

    #iterate until no p-value is bigger then p_val
    while True:
        #create the formula to be used based on the (remaining) combinations
        formula = ''.join([y_name+' ~ ']+["+I(%s)"%c for c in comb])
        print("Regressing with: "+formula)

        #create the model and fit
        model = ols(formula,df)
        fit = model.fit()
        
        #check if regression has no more unnecesary components (return)
        if (fit.pvalues < p_val).all():
            print("Regression done")
            return fit
        
        #remove component with highest p-value
        print("Removing %s wit highest p-Value of:%f"%(comb[fit.pvalues.argmax()-1],fit.pvalues.max()))
        comb.remove(comb[fit.pvalues.argmax()-1])



#TODO add terms explict , terms implicit (singleM, interM, fullM)  seperate function for implicit
def Mult_M_regression(df, y_name, M=2, full=True, extend_terms=[], p_val=0.05):
    '''
    \nreturns a fitted regression of the df using y_name as output and a full quadatic(M-fold) model of the remaining columns.
    \ncomponents with p-value > 0.05 are gradually discarded.
    
    \nExample:
    \nMult_M_regression(df, 'a', extend_terms=["np.log(b)","2**b"])
    \nwith df columns ["a","b","c"] will result in the following first regression function
    \na ~ +I(b)+I(b*b)+I(b*c)+I(c)+I(c*c)+I(np.log(b))+I(2**b)
    '''

    #get the inputs
    words = [col for col in df.columns if col != y_name]
    #get all possible combinations up to degree M to create the formula
    if full:
        comb = word_multiplicator(words,M)
    else:
        comb = word_interactor(words,M)

    comb.extend(extend_terms)

    return regression_fit(df, y_name, comb, p_val=p_val)


#TODO  finish   
def replace_simple(instring,words):
    
    pattern = re.compile(f'{re.escape("single")}(\d+)?')

    def replacement(match):
        # Extract the number from the match, or default to 1 if not present
        count = int(match.group(1)) if match.group(1) else 1
        # Repeat the replacement string according to the count
        return ["a**"+count,"a"]  #a + a**2 + ... +a**count

    # Use the re.sub function with a replacement function
    result = pattern.sub(replacement, instring)
    return result

#TODO add terms  terms implicit (singleM, interM, fullM)
def formula_regression(df, y_name, formula_elements,p_val=0.05):
    '''
    \nreturns a fitted regression of the df using y_name as output and a full quadatic(M-fold) model of the remaining columns.
    \ncomponents with p-value > 0.05 are gradually discarded.
    
    \nExample:
    \nMult_M_regression(df, 'a', extend_terms=["np.log(b)","2**b"])
    \nwith df columns ["a","b","c"] will result in the following first regression function
    \na ~ +I(b)+I(b*b)+I(b*c)+I(c)+I(c*c)+I(np.log(b))+I(2**b)
    '''
    #TODO write new description
    comb=[]
    #get the inputs
    words = [col for col in df.columns if col != y_name]
    #get all possible combinations up to degree M to create the formula
    #comb = word_multiplicator(words,M)

    #comb = word_interactor(words,M)

    comb.extend(extend_terms)

    return regression_fit(df, y_name, comb, p_val=p_val)

class ExcludingScaler():
    '''
    Scaler with (X - X.mean)/X.std\n
    The Scaler ignores the given column.\n
    std has a ddof option.\n
    the scaler has the functions:\n
    fit to fit the scaler\n
    std to transform the given data\n
    rev to inverse_transfomr the given data
    '''
    def __init__(self, exclude_col=None, mean_val=None, std_val=None, ddof=1, **kwargs):
        self.exclude_col = exclude_col
        self.ddof=ddof
        self.mean_val_orig=mean_val
        self.std_val_orig=std_val


    def fit(self, X):
         # check for the specified column
        if self.exclude_col is not None and self.exclude_col in X.columns:
            # Exclude the specified column
            X_to_fit = X.drop(self.exclude_col, axis=1)
        else:
            # No column to exclude, simply fit
            X_to_fit = X.copy()

        #TODO check if the init overwrite works
        #calcuate mean
        self.mean_val = X_to_fit.mean()
        #if mean values where given at init overwrite the calculated values with the given ones
        if self.mean_val_orig != None:
            mean_val = pd.concat([self.mean_val, self.mean_val_orig], ignore_index=True, sort=False)
            self.mean_val = mean_val.drop_duplicates(keep="last",  ignore_index=True)
        
        #calcuate std
        self.std_val = X_to_fit.std(ddof=self.ddof)
        #if std values where given at init overwrite the calculated values with the given ones
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

        # Transform the Data
        X_transformed= (X_to_scale - self.mean_val)/self.std_val

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

        # Transform the Data
        X_transformed= X_to_scale * self.std_val + self.mean_val

        if self.exclude_col is not None and self.exclude_col in X.columns:
            # Add the column back to the DataFrame
            X_transformed.insert(0, self.exclude_col, X[self.exclude_col])

        return X_transformed



def stder(X, exclude_col=None, ddof=1):
    '''
    creates a ExcludingScaler fitted for X excluding the column exclude_col.\n
    returns the fitted values and the scaler.
    '''
    scaler = ExcludingScaler(exclude_col,ddof=1)
    fitted = scaler.fit(X)
    x_std = fitted.std(X)
    return x_std, fitted
