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
    if M==1:
        return words
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

def word_squarer(words,M=1):
    'returns all squares of words from 1 to the degree M.'
    [[str(word)+"**"+str(i+1) for word in words] for i in range(M)]
    comb=[]
    for word in words:
        for i in range(M):
            if i == 0:
                comb.append(str(word))
            else:
                comb.append(str(word)+"**"+str(i+1))
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
        #print(fit.pvalues,comb[fit.pvalues.argmax()-1])
        if np.isnan(fit.pvalues.max()):
            raise Exception("Regression was unsuccesful: p-Values are None\nPlease revise your formula") 
        print("Removing %s wit highest p-Value of:%f"%(comb[fit.pvalues.argmax()-1],fit.pvalues.max()))

        comb.remove(comb[fit.pvalues.argmax()-1])

def Mult_M_regression(df, y_name, M=2, full=True, extend_terms=[], p_val=0.05):
    '''
    \nreturns a fitted regression of the df using y_name as output and a full quadatic(M-fold) model of the remaining columns.
 components with p-value > 0.05 are gradually discarded.
    
    \nExample:
 Mult_M_regression(df, 'a', extend_terms=["np.log(b)","2**b"])
 with df columns ["a","b","c"] will result in the following first regression function:
    \n a ~ +I(b)+I(b*b)+I(b*c)+I(c)+I(c*c)+I(np.log(b))+I(2**b)
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


def keyword_match(input_string, keyword, defaultM=1):
    '''
    \nsearch input_string for keywordN and return [string[:keyword],string[keyword:],N]
    '''
    #set the filter
    pattern = re.compile(f'(.*?)({re.escape(keyword)}(\d+)?)')

    #search for the first instance of the keyword
    match = re.search(pattern, input_string)

    #if a match is found
    if match:
        # Extract the matched groups
        before_keyword = match.group(1)
        after_keyword = input_string[match.end(2):]
        number = int(match.group(3)) if match.group(3) else defaultM

        return before_keyword, after_keyword, number

    # Return None if the keyword is not found
    return None

def formula_regression(df, y_name, formula_elements,p_val=0.05, single="single", inter="inter", full="full"):
    '''
    \nreturns a fitted regression of the df using y_name as output. The formula is derived from formula_elements.
 components with p-value > 0.05 are gradually discarded.

    \nEach element of formula_elements is used as a part of the regression function. The keywords singleM, interM, and fullM (with M beeing a number) are replaced by:
    singleM: col_names**range(M)
    interM: interaction between all col_names up to degree M
    fullM: full quadratic (M-fold) model
    
    \nExample:
 formula_regression(df_scal, "a", ["inter2**single1","np.log(full2)","c**5*a"])
 with df columns ["a","b","c"] will result in the following first regression function:
    \n a ~ +I((b)**(b))+I((b*c)**(b))+I((c)**(b))+I((b)**(c))+I((b*c)**(c))+I((c)**(c))+I(np.log((b)))+I(np.log((b*b)))+I(np.log((b*c)))+I(np.log((c)))+I(np.log((c*c)))+I(c**5*a)
    \n
    \nPlease note that the formula can grow to completely ridiculus lenghts when using this function. Be mindfull when using it and dont do stupid shit like shown in this example.
    '''
    comb=[]
    #get the inputs
    words = [col for col in df.columns if col != y_name]
    #get all possible combinations up to degree M to create the formula
    
    #pain and suffering x3
    single_replaced = []
    for element in formula_elements:
        
        element_insert = [element]
        while True:
            inserter=[]
            for ele in element_insert:
                
                match = keyword_match(ele, single)
                if match == None:
                    single_replaced.extend([ele])
                    continue

                for instert in word_squarer(words, match[2]):
                    inserter.append(match[0]+"("+instert+")"+match[1])
            if element_insert == inserter:
                break
            element_insert=inserter

    inter_replaced=[]
    for element in single_replaced:
        element_insert = [element]
        while True:
            inserter=[]
            for ele in element_insert:
                
                match = keyword_match(ele, inter,defaultM=2)
                if match == None:
                    inter_replaced.extend([ele])
                    continue

                for instert in word_interactor(words, match[2]):
                    inserter.append(match[0]+"("+instert+")"+match[1])
            if element_insert == inserter:
                break
            element_insert=inserter
    
    for element in inter_replaced:
        element_insert = [element]
        while True:
            inserter=[]
            for ele in element_insert:
                
                match = keyword_match(ele, full,defaultM=2)
                if match == None:
                    comb.extend([ele])
                    continue

                for instert in word_multiplicator(words, match[2]):
                    inserter.append(match[0]+"("+instert+")"+match[1])
            if element_insert == inserter:
                break
            element_insert=inserter

    #pass to comb and input to regression_fit
    return regression_fit(df, y_name, comb, p_val=p_val)

class ExcludingStdScaler():
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
    scaler = ExcludingStdScaler(exclude_col,ddof=1)
    fitted = scaler.fit(X)
    x_std = fitted.std(X)
    return x_std, fitted


#TODO ExcludingNormScaler and normer   {'a': -1,'b': -2} for custom min and max