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

import scipy.stats as stats
from sympy import symbols, sympify
import sympy as syms
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler


def _word_multiplicator_backend(string,words,M,comb):
    'only used by word_multiplicator'
    comb.append(string)
    words_copy=words.copy()
    for word in words:
        if M==1:
            comb.append(string+"*"+word)
        else: 
            _word_multiplicator_backend(string+"*"+word,words_copy,M-1,comb)
        words_copy.remove(word)


def _word_multiplicator(words,M=2):
    'returns all possible products with words up to degree M.'
    if M==1:
        return words
    comb=[]
    words_copy=words.copy()
    for word in words:
        _word_multiplicator_backend(word,words_copy,M-1,comb)
        words_copy.remove(word)
    return comb


def _word_interactor_backend(string,words,M,comb):
    'only used by word_interactor'
    comb.append(string)
    words_copy=words.copy()
    for word in words:
        if M==1:
            comb.append(string+"*"+word)
        else: 
            words_copy.remove(word)
            _word_interactor_backend(string+"*"+word,words_copy,M-1,comb)
        


def _word_interactor(words,M=2):
    'returns all possible interactions with words up to degree M.'
    #m must be smaller then len(words)
    if M>len(words):
        M=len(words)
    comb=[]
    words_copy=words.copy()
    for word in words:
        words_copy.remove(word)
        _word_interactor_backend(word,words_copy,M-1,comb)
    return comb

def _word_squarer(words,M=1):
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


def regression_fit(df, y_name, comb, p_val=0.05,verbose=False):
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
        
        if verbose:
            print(fit.result())
        
        #check if regression has no more unnecesary components (return)
        if (fit.pvalues < p_val).all():
            print("Regression done")
            return model, fit
        
        #remove component with highest p-value
        #print(fit.pvalues,comb[fit.pvalues.argmax()-1])
        if np.isnan(fit.pvalues.max()):
            raise Exception("Regression was unsuccesful: p-Values are None\nPlease revise your formula") 
            
        print("Removing %s wit highest p-Value of:%f"%(comb[fit.pvalues.argmax()-1],fit.pvalues.max()))
        comb.remove(comb[fit.pvalues.argmax()-1])

def Mult_M_regression(df, y_name, M=2, full=True, extend_terms=[], p_val=0.05,verbose=False):
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
        comb = _word_multiplicator(words,M)
    else:
        comb = _word_interactor(words,M)

    comb.extend(extend_terms)

    return regression_fit(df, y_name, comb, p_val=p_val,verbose=verbose)


def _keyword_match(input_string, keyword, defaultM=1):
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

def formula_regression(df, y_name, formula_elements,p_val=0.05, single="single", inter="inter", full="full",verbose=False):
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
                
                match = _keyword_match(ele, single)
                if match == None:
                    single_replaced.extend([ele])
                    continue

                for instert in _word_squarer(words, match[2]):
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
                
                match = _keyword_match(ele, inter,defaultM=2)
                if match == None:
                    inter_replaced.extend([ele])
                    continue

                for instert in _word_interactor(words, match[2]):
                    inserter.append(match[0]+"("+instert+")"+match[1])
            if element_insert == inserter:
                break
            element_insert=inserter
    
    for element in inter_replaced:
        element_insert = [element]
        while True:
            inserter=[]
            for ele in element_insert:
                
                match = _keyword_match(ele, full,defaultM=2)
                if match == None:
                    comb.extend([ele])
                    continue

                for instert in _word_multiplicator(words, match[2]):
                    inserter.append(match[0]+"("+instert+")"+match[1])
            if element_insert == inserter:
                break
            element_insert=inserter

    #pass to comb and input to regression_fit
    return regression_fit(df, y_name, comb, p_val=p_val,verbose=verbose)

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

    def scl(self, X):
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
    scaler = ExcludingStdScaler(exclude_col,ddof=ddof)
    fitted = scaler.fit(X)
    x_std = fitted.scl(X)
    return x_std, fitted


class ExcludingNormScaler():
    '''
    Scaler with (X - X.min)/(X.max-X.min)\n
    The Scaler ignores the given column.\n
    std has a ddof option.\n
    the scaler has the functions:\n
    fit to fit the scaler\n
    std to transform the given data\n
    rev to inverse_transfomr the given data
    '''
    def __init__(self, exclude_col=None, min_val={}, max_val={}, **kwargs):
        self.exclude_col = exclude_col
        self.min_val_orig=min_val
        self.max_val_orig=max_val


    def fit(self, X):
         # check for the specified column
        if self.exclude_col is not None and self.exclude_col in X.columns:
            # Exclude the specified column
            X_to_fit = X.drop(self.exclude_col, axis=1)
        else:
            # No column to exclude, simply fit
            X_to_fit = X.copy()

        #TODO check if the init overwrite works
        #get min and max values
        self.min_val = X_to_fit.min()
        self.max_val = X_to_fit.max()
        #if min or max values where given at init overwrite the calculated values with the given ones
        for key in self.min_val.index:
            if key in self.min_val_orig:
                self.min_val[key] = self.min_val_orig[key]

            if key in self.max_val_orig:
                self.max_val[key] = self.max_val_orig[key]

        return self

    def scl(self, X):
        if self.exclude_col is not None and self.exclude_col in X.columns:
            # Exclude the specified column
            X_to_scale = X.drop(self.exclude_col, axis=1)
        
        else:
            # No column to exclude, simply apply standardization
            X_to_scale = X

        # Transform the Data
        X_transformed = (X_to_scale - self.min_val)/(self.max_val-self.min_val)

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
        X_transformed = X_to_scale * (self.max_val-self.min_val)  + self.min_val

        if self.exclude_col is not None and self.exclude_col in X.columns:
            # Add the column back to the DataFrame
            X_transformed.insert(0, self.exclude_col, X[self.exclude_col])

        return X_transformed

def normer(X, exclude_col=None, min_val={}, max_val={}):
    '''
    creates a ExcludingScaler fitted for X excluding the column exclude_col.\n
    returns the fitted values and the scaler.
    '''
    scaler = ExcludingNormScaler(exclude_col,min_val=min_val, max_val=max_val)
    fitted = scaler.fit(X)
    x_std = fitted.scl(X)
    return x_std, fitted

#TODO: MinMaxScaler

def convolute(prob, formula_str, res, x_scaler=5, verbose=False):
    '''
    Parameters
    ----------
    \nprob : {"name": scipy.stats.rv_continous}
        names and distributions of the formula components
    \nformula_str : string
        the formula
    \nres : Number
        resolution of the convolution
    \nx_scaler: Number, optional The default is 5.
        size of the x axis is 2 * x_scaler * std_dev * local_sensitivity

    Returns
    -------
    conv_y, conv_x
    convolution propability density function and x values of the convolution
    '''
    
    #get list of names and prefix _ to all names to avoid convusion with predefined symbols
    names_list_orig=list(prob.keys())
    names_list = ['_'+name for name in names_list_orig]
    prob = {'_'+k: v for k, v in prob.items()}
    for name in names_list_orig:
        formula_str=formula_str.replace(name,'_'+name)

    #get means and std
    means={k: v.mean() for k, v in prob.items()}
    stds={k: v.std() for k, v in prob.items()}
    
    # Create Symbol objects from variable names
    symbols_list = symbols(names_list)

    # Build the expression using the formula string
    expression = sympify(formula_str)
    
    #if verbose is set print the symbols and expression
    if verbose:
        print("symbols: ",symbols_list)
        print("expression: ",expression)
        

    # Replace any variable names in the expression with the corresponding symbols
    for var_name, var_symbol in zip(names_list, symbols_list):
        expression = expression.subs(var_name, var_symbol)
     

    #calculation of sensitivities
    sens={}
    for name in names_list:
        sens[name] = abs(float(expression.diff(name).evalf(subs=means)))
        
    #if verbose is set print the sensitivities
    if verbose:
        print("sensitivities: ",sens)

    #calculate propability density functions
    pdfs={}
    w_ges=0
    for name in names_list:
        w = float(np.abs(x_scaler*stds[name]*sens[name]))
        w_ges+=w
        scaled_ax = np.array(np.arange(-w, w, res)/sens[name]+means[name], dtype=np.float64)

        pdfs[name]=prob[name].pdf(scaled_ax)
        
    #if verbose is set print the lenght of the pdfs
    if verbose:
        print("pdf_lenght: ",{key:len(pdf) for key,pdf in pdfs.items()})
        
    #convolute
    conv=[1]   #/res  to make it constant but canceled out by /np.max later
    for name in names_list:
        conv=np.convolve(conv, pdfs[name])

    cdf = np.cumsum(conv)
    conv = conv/np.max(cdf)
    
    #if verbose is set print the sum of all objects. this value should be close to one
    if verbose:
        #add the scaling that is normally done during convolution and cumsum
        man_scaling_check=np.max(cdf)*res**(len(names_list))
        #add the scaling that results from changing the x axis  instead of the stdev when generating the pdfs
        for sens in sens.values():
            man_scaling_check=man_scaling_check/sens
        print("cumsum: ",man_scaling_check)
    

    return conv, np.linspace(-w_ges, w_ges, len(conv))


#confidence Range calculation
def confidenceRange(data, gamma, std=None, verbose=False):
    '''
    \nif std is given calculate confidence range of mean of data with gamma and return mu_min, mu_max.
    \nif std is not given calculate confidence range of mean and std of data with gamma and return mu_min, mu_max, sdev_min, sdev_max.
    \n
    \nTBH itsprobably better to simply use stats.t.interval(gamma, len(data)-1,loc=np.mean(data),scale=np.std(data,ddof=1)/np.sqrt(len(data))) or simmilar
    '''
    #base data
    m = np.mean(data)
    s = np.std(data, ddof=1) if std is None else std
    N=len(data)

    # Determine konstants for mean
    c1 = stats.t.ppf((1-gamma)/2, N-1)
    c2 = stats.t.ppf((1+gamma)/2, N-1)
    mu_min = m - ((c2*s)/np.sqrt(N))
    mu_max = m - ((c1*s)/np.sqrt(N))

    if verbose:
        if std==None:
            print("Calculating confidence range with unknown variance:")
        else:
            print("Calculating confidence range with known variance:")

        print("c1: ",c1,"   c2: ",c2, "   mu_min: ",mu_min, "   mu_max: ",mu_max)

    if std==None:
        c1 = stats.chi2.ppf((1-gamma)/2, N-1)
        c2 = stats.chi2.ppf((1+gamma)/2, N-1)
        sdev_min = np.sqrt((N-1)/c2)*s
        sdev_max = np.sqrt((N-1)/c1)*s

        if verbose:
            print("c1: ",c1,"   c2: ",c2, "   sdev_min: ",sdev_min, "   sdev_max: ",sdev_max)

        return (mu_min, mu_max) , (sdev_min, sdev_max)
    
    return (mu_min, mu_max) , (std, std)


#confidence Range for the Comparison of Populations calculation
def confidenceRangeComp(data1, data2, gamma, std=None, verbose=False):
    '''
    \nif std is given calculate confidence range of the difference in mean values with gamma and return dmu_min, dmu_max.
    \nif std is not given calculate confidence range of  the difference in mean values  and rato s2/s1 of the std values with gamma and return mu_min, mu_max, sdev_min, sdev_max.
    '''
    
    #base data
    m1 = np.mean(data1)
    m2 = np.mean(data2)
    N1=len(data1)
    N2=len(data2)
    s1 = np.std(data1, ddof=1) 
    s2 = np.std(data2, ddof=1) 
    s = np.sqrt(((N1-1)*s1**2 + N2*s2**2) / (N1+N2-2))  if std is None else std


    # Determine konstants for mean
    c1 = stats.t.ppf((1-gamma)/2, N1+N2-2)
    c2 = stats.t.ppf((1+gamma)/2, N1+N2-2)
    dmu_min = m1 - m2 - c2*np.sqrt(1/N1+1/N2)*s
    dmu_max = m1 - m2 - c1*np.sqrt(1/N1+1/N2)*s

    if verbose:
        if std==None:
            print("Calculating confidence range with unknown variance:")
        else:
            print("Calculating confidence range with known variance:")

        print("c1: ",c1,"   c2: ",c2, "   dmu_min: ",dmu_min, "   dmu_max: ",dmu_max)

    if std==None:
        c1 = stats.f.ppf((1-gamma)/2, N1-1, N2-1)
        c2 = stats.f.ppf((1+gamma)/2, N1-1, N2-1)
        sdev_ratio_min = s2/s1*c1
        sdev_ratio_max = s2/s1*c2

        if verbose:
            print("c1: ",c1,"   c2: ",c2, "   sdev_min: ",sdev_ratio_min, "   sdev_max: ",sdev_ratio_max)

        return dmu_min, dmu_max, sdev_ratio_min, sdev_ratio_max
    
    return dmu_min, dmu_max


def predictionRange(data,gamma,mean=None,std=None,verbose=False):
    
    m = np.mean(data) if mean is None else mean
    s = np.std(data, ddof=1) if std is None else std
    N=len(data)
    

    if mean==None and std==None:
        range=stats.t.interval(gamma,N-1,loc=m,scale=s*np.sqrt(1+1/N))
    elif std==None:
        range=stats.t.interval(gamma,N-1,loc=m,scale=s)
    elif mean==None:
        range=stats.norm.interval(gamma,loc=m,scale=s*np.sqrt(1+1/N))
    else:
        range=stats.norm.interval(gamma,loc=m,scale=s)

    return range



def HistDist(data, bins=15, da=100, figsize=(12, 4), subtitle="Histogramm with Distribution", verbose=False,set_axis=False):
    """
    Plots histograms with corresponding distribution curves for each column in the given data.

    Parameters:
    - data: pandas DataFrame
        The input data containing the columns to be plotted.
    - bins: int, optional
        The number of bins to use for the histogram. Default is 15.
    - da: int, optional
        The number of data points to use for the distribution curve. Default is 100.
    - figsize: tuple, optional
        The size of the figure. Default is (12, 4).
    - subtitle: str, optional
        The subtitle for the figure. Default is "Histogramm with Distribution".
    - verbose: bool, optional
        Whether to print additional information. Default is False.
    - set_axis: bool, optional
        Whether to set the axis limits based on the data range. Default is False.

    Returns:
    - axis: numpy array
        The axes objects of the subplots.

    """
    
    axis = []
    fig1 = plt.figure(1, figsize=figsize)
    fig1.subtitle = subtitle
    axis = fig1.subplots(1, len(data.columns), squeeze=False)
    for i, (col, ax) in enumerate(zip(data.columns, axis[0])):
        sides = (data[col].max() - data[col].min())/10
        x_ax = np.linspace(data[col].min()-sides, data[col].max()+sides, da)
        ax.hist(data[col], density=True, bins=bins, facecolor='b', label="Meassured")
        ax.plot(x_ax, stats.norm.pdf(x_ax, np.mean(data[col]), np.std(data[col], ddof=1)), 'r', label="Distribution")
        if set_axis:
            ax.axis([data[col].min()-sides, data[col].max()+sides, 0, 1])
        ax.grid(True)
        ax.set_xlabel(r'Unit')
        ax.set_ylabel(r'Relative Frequencie')
        ax.set_title(col)
    axis[0][-1].legend()

    return axis


def correlation(data1, data2, gamma=0.95):
    """
    Calculates the Pearson's correlation coefficient between two datasets and returns the correlation coefficient,
    p-value, and confidence interval.

    Parameters:
    data1 (array-like): The first dataset.
    data2 (array-like): The second dataset.
    gamma (float, optional): The confidence level for the confidence interval. Default is 0.95.

    Returns:
    corr (float): The Pearson's correlation coefficient.
    p (float): The p-value.
    corr_min (float): The lower bound of the confidence interval.
    corr_max (float): The upper bound of the confidence interval.
    """
    
    corr, p = stats.pearsonr(data1, data2)

    print('Pearsons correlation: %.3f' % corr)
    print('p-value: %.3f' % p)

    N = len(data1)

    c1 = stats.norm.ppf((1 - gamma)/2)
    c2 = stats.norm.ppf((1 + gamma)/2)
    corr_min = np.tanh(np.arctanh(corr) - c2/np.sqrt(N-3))
    corr_max = np.tanh(np.arctanh(corr) - c1/np.sqrt(N-3))

    print(f"Confidence Range: {round(corr_min, 4)} <= {round(corr, 4)} <= {round(corr_max, 4)}")
    
    return corr, p, corr_min, corr_max



def tolerancing(prob, formula_str, gamma, corr={}, verbose=False, res=0.000001,x_scaler=5,N=10000):

    tol={}          #the tolerance values are stored here
    functions={}    #if the method to calculate the tolerance uses a function it is stored here

    #get list of names and prefix _ to all names to avoid convusion with predefined symbols
    names_list_orig=list(prob.keys())
    names_list = ['_'+name for name in names_list_orig]
    corr = {"_"+k: [("_"+name, val) for (name, val) in values] for k, values in corr.items()}
    prob = {'_'+k: v for k, v in prob.items()}
    for name in names_list_orig:
        formula_str=formula_str.replace(name,'_'+name)

    #get means and std
    means={k: v.mean() for k, v in prob.items()}
    stds={k: v.std() for k, v in prob.items()}
    
    ###########statistical tolerancing with linerization#################

    # Create Symbol objects from variable names
    symbols_list = symbols(names_list)

    # Build the expression using the formula string
    expression = sympify(formula_str)
    
    #if verbose is set print the symbols and expression
    if verbose:
        print("symbols: ",symbols_list)
        print("expression: ",expression)
        

    # Replace any variable names in the expression with the corresponding symbols
    for var_name, var_symbol in zip(names_list, symbols_list):
        expression = expression.subs(var_name, var_symbol)
     

    #calculation of sensitivities
    sens={}
    for name in names_list:
        sens[name] = abs(float(expression.diff(name).evalf(subs=means)))

    #if verbose is set print the sensitivities
    if verbose:
        print("sensitivities: ",sens)

    #calculate tolerance according to central limit theorem
    centr_limit=sum([(stds[name] * sens[name])**2 for name in names_list])


    c1 = stats.norm.ppf((1 - gamma)/2)
    c2 = stats.norm.ppf((1 + gamma)/2)
    centr_limit_base = (c2 - c1)
    tol["CentralLimit_noCorr"] = np.sqrt(centr_limit)*centr_limit_base

    for name, values in corr.items():
        for corr_name, corr_val in values:
            centr_limit+=2*stds[name]*stds[corr_name]*sens[name]*sens[corr_name]*corr_val

    tol["CentralLimit"] = np.sqrt(centr_limit)*centr_limit_base
        
    

    #calculate propability density functions
    pdfs={}
    w_ges=0
    for name in names_list:
        w = float(np.abs(x_scaler*stds[name]*sens[name]))
        w_ges+=w
        scaled_ax = np.array(np.arange(-w, w, res)/sens[name]+means[name], dtype=np.float64)

        pdfs[name]=prob[name].pdf(scaled_ax)
        
    #if verbose is set print the lenght of the pdfs
    if verbose:
        print("pdf_lenght: ",{key:len(pdf) for key,pdf in pdfs.items()})
        
    #convolute
    conv=[1]   #/res  to make it constant but canceled out by /np.max later
    for name in names_list:
        conv=np.convolve(conv, pdfs[name])

    cdf = np.cumsum(conv)
    conv = conv/np.max(cdf)
    
    #if verbose is set print the sum of all objects. this value should be close to one
    if verbose:
        #add the scaling that is normally done during convolution and cumsum
        man_scaling_check=np.max(cdf)*res**(len(names_list))
        #add the scaling that results from changing the x axis  instead of the stdev when generating the pdfs
        for sens in sens.values():
            man_scaling_check=man_scaling_check/sens
        print("cumsum: ",man_scaling_check)

    x = np.linspace(-w_ges, w_ges, len(conv))

    conv_cdf = np.cumsum(conv)

    indexmin = np.min(np.where(conv_cdf >= (1-gamma)/2))
    indexmax = np.min(np.where(conv_cdf >= (1+gamma)/2))

    z_maxCon = x[indexmax]
    z_minCon = x[indexmin]
    z_tolerance_con = z_maxCon - z_minCon

    tol["convolution"] = (z_tolerance_con)

    

    functions["convolution"] = (x,conv_cdf)

    #statistical simulation
    sim = {}

    for name in names_list:
        if name not in corr:
            sim[name] = prob[name].rvs(size=N)
        
    for name in names_list:
        if name in corr:
            sim[name] = np.zeros(N)
            rem=0
            for corr_name, corr_val in corr[name]:
                sim[name] += (sim[corr_name]-means[corr_name])*corr_val*stds[name]/stds[corr_name]
                rem+=corr_val**2
            
            sim[name] += means[name] + (prob[name].rvs(size=N)-means[name])*np.sqrt(1-rem)


    stat_sim =[]
    for i in range(N):
        #print([(name,sim[name][i]) for name in names_list])
        #print(expression)
        stat_sim.append(float(expression.evalf(subs={name:sim[name][i] for name in names_list})))

    # Tolerance calculation according monte carlo simulation mcs
    c1 = stats.t.ppf((1-gamma)/2, N-1)
    c2 = stats.t.ppf((1+gamma)/2, N-1)
    tolerance_mcs = np.std(stat_sim, ddof=1)*np.sqrt(1+1/N)*(c2-c1)

    tol["monteCarloSimulation"] = tolerance_mcs

    stat_sim_sort = np.sort(stat_sim)
    stat_sim_cdf = np.arange(1, N+1, 1)/N

    index_min = np.min(np.where(stat_sim_cdf >= (1-gamma)/2))
    index_max = np.min(np.where(stat_sim_cdf >= (1+gamma)/2))
    stat_sim_num = stat_sim_sort[index_max] - stat_sim_sort[index_min]

    tol["numericalSimulation"] = stat_sim_num

    functions["numericalSimulation"] = (stat_sim_sort,stat_sim_cdf)



    return tol, functions



def hypothesis_t_test(HYP_GAMMA0, HYP_ALPHA, HYP_SIG, DELTA_MAX, ds, detect_warsch, plot=True,x_label="What/Unit"):
    
    c1 = stats.norm.ppf(HYP_ALPHA/2)
    c2 = stats.norm.ppf(1 - HYP_ALPHA/2)
    hyp_limit_1 = HYP_GAMMA0 + c1*HYP_SIG
    hyp_limit_2 = HYP_GAMMA0 + c2*HYP_SIG

    print("Lower bound of hypothesis test:", hyp_limit_1)
    print("Upper bound of hypothesis test:", hyp_limit_2)


    # Standarddeviation of poulation is known, z-test to gamma0 = 1
    # calculation interval for accpetance of hypothesis
    hyp_gamma_var = np.arange(HYP_GAMMA0-DELTA_MAX, HYP_GAMMA0+DELTA_MAX+ds, ds)

    hyp_guete = 1 + stats.norm.cdf((hyp_limit_1 - hyp_gamma_var)/HYP_SIG) - stats.norm.cdf((hyp_limit_2 - hyp_gamma_var)/HYP_SIG)
    index_limit = np.min(np.where(hyp_guete <= detect_warsch))
    gamma0_delta_limit = HYP_GAMMA0 - hyp_gamma_var[index_limit]
    print(r"With probability of {detect_warsch}% detectable deviation:", gamma0_delta_limit)


    if plot:
        fig2 = plt.figure(2, figsize=(6, 4))
        fig2.suptitle('')
        ax1 = fig2.subplots(1, 1)
        ax1.plot(hyp_gamma_var, hyp_guete, 'b')
        ax1.set_xlabel(x_label)
        ax1.set_ylabel('Quality function')
        ax1.grid(True)




def hypothesistest(data, alpha, mu0, std0, verbose = False, distribution=stats.norm(),gamma=0.95):
    '''
    \nhypothesistest for mu0 and sig0
    \n
    '''


    return "Not Done Yet"
    #ttest_1samp
    
    #base data

    m = np.mean(data)
    s = np.std(data, ddof=1)
    N=len(data)

    # Determine konstants for mean
    c1 = stats.t.ppf((1-gamma)/2, N-1)
    c2 = stats.t.ppf((1+gamma)/2, N-1)
    mu_min = m - ((c2*s)/np.sqrt(N))
    mu_max = m - ((c1*s)/np.sqrt(N))

    if verbose:
        if std==None:
            print("Calculating confidence range with unknown variance:")
        else:
            print("Calculating confidence range with known variance:")

        print("c1: ",c1,"   c2: ",c2, "   mu_min: ",mu_min, "   mu_max: ",mu_max)

    if std==None:
        c1 = stats.chi2.ppf((1-gamma)/2, N-1)
        c2 = stats.chi2.ppf((1+gamma)/2, N-1)
        sdev_min = np.sqrt((N-1)/c2)*s
        sdev_max = np.sqrt((N-1)/c1)*s

        if verbose:
            print("c1: ",c1,"   c2: ",c2, "   sdev_min: ",sdev_min, "   sdev_max: ",sdev_max)

        return mu_min, mu_max, sdev_min, sdev_max
    
    
    return mu_min, mu_max


#hypothesentest mult data
    #levene test for std
    #ttest_ind for mean

    #return("prob_arr [[correct decision mu0, error type2],[error type1, correct decision mu1]]")


#TODO: Tolerierung alle varianten  (Faltung,Monte Carlo, arithmetischer, Grenzwertsatz etc.)  +Vergleich empfindlichkeiten
#TODO: Tolerierung with correlation


#TODO: Hypothesentest   one sample difference in samples
#TODO: Data to vert??      shapiro for norm
#TODO: Gütefunktion  bigger smaller both

#TODO: Correlated Variables
#TODO: Measurement system analysis
#TODO: ? Plots    Histogramm
#TODO: Normal Distribution Test
#TODO: C_g-  and C_gk- ?

#TODO: use _ on internal functions

#TODO: list all important distributions: norm uniform t chi chi2 weißbull etc

    
