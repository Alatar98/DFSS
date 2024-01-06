#%matplotlib inline
import numpy as np
import pandas as pd
from  statsmodels.formula.api import ols

import TSST as TT


import numpy as np
import scipy.stats as stats
from sympy import symbols, sympify

import sympy as syms

import matplotlib.pyplot as plt


GAMMA = 0.95

# Data according to problem
x_0 = 50
x_tol = 0.01
x_sig = x_tol/np.sqrt(12)
y_0 = 30
y_tol = 0.01
y_sig = y_tol/np.sqrt(12)

RHO_0 = 7.850e-3
RHO_SIG = 0.05e-3/6

# Definition of symbolic variables and function
x_sym, y_sym = syms.symbols('x_sym, y_sym')
#z_sym = syms.sqrt(x_sym**2 + y_sym**2)
z_sym = (x_sym**2 + y_sym**2)**0.5

# Symbolic calculation of sensitivities
e_x_sym = z_sym.diff(x_sym)
e_y_sym = z_sym.diff(y_sym)


# Substitute symbols by values, numeric calculation of sensitivities
values = {x_sym: x_0, y_sym: y_0}
e_x = float(e_x_sym.evalf(subs=values))
e_y = float(e_y_sym.evalf(subs=values))

print(e_x,e_y)

print("Toleranzbereich bei central limit?: ",np.sqrt(e_x**2*x_sig**2+e_y**2*y_sig**2))

# Resolution of distance
DZ = 0.00001

# Propability density functions
z_x_min = - 5*x_sig*np.abs(e_x)
z_x_max = + 5*x_sig*np.abs(e_x)
z_x = np.arange(z_x_min, z_x_max+DZ, DZ)
f_x = stats.uniform.pdf(z_x, -x_tol/2*np.abs(e_x), x_tol*np.abs(e_x))


# Propability density functions
z_y_min = - 5*y_sig*np.abs(e_y)
z_y_max = + 5*y_sig*np.abs(e_y)
z_y = np.arange(z_y_min, z_y_max+DZ, DZ)
f_y = stats.uniform.pdf(z_y, -y_tol/2*np.abs(e_y), y_tol*np.abs(e_y))

#h_H_min = - 5*H_SIG*np.abs(e_H)
#h_H_max = 5*H_SIG*np.abs(e_H)
#h_H = np.arange(h_H_min, h_H_max+DH, DH)
#f_H = norm.pdf(h_H, 0, np.abs(e_H*H_SIG))

# Convolute propability density functions
f12 = np.convolve(f_x, f_y)*DZ
z12_min = z_x_min + z_y_min
z12_max = z_x_max + z_y_max
z12 = np.arange(z12_min, z12_max+DZ, DZ)

# Determin cumulative density function
F12 = np.cumsum(f12)*DZ
F12 = F12/np.max(F12)

# Berechnung der Toleranzgrenzen über Ausfallwahrscheinlichkeiten
indexmin = np.min(np.where(F12 >= (1-GAMMA)/2))
indexmax = np.min(np.where(F12 >= (1+GAMMA)/2))
z_maxCon = z12[indexmax]
z_minCon = z12[indexmin]
z_tolerance_con = z_maxCon - z_minCon
print(' ')
print('Toleranzbereich bei Faltung =', z_tolerance_con)



exit()


from sympy import Symbol 

x=Symbol('x') 
y=Symbol('y') 
expr=x**2+y**2
print(expr)


from sympy import symbols, sympify

def get_sympy_expression(formula_str, variable_names):
    # Create Symbol objects from variable names
    symbols_list = symbols(variable_names)

    # Build the expression using the formula string
    expression = sympify(formula_str)

    # Replace any variable names in the expression with the corresponding symbols
    for var_name, var_symbol in zip(variable_names, symbols_list):
        expression = expression.subs(var_name, var_symbol)

    return expression

# Example usage:
formula = "a*x**2 + b*x + c"
variables = ['a', 'b', 'x']

result = get_sympy_expression(formula, variables)
print(result)




exit()

data = {'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': [100, 200, 300, 400, 500]}


def calculate_mean_and_std(df):
    # Calculate mean and standard deviation for each column
    mean_values = df.mean()
    std_values = df.std()

    # Create a new DataFrame with mean and std for each column
    result_df = pd.DataFrame({
        'Mean': mean_values,
        'Std': std_values
    })

    return result_df.transpose()

# Example usage:
df = pd.DataFrame(data)

result = calculate_mean_and_std(df)
print(result)


exit()

df = pd.DataFrame(data)


M=2

#wanted output  A ~ B + C + A**2 + B**2 + A*B

from itertools import permutations, product


# Example usage:
words = ['wordA', 'wordB', 'wordC']
#print([permutation for permutation in permutations(words,r=2)])
#print([permutation for permutation in product(words)])


words_copy=words.copy()

for word in words:
    print(word)
    words_copy2=words_copy.copy()

    for word2 in words_copy:
        print(word+"*"+word2)

        for word3 in words_copy2:
            print(word+"*"+word2+"*"+word3)

        words_copy2.remove(word2)

    words_copy.remove(word)

print("\n")

def word_iterator(string,words,M,comb):
    print(string)
    comb.append(string)
    words_copy=words.copy()
    for word in words:
        if M==1:
            print(string+"*"+word)
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

comb = word_multiplicator(words,M=3)

print(comb)
#result = generate_word_combinations(words)
#print(result)
