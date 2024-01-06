

import numpy as np
import pandas as pd
from  statsmodels.formula.api import ols
from scipy.io import loadmat
import scipy.stats as stats
import matplotlib.pyplot as plt

import re

import TSST as TT


GAMMA = 0.95

# Data according to problem
x_0 = 50
x_tol = 0.01
x_sig = x_tol/np.sqrt(12)
y_0 = 30
y_tol = 0.01
y_sig = y_tol/np.sqrt(12)

# Resolution of distance
DZ = 0.00001



probab={"x": stats.uniform(loc=x_0-(x_tol/2),scale=x_tol),
        "y": stats.uniform(loc=y_0-(y_tol/2),scale=y_tol)}

conv, x =TT.conv(probab,"(x**2 + y**2)**0.5",res=DZ)




ax1 = plt.figure(1, figsize=(6, 4)).subplots(1, 1)

ax1.plot(x, conv)



##############NEW FUNC

# Determin cumulative density function
F12 = np.cumsum(conv)

# Berechnung der Toleranzgrenzen über Ausfallwahrscheinlichkeiten
indexmin = np.min(np.where(F12 >= (1-GAMMA)/2))
indexmax = np.min(np.where(F12 >= (1+GAMMA)/2))
z_maxCon = x[indexmax]
z_minCon = x[indexmin]
z_tolerance_con = z_maxCon - z_minCon

print(z_tolerance_con)


exit()

data = loadmat('Feuchtesensor')
regress = pd.DataFrame({'a': np.reshape(data['Cp'], -1),
                        'b': np.reshape(data['RF'], -1),
                        'c': np.reshape(data['T'], -1)})


print(regress.min())
print(regress.max())


df_scal, scaler = TT.normer(regress,'a',min_val={'b': 0},max_val={'b': 100})

print(df_scal)

print(scaler.rev(df_scal))

exit()

result = TT.Mult_M_regression(df_scal, 'a')#, extend_terms=["np.log(b)","2**b"])

print(result.summary())


#print(TT.replace_simple("single2**2+single3","a"))

liste = [1]

liste.extend([1,2,3])
#liste.extend(1)
liste.extend([])

print(liste)


print(TT.keyword_match("single**2+single1", "single"))

result = TT.Mult_M_regression(df_scal, 'a',M=4)#, extend_terms=["np.log(b)","2**b"])

print(result.summary())

#result =TT.formula_regression(df_scal, "a", ["inter2**single1","np.log(full2+5)","c**5*a","b"])

something_stupid=result =TT.formula_regression(df_scal, "a", ["full3**full3"])

print(result.summary())

print(range(0))

exit()


tmp = np.linspace(20, 50, 100)

volt = 30*tmp + 0.0214*tmp**3 + np.random.normal(-100,100) +25

df = pd.DataFrame(np.array([volt,tmp]).T, columns=['volt', 'tmp'])



result2 = TT.Mult_M_regression(df,'volt',M=3)

print(result2.summary())

TT.word_multiplicator

exit()


print(TT.word_multiplicator(["A","B","C"],M=5))



tmp = np.linspace(20, 50, 100)

volt = 30*tmp + 0.0214*tmp**3 + np.random.normal(-100,100) +25

df = pd.DataFrame(np.array([volt,tmp]).T, columns=['volt', 'tmp'])



result = TT.M_regression(df["tmp"],df["volt"],3)

print(result.summary())


# Sample DataFrame
data = {'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': [100, 200, 300, 400, 500]}
df = pd.DataFrame(data)

# Row to exclude for standardization
col_to_exclude = 'A'

# Standardize the DataFrame excluding one row
data_to_standardize = df.drop(columns=col_to_exclude)
df_scal, scaler = TT.stder(df,'A')

# Additional data to standardize
new_data = pd.DataFrame({'A': [6], 'B': [60], 'C': [600]})

# Apply the same standardization to additional data
standardized_new_data = scaler.std(new_data)


# Additional data to reverse
new_data = pd.DataFrame({'A': [7], 'B': [2.5], 'C': [2.5]})

# Apply the same standardization to additional data
reversed_new_data = scaler.rev(new_data)

# Display results
print("Original DataFrame:")
print(df)
print("\nStandardized DataFrame:")
print(df_scal)
print("\nStandardized Additional Data:")
print(standardized_new_data)
print("\nReverse Additional Data:")
print(reversed_new_data)