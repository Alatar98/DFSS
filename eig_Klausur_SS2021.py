%matplotlib inline

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

ax1.hist(SF_data.values.reshape(-1), bins=10, density=True,# weights=weights,
         label='Histogramm')
#ax1.axis([np.min(SF_data), np.min(SF_data), 0, 10])
ax1.set_xlabel(r'Systematischer Messfehler')
ax1.set_ylabel(r'Relative Häufigkeit')


x=np.linspace(SF_data.min(), SF_data.max(),100)
ax1.plot(x,stats.norm.pdf(x,loc=SF_data.mean(),scale=SF_data.std(ddof=1)))

print("#################### c) ###################################")

gamma=0.95
mu_min, mu_max, sdev_min, sdev_max = TT.confidenceRange(SF_data,gamma)

print(f"conf range mean: {mu_min}-{mu_max} \n conf range std: {sdev_min}-{sdev_max}")



print("########################### d) #############################")


T=0.02

C_g = 0.2*T/6/SF_data.std(ddof=1)   #>1.33
C_gk = (0.1*T-abs(SF_data.mean()-1))/3/SF_data.std(ddof=1) #>1.33


print(f"C_g: {C_g}     C_gk: {C_gk}")


print("################################  e) ##########################")


SMF_data = pd.read_csv('Linearitaet.csv', sep=',')


SMF_data = SMF_data.rename(columns={"0": "1.2", "1": "1.4", "2": "1.6", "3": "1.8", "4": "2"})



ax1 = plt.figure(2, figsize=(6, 4)).subplots(1, 1)

ax1.plot(np.repeat([1.2,1.4,1.6,1.8,2],10),SMF_data.T.values.reshape(-1),marker='o')
#ax1.axis([np.min(SF_data), np.min(SF_data), 0, 10])
ax1.set_xlabel(r'Referenzwert')
ax1.set_ylabel(r'Messwert')

print("################################  f) ##########################")


STF_data = pd.read_csv('Streuverhalten.csv', sep=',')


ax1 = plt.figure(3, figsize=(6, 4)).subplots(1, 1)

ax1.boxplot(STF_data)
#for col in STF_data.columns:
    #ax1.scatter(int(col), np.mean(STF_data[col]), label=f'mean {col}')

    #ax1.errorbar(int(col), np.mean(STF_data[col]), yerr=np.std(STF_data[col],ddof=1), label=f'column {col}', capsize=5)

#ax1.axis([np.min(SF_data), np.min(SF_data), 0, 10])
ax1.set_xlabel(r'Messwert')
ax1.set_ylabel(r'Messung')

ax1 = plt.figure(4, figsize=(6, 4)).subplots(1, 1)


STF_data_comb =pd.DataFrame()

STF_data_comb['0']=STF_data['0'].append(STF_data['1'])
STF_data_comb['1']=STF_data['2'].append(STF_data['3'])
STF_data_comb['2']=STF_data['4'].append(STF_data['5'])


ax1.boxplot(STF_data_comb)
#for col in STF_data.columns:
    #ax1.scatter(int(col), np.mean(STF_data[col]), label=f'mean {col}')

    #ax1.errorbar(int(col), np.mean(STF_data[col]), yerr=np.std(STF_data[col],ddof=1), label=f'column {col}', capsize=5)

#ax1.axis([np.min(SF_data), np.min(SF_data), 0, 10])
ax1.set_xlabel(r'Messwert')
ax1.set_ylabel(r'Messung')


print("Einzelne Messreihen:")
print("Means: ",STF_data.mean(axis=0))
print("Std: ",STF_data.std(axis=0))

print("Zusammengefasste Messreihen:")
print("Means: ",STF_data_comb.mean(axis=0))
print("Std: ",STF_data_comb.std(axis=0))


#GRR Value!!!!!


print("################################  ...) ##########################")
print("################################  m) ##########################")


FS_data = pd.read_csv('FormaldehydSchaetzung.csv', sep=';')

scaled_DF, stder =TT.stder(FS_data,'F')


model = TT.Mult_M_regression(scaled_DF,'F',M=1)

print(model.summary())

print("################################  n) ##########################")

FS_data_relevant=scaled_DF[["NOx","O2","CO","F"]]

model2 = TT.Mult_M_regression(FS_data_relevant,'F',M=2)

print(model2.summary())

print("################################  n) ##########################")

to_predict=stder.scl(pd.DataFrame({"NOx": [0.426], "CO": [0.372], "O2": [5.5]}))
print("prediction: ",model2.predict(to_predict))


#there is a function for this


#other function
# model.conf_int

#prob = {"NOx": stats.norm(loc=0.426,scale=scaled_DF["NOx"].std()),
#        "CO": stats.norm(loc=0.372,scale=scaled_DF["CO"].std())}
#pdf_conv = TT.conv(prob,"26.4042 + 31.3236*CO - 12.8606*CO**2 + 13.6842*CO*NOx",verbose=True,res=0.0001)



