# -*- coding: utf-8 -*-

""" Musterlösung zur Klausur Design For Six Sigma WS 2022/23

Update on jan 17 2023
@author: stma0003
"""

from scipy import stats
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import sympy as syms
from scipy.stats import norm
from scipy.stats import t
from scipy.stats import chi2
from scipy.stats import uniform


def conf_pred_band_ex(regress_ex, poly, model, alpha=0.05):
    """ Function calculates the confidence and prediction interval for a
    given multivariate regression function poly according to lecture DFSS,
    regression parameters are already determined in an existing model,
    identical polynom is used for extrapolation

    Parameters
    ----------
    regress_ex : DataFrame
        Extended dataset for calculation.
    poly : OLS object of statsmodels.regression.linear_model modul
        definition of regression model.
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        Model parameters.
    alpha : float, optional
        Significance level. The default is 0.05.

    Returns
    -------
    lconf_ex : Series
        Distance of confidence limit to mean regression.
    lpred_ex : Series
        Distance of prediction limit to mean regression.
    """
    # ols is used to calculte the complets vector x0 of input variables
    poly_ex = ols(poly.formula, regress_ex)
    x0 = poly_ex.exog
    # Calculation according lecture book
    d = np.dot(x0, np.dot(poly.normalized_cov_params, x0.T))
    c1 = stats.t.isf(alpha/2, model.df_resid)
    lconf_ex = c1*np.sqrt(np.diag(d)*model.mse_resid)
    lpred_ex = c1*np.sqrt((1+np.diag(d))*model.mse_resid)

    return lconf_ex, lpred_ex


# # Generate data for part a
# MU = 5.47
# SIG = 1.38
# data = np.random.normal(MU, SIG, 50)
# pd.DataFrame(data).to_csv('Rautiefe.csv', index=False)


""" Aufgabenteil a: Grafische Darstellung der Messwerte als Histogramm """

# Import and format data
rz_data = pd.read_csv('Rautiefe.csv', sep=',')
rz_plot = np.arange(0, 10.1, 0.1)
ax1 = plt.figure(1, figsize=(6, 4)).subplots(1, 1)
ax1.hist(rz_data.values.reshape(-1), bins=11, range=(0, 10), density=True,
         label='Histogramm')
ax1.axis([1, 10, 0, 0.4])
ax1.set_xlabel(r'Rautiefe Rz / $\mu$m')
ax1.set_ylabel(r'Relative Häufigkeit $h(Rz)$')


""" Aufgabenteil b: Konfidenzbereiche für Mittelwert und Standardabweichung"""

GAMMA = 0.95
N = rz_data.shape[0]
c1 = t.ppf((1-GAMMA)/2, N-1)
c2 = t.ppf((1+GAMMA)/2, N-1)
mean = float(rz_data.mean(axis=0))
s = float(rz_data.std(axis=0, ddof=1))
muc1 = float(round(mean - c2*s/np.sqrt(N), 5))
muc2 = float(round(mean - c1*s/np.sqrt(N), 5))
c1 = chi2.ppf((1-GAMMA)/2, N-1)
c2 = chi2.ppf((1+GAMMA)/2, N-1)
sig = round(s, 5)
sigc1 = round(s*np.sqrt((N-1)/c2), 5)
sigc2 = round(s*np.sqrt((N-1)/c1), 5)
print(' ')
print('Konfidenzbereiche Stichprobe')
print('Mittelwert : ', muc1, '<=', mean, '<=', muc2)
print('Standardabweichung  : ', sigc1, '<=', sig, '<=', sigc2)


""" Aufgabenteil c: Darstellung geschätzten Wahrscheinlichkeitsdichte """

ax1.plot(rz_plot,
         norm.pdf(rz_plot, rz_data.mean(axis=0),
                  rz_data.std(axis=0, ddof=1)),
         'C1', label='Wahrscheinlichkeitsdichte')
ax1.grid(False)
ax1.legend(ncol=1)


""" Aufgabenteil d: Hypothesentest mu < 6 um """

ALPHA = 0.05
RZ0 = 6
# t-test due to unknown variance
hypo_test = stats.ttest_1samp(rz_data, popmean=RZ0, alternative='greater')
print("")
print("Hypothesentest auf Abweichung mit p-value = ",
      round(float(hypo_test[1]), 4))
if hypo_test[1] <= ALPHA:
    print("Hypothese wird verworfen")
else:
    print("Hypothese wird nicht verworfen")


""" Aufgabenteil e: Bestimmung der Gütefunktion """

# Determine acceptance limit with alternative greater than nominal value
c = t.ppf((1-ALPHA), N-1)
rz_c = RZ0 + c*s/np.sqrt(N)
print("")
print("Grenze Annahmebereich = ", round(rz_c, 3))

# Determine quality function for vector of alternatives
rz_alt = np.arange(5, 7.01, 0.01)
quality = 1 - t.cdf((rz_c-rz_alt)/s*np.sqrt(N), N-1)
quality_0 = 1 - t.cdf((rz_c-7)/s*np.sqrt(N), N-1)
print("")
print("Abweichung von Rz = 7 wird mit der Sicherheit Q =",
      round(quality_0, 3), 'erkannt.')

# Plot result
ax1 = plt.figure(2, figsize=(6, 4)).subplots(1, 1)
ax1.plot(rz_alt, quality, 'C0')
ax1.set_xlabel(r'Rautiefe Rz / $\mu$m')
ax1.set_ylabel(r'Gütefunktion')
ax1.grid()


""" Aufgabenteil f: Erstellen Regressionsmodells mit Wechselswirkungen """

# Read and format data
data = pd.read_csv('UntersuchungLaserschneiden.csv', sep=';')
data = data.drop(['n'], axis=1)

# Standardize data
data_norm = (data - data.mean())/data.std()
# data_norm['h'] = data['h']

# Generate liner regression model with interactions
poly1 = ols("h ~ v + g + d + l + P\
             + I(v*g) + I(v*d) + I(v*l) + I(v*P)\
             + I(g*d) + I(g*l) + I(g*P)\
             + I(d*l) + I(d*P)\
             + I(l*P)", data_norm)
model1 = poly1.fit()

# Predict height for standardized operation point
test_1 = pd.DataFrame({'v': [4], 'g': [13],  'd': [1.1],
                       'l': [0.7], 'P': [1.5]})
test_1_norm = (test_1 - data.mean())/data.std()
test_1_norm['h'] = float(model1.predict(test_1_norm))
test_1['h'] = test_1_norm['h']*data.std()['h'] + data.mean()['h']
print("")
print("Grathöhe für vollständiges Regressionsmodell h =",
      round(float(test_1['h']), 3))


""" Aufgabenteil g: Reduktion der Regressionsfunktion """

# Define and fit regression model 
poly2 = ols("h ~ g + d + l\
            + I(v*g) + I(v*d) + I(v*l) + I(v*P)\
            + I(g*d) + I(g*l) + I(g*P)\
            + I(d*l) + I(d*P)\
            + I(l*P) -1", data_norm)           
model2 = poly2.fit()
print("")
print(model2.summary())

test_2 = pd.DataFrame({'v': [4], 'g': [13],  'd': [1.1],
                       'l': [0.7], 'P': [1.5]})
test_2_norm = (test_2 - data.mean())/data.std()
test_2_norm['h'] = float(model2.predict(test_2_norm))
test_2['h'] = test_2_norm['h']*data.std()['h'] + data.mean()['h']
print("")
print("Grathöhe für vollständiges Regressionsmodell h =",
      round(float(test_2['h']), 3))


""" Aufgabenteil h: Prognosebereich """

# Determine confidence bound with internal function
dh_norm_conf, dh_norm_pred = conf_pred_band_ex(test_2_norm, poly2, model2,
                                               alpha=0.05)
h_min = (test_2_norm['h'] - dh_norm_conf)*data.std()['h'] + data.mean()['h']
h_max = (test_2_norm['h'] + dh_norm_conf)*data.std()['h'] + data.mean()['h']
print("")
print("Minimum Konfidenzbereich h =", round(float(h_min), 3))
print("Maximum Konfidenzbereich h =", round(float(h_max), 3))


""" Aufgabenteil i: Bergründung keine Addition von Varianzen """

print("")
print("Werden zwei gleichverteilte Größen addiert ergibt sich eine trapez-")
print("förmige Wahrscheinlichkeitsdichte, die stark von der Normalverteilung")
print("abweicht. Damit ist eine Voraussetzung für die Anwendung verletzt.")


""" Aufgabenteil j: Statistische Tolerierung über Faltung """

# Data according to problem
x_0 = 50
x_tol = 0.01
x_sig = x_tol/np.sqrt(12)
y_0 = 30
y_tol = 0.01
y_sig = y_tol/np.sqrt(12)

# Definition of symbolic variables and function
x_sym, y_sym = syms.symbols('x_sym, y_sym')
z_sym = syms.sqrt(x_sym**2 + y_sym**2)

# Symbolic calculation of sensitivities
e_x_sym = z_sym.diff(x_sym)
e_y_sym = z_sym.diff(y_sym)

# Substitute symbols by values, numeric calculation of sensitivities
values = {x_sym: x_0, y_sym: y_0}
e_x = float(e_x_sym.evalf(subs=values))
e_y = float(e_y_sym.evalf(subs=values))


# Resolution of distance
DZ = 0.00001

# Propability density functions
z_x_min = - 5*x_sig*np.abs(e_x)
z_x_max = + 5*x_sig*np.abs(e_x)
z_x = np.arange(z_x_min, z_x_max+DZ, DZ)
f_x = uniform.pdf(z_x, x_tol/2*np.abs(e_x), x_tol*np.abs(e_x))

# Propability density functions
z_y_min = - 5*y_sig*np.abs(e_y)
z_y_max = + 5*y_sig*np.abs(e_y)
z_y = np.arange(z_y_min, z_y_max+DZ, DZ)
f_y = uniform.pdf(z_y, y_tol/2*np.abs(e_y), y_tol*np.abs(e_y))

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
print('Toleranzbereich bei Faltung =', round(z_tolerance_con, 5))


""" Aufgabenteil k: Statistische Tolerierung über Monte Carlo Simulation """

# Generation of random numbers according to specified distribution
N = 100000
x_sim = np.random.uniform(x_0 - x_tol/2, x_0 + x_tol/2, N)
y_sim = np.random.uniform(y_0 - y_tol/2, y_0 + y_tol/2, N)

# Calculation of distance z
z_sim = np.sqrt(x_sim**2 + y_sim**2)

# Comparison to numerical evaluation of simulated data
z_sort = np.sort(z_sim)
z_cdf = np.arange(1, N+1, 1)/N
index_min = np.min(np.where(z_cdf >= (1-GAMMA)/2))
index_max = np.min(np.where(z_cdf >= (1+GAMMA)/2))
z_tolerance_num = z_sort[index_max] - z_sort[index_min]
print("Toleranz bei numerischer Simulation :", round(z_tolerance_num, 5))


""" Aufgabenteil l: Vergleich mit arithmetischer Tolerierung """

z_ari_max = np.sqrt((x_0 + x_tol/2)**2 + (y_0 + y_tol/2)**2)
z_ari_min = np.sqrt((x_0 - x_tol/2)**2 + (y_0 - y_tol/2)**2)
z_tolerance_ari = z_ari_max - z_ari_min
print("Toleranz bei statistischer Simulation :", round(z_tolerance_ari, 5))
print(' ')
print('Statistische Tolerierung führt zu kleinerem Toleranzbereich, die ')
print('beiden Verfahren zur statistischen Tolerierung führen zu')
print('vergleichbaren Ergebnissen.')


""" Aufgabenteil m: Vergleich der Empfindichkeiten"""

comp = pd.DataFrame({'x': x_sim,
                     'y': y_sim,
                     'z': z_sim})
poly = ols('z ~ x + y', comp)
model = poly.fit()
print(' ')
print('Bestimmung der Empfindlichleiten über die analytische Rechnung:')
print('E_x_ana =', round(e_x, 4))
print('E_y_ana =', round(e_y, 4))
print('Bestimmung der Empfindlichleiten über die tatistische Simulation:')
print('E_x_sim =', round(model.params[1], 4))
print('E_y_sim =', round(model.params[2], 4))
print('Rechenwege führen zu vergleichbaren Ergebnissen.')
