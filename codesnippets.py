import numpy as np
import scipy.stats as stats
from  statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

import TSST as TT



def conf_pred_band_ex(_regress_ex, _model, _fit, alpha=0.05):
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

    # ols is used to calculte the complets vector x_0 of input variables
    poly_ex = ols(_model.formula, _regress_ex)
    x_0 = poly_ex.exog
    # Calculation according lecture book
    d = np.dot(x_0, np.dot(_model.normalized_cov_params, x_0.T))
    c_1 = stats.t.isf(alpha/2, _fit.df_resid)
    lconf_ex = c_1*np.sqrt(np.diag(d)*_fit.mse_resid)
    lpred_ex = c_1*np.sqrt((1+np.diag(d))*_fit.mse_resid)

    return lconf_ex, lpred_ex

#how to use:
measurement_plot = pd.DataFrame({"input": np.arange(min, max, ds)})
measurement_plot["result"] = fit.predict(measurement_plot)
measurement_plot["confidence"], measurement_plot["prediction"] = conf_pred_band_ex(measurement_plot, model, fit, alpha=alpha)

ax.plot(measurement["input"], measurement["result"], 'bo',label='Messwerte')
ax.plot(measurement_plot["input"], measurement_plot["result"], 'r',label='Regressionsfunktion')
ax.plot(measurement_plot["ki"],measurement_plot["tr"]+measurement_plot["confidence"], 'r:',label='Konfidenzbereich')
ax.plot(measurement_plot["ki"],measurement_plot["tr"]-measurement_plot["confidence"], 'r:')
ax.plot(measurement_plot["ki"],measurement_plot["tr"]+measurement_plot["prediction"], 'g--',label='Prognosebereich')
ax.plot(measurement_plot["ki"],measurement_plot["tr"]-measurement_plot["prediction"], 'g--')

'confidence range'
TT.confidenceRange()


'confidence range comparison'
TT.confidenceRangeComp()


'Histogramm und Wahrscheinlichkeitsdichte'
TT.HistDist()


'correlation with confidence range'
TT.Correlation()


'plot correlation'
ax.plot(data1, data2, 'b+')


'conf iterval of model'
fit.conf_int(alpha=alpha)


'tolerancing (all)'
TT.tolerancing()


'hypothesis test form norm dist ALPHA=1-GAMMA= Signifikanznivau  Zu_testender_Punkt=Gamma0 STD=STD'
print(stats.norm.interval(ALPHA,loc=GAMMA0,scale=SIG))

'Gütefunktion für Hypothesentest'
TT.hypothesis_t_test()


'cg  cgk'
REFERENCE = 1
Y_TOLERANCE = 0.1
DATA=[0.9,1.1,1.2,1.0,1.3,1.0,0.8,1.1,1.3]
y_deviation = np.mean(DATA) - REFERENCE
c_g = 0.1*Y_TOLERANCE/3/np.std(DATA, ddof=1)
if c_g >= 1.33:
    print("Wiederholbarkeit ausreichend")
else:
    print("Wiederholbarkeit ist nicht ausreichend")
c_gk = (0.1*Y_TOLERANCE - np.abs(y_deviation))/3/np.std(DATA, ddof=1)
if c_gk >= 1.33:
    print("Wiederholbarkeit und sytematische Abweichung ausreichend")
elif c_g >= 1.33:
    print("Systematische Abweichung zu groß")
else:
    print("Auflösung und systematische Abweichung nicht ausreichend")

'Hypothesistest with H0: y_repeat_test = Y_REPEAT_REFERENCE'
hypo_test = stats.ttest_1samp(DATA, REFERENCE)
print("")
print("Hypothesentest auf Abweichung mit p-value = ",
      round(float(hypo_test[1]), 4))
if hypo_test[1] <= 0.05:
    print("Abweichung signifikant")
else:
    print("Abweichung nicht signifikant")


'Streuverhalten   annovatable'
Y_K = 25   # number of Parts
Y_N = 2    # number of measurements
model = ols('Value ~ C(Part)', data=y_variation_3).fit()
anova1 = sm.stats.anova_lm(model, typ=2)
anova1["M"] = anova1["sum_sq"]/anova1["df"]

# estimations of variance and calculation of GRR and ndc
equipment_variation = np.sqrt(anova1.loc["Residual", "M"])
part_variation = np.sqrt((anova1.loc["C(Part)", "M"] - anova1.loc["Residual", "M"])/Y_N)
grr = equipment_variation
grr_relative = 6*grr/Y_TOLERANCE
ndc = 1.41*part_variation/grr
print("Relativer GRR-Wert %GRR = ", round(grr_relative*100, 3), "%")
print("Number of Distict Categories ndc = ", round(ndc, 3))
# Visualization
y_variation_3_multi= y_variation_3.set_index(['Measurement', 'Part'])
fig4 = plt.figure(4, figsize=(12, 4))
fig4.suptitle('')
ax1, ax2 = fig4.subplots(1, 2)
ax1.plot(np.arange(1, Y_K+1, 1), y_variation_3_multi.loc[1, :],'b', label='Messung 1')
ax1.plot(np.arange(1, Y_K+1, 1), y_variation_3_multi.loc[2, :],'r:', label='Messung 2')
ax1.axis([0, 26, 0.9, 1.1])
ax1.set_xlabel('Stichprobe')
ax1.set_ylabel(r'Leitfähigkeit $\gamma$ / $\mu$S')
ax1.set_title('Streuverhalten ohne Temperaturkompensation')
ax1.grid(True)
ax1.legend(loc=9, ncol=3)



'hypothese test for same mean'
# Load and format data
data = loadmat('Signifikanz')
m_hyp = pd.DataFrame({'m1': data["M1"].reshape(-1),
                      'm2': data["M2"].reshape(-1),
                      'n1': data["n1"].reshape(-1),
                      'n2': data["n2"].reshape(-1)})

# t-test due to unknown variance, difference of means
GAMMA = 0.95
N = m_hyp.m1.shape[0]
c1 = stats.t.ppf((1 - GAMMA)/2, 2*N - 2)
c2 = stats.t.ppf((1 + GAMMA)/2, 2*N - 2)
m_hyp_diff = np.mean(m_hyp.m1 - m_hyp.m2)
m_hyp_s1 = np.std(m_hyp.m1, ddof=1)
m_hyp_s2 = np.std(m_hyp.m2, ddof=1)
m_hyp_s = np.sqrt((m_hyp_s1**2 + m_hyp_s2**2) / 2)
m_hyp_c1 = c1*np.sqrt(2/N)*m_hyp_s
m_hyp_c2 = c2*np.sqrt(2/N)*m_hyp_s
print("")
print("Abweichung der Stichprobe =", round(m_hyp_diff, 4))
print("Unterer Annahmegrenze =", round(m_hyp_c1, 4))
print("Obere Annahmegrenze =", round(m_hyp_c2, 4))
if ((m_hyp_c1 <= m_hyp_diff) & (m_hyp_diff <= m_hyp_c2)):
    print("Abweichung nicht signifikant")
else:
    print("Abweichung signifikant")
