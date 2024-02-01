import numpy as np
import scipy.stats as stats
from  statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import pandas as pd

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