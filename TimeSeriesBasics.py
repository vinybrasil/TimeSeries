#parte 1 de https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-9-time-series-analysis-in-python-a270cb05e0b3 

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import seaborn as sns

from dateutil.relativedelta import relativedelta #datas melhores
from scipy.optimize import minimize

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as smf
import scipy.stats as scs
from itertools import product
from tqdm import tqdm_notebook #progress bar
import warnings #modo não disturbe
warnings.filterwarnings('ignore')
#%matplotlib inline

ads = pd.read_csv('data/ads.csv', index_col=['Time'], parse_dates=['Time'])
currency = pd.read_csv('data/currency.csv', index_col=['Time'], parse_dates=['Time'])

'''
Plotando:


plt.figure(figsize=(12,6))
plt.plot(ads.Ads)
plt.title("Ads watched(hourly data)")
plt.grid(True)
plt.show()

plt.figure(figsize=(12,6)) #12,5.6 é bom, 15,7 é o default
plt.plot(currency.GEMS_GEMS_SPENT)
plt.title("In-game currency spent(daily data)")
plt.grid(True)
plt.show()

===============================================================

Medidas de qualidade da forecast:

R squared(R^2): (-inf,1]
Mean Absolute Error [0,+inf)
Median Absolute Error [0,+inf)
Mean Squared Error [0,+inf)
Mean Squared Logarithmic Error [0,+inf)
Mean Absolute Percentage Error [0,+inf)

'''

#importando as forecast quality metrics:

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

def mean_absolute_percentage_error(y_true, y_pred): #pois nao ta implemetado no sklearn
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#Média móvel: o valor y em t depende dos valores dos ultimos n t's. Boa para detectar trends:
print(ads['Ads'].head())
print(ads['Ads'].tail())
print(ads['Ads'][-4:])
def moving_average(series, n):
    return np.average(series[-n:])
    
print(moving_average(ads, 24))

#O pandas tem uma função boa pra detectar trends tbm (smoothing of the original time series to indicate trends):

def plotMovingAverage(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):

    rolling_mean = series.rolling(window=window).mean()
    
    plt.figure(figsize=(13.4,6.2))
    plt.title("Moving average\n window size ={}".format(window))
    plt.plot(rolling_mean, "g", label="Rolling mean trend") #g de green
    
    #intervalo de confiança:
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper bond/Lower bond")
        plt.plot(lower_bond, "r--")
        
    #achar valores anormais:
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series<lower_bond] = series[series<lower_bond]
            anomalies[series>upper_bond] = series[series>upper_bond]
            plt.plot(anomalies, "ro", markersize=10)
            
    plt.plot(series[window:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)

'''
Plotando:
    
plotMovingAverage(ads, 4) #4 horas
plt.show()
plotMovingAverage(ads, 12) #12 horas
plt.show()
plotMovingAverage(ads, 4, plot_intervals=True) #nao funciona bem para valores altos do window
plt.show()
'''

#criando uma anomalia(fica um ponto vermelho nela):
'''

ads_anomaly = ads.copy()
ads_anomaly.iloc[-20] = ads_anomaly.iloc[-20] * 0.2 #reduzindo ele em 80%, deixando o valor com 20% do original 
plotMovingAverage(ads_anomaly, 4, plot_intervals=True, plot_anomalies=True)
plt.show()

'''
#analisando a currency com as médias móveis, há inúmeras anomalias, causadas pela presença de sazonalidade nos dados. Por isso, será
#necessário a utilização de modelos mais complexos.

'''
plotMovingAverage(currency, 7, plot_intervals=True, plot_anomalies=True)
plt.show()
'''

#media móvel ponderada: sendo n o len do vetor de pesos, a média pondera o valor no tempo t com os valores de t-1, t-2,...,t-n e dps soma tudo, sendo 
#esse o valor no t-ésimo período

def weighted_average(series, weights):
    result = 0.0
    weights.reverse()
    for n in range(len(weights)):
        result += series.iloc[-n-1] * weights[n]
    return float(result)
    
print("Média ponderada nos últimos três perídos: %f" %  weighted_average(ads, [0.6, 0.3, 0.1])) #os pesos somam 1 e os y mais recentes tem mais peso

#exponential smoothing: usa toda a available data, dando exponencialmente menos pesos para os valores mais longes de t

def exponential_smoothing(series, alpha):
    result = [series[0]]
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result

def plotExponentialSmoothing(series, alphas):
    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(13.4,6.2))
        for alpha in alphas: 
            plt.plot(exponential_smoothing(series, alpha), label="Alpha {}".format(alpha))
        plt.plot(series.values, "c", label="Actual")
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Exponential Smoothing")
        plt.grid(True)



plotExponentialSmoothing(ads.Ads, [0.3, 0.05])
plotExponentialSmoothing(currency.GEMS_GEMS_SPENT, [0.3,0.05])
#plt.show()

#double exponential smoothing: decompondo a série, temos dois componentes: o intercepto e a trend(que foi descoberto ali em cima).

def double_exponential_smoothing(series, alpha, beta):
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series):
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    return result

def plotDoubleExponentialSmoothing(series, alphas, betas):
    
        


print("salve")