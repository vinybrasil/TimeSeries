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

'''
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

#Média móvel: o valor y em t depende dos valores dos ultimos n t's:

def moving_average(series, n):
    return np.average(series[-n:])
    
print(moving_average(ads, 24))

print("salve")