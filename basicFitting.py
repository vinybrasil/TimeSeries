#simple linear and exponential fitting
#http://emilygraceripka.com/blog/14

from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import scipy as sc
from scipy import optimize 
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
import matplotlib.ticker as ticker

#======Linear fitting==============

#criando fake linear data e criando um ruido

x_array = np.linspace(1,10,10) #10 valores, de 1 a 10
y_array = np.linspace(5,200,10)
y_noise = 30*(np.random.ranf(10))
y_array += y_noise

#plotando do jeito bonito

fig = plt.figure(figsize=(4,3))
gs = gridspec.GridSpec(1,1)
ax1 = fig.add_subplot(gs[0])
ax1.plot(x_array, y_array, "ro")
ax1.set_xlim(0,11)
ax1.set_ylim(-10,250)
ax1.set_xlabel("x_array", family="arial", fontsize=12)
ax1.set_ylabel("y_array", family="arial", fontsize=12)

ax1.xaxis.set_major_locator(ticker.MultipleLocator(2))  #aqueles trequinhos grandes na reta
ax1.yaxis.set_major_locator(ticker.MultipleLocator(50))

ax1.xaxis.set_minor_locator(AutoMinorLocator(2))    #aqueles trequinhos pequenos na reta
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

ax1.tick_params(axis="both",which="major", direction="out", top="on", right="on", bottom="on", length=8, labelsize=8)
ax1.tick_params(axis="both",which="minor", direction="out", top="on", right="on", bottom="on", length=5, labelsize=8)

fig.tight_layout()
fig.savefig("rawLinear.png", format="png", dpi=1000)
plt.title("Linear fitting")
plt.grid(True)
plt.show()

#fitting a line:

def linear(x, m, b):
    return m*x + b
    
#popt_linear: fitting parameters
#pcov_linear: estimated covariance of the fitting parameters
#p0: initial guess of the parameters

#popt_linear, pcov_linear = sc.optimize.curve_fit(linear, x_array, y_array, p0=[((75-25)/(44-2)), 0])
popt_linear, pcov_linear = sc.optimize.curve_fit(linear, x_array, y_array)
perr_linear = np.sqrt(np.diag(pcov_linear)) #tira a diagonal da matriz de correlação, dps tira a sqrt para achar o erro

print("slope = %0.2f (+/-) %0.2f" % (popt_linear[0], perr_linear[0]))
print("y-intercept = %0.2f (+/-) %0.2f" % (popt_linear[1], perr_linear[1]))

print(pcov_linear)
print("\n", popt_linear)

#plotando de novo, mas com a linha: 

fig = plt.figure(figsize=(4,3))
gs = gridspec.GridSpec(1,1)
ax1 = fig.add_subplot(gs[0])
ax1.plot(x_array, y_array, "ro")
ax1.plot(x_array, linear(x_array, *popt_linear), 'k--', label="y= %0.2fx + %0.2f" % (popt_linear[0], popt_linear[1]))
ax1.set_xlim(0,11)
ax1.set_ylim(-10,250)

ax1.legend(loc="best")
ax1.set_xlabel("x_array", family="arial", fontsize=12)
ax1.set_ylabel("y_array", family="arial", fontsize=12)

ax1.xaxis.set_major_locator(ticker.MultipleLocator(2))  #aqueles trequinhos grandes na reta
ax1.yaxis.set_major_locator(ticker.MultipleLocator(50))

ax1.xaxis.set_minor_locator(AutoMinorLocator(2))    #aqueles trequinhos pequenos na reta
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

ax1.tick_params(axis="both",which="major", direction="out", top="on", right="on", bottom="on", length=8, labelsize=8)
ax1.tick_params(axis="both",which="minor", direction="out", top="on", right="on", bottom="on", length=5, labelsize=8)

fig.tight_layout()
fig.savefig("fittedLinear.png", format="png", dpi=1000)
plt.title("Linear fitting")
plt.grid(True)
#grid(color='r', linestyle='-', linewidth=2)

plt.show()

#===========single exponential fitting==============

#tipo y = e^(kx)
#o x vai ser o mesmo

y_array_exp = np.exp(x_array*0.7)
#noise:
y_noise_exp = (np.exp((np.random.ranf(10))))/30
y_array_exp += y_noise_exp

#plotando: 

fig = plt.figure(figsize=(4,3))
gs = gridspec.GridSpec(1,1)
ax1 = fig.add_subplot(gs[0])
ax1.plot(x_array, y_array_exp, "ro")
ax1.set_xlim(0,11)
ax1.set_ylim(-0.1,0.7)
ax1.set_xlabel("x_array", family="arial", fontsize=12)
ax1.set_ylabel("y_array", family="arial", fontsize=12)

ax1.xaxis.set_major_locator(ticker.MultipleLocator(2))  #aqueles trequinhos grandes na reta
ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

ax1.xaxis.set_minor_locator(AutoMinorLocator(2))    #aqueles trequinhos pequenos na reta
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

ax1.tick_params(axis="both",which="major", direction="out", top="on", right="on", bottom="on", length=8, labelsize=8)
ax1.tick_params(axis="both",which="minor", direction="out", top="on", right="on", bottom="on", length=5, labelsize=8)

fig.tight_layout()
fig.savefig("rawExponential.png", format="png", dpi=1000)
plt.title("Exponential Error")
plt.grid(True)
plt.show()















