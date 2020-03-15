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
#plt.show()

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

#plt.show()

#===========single exponential fitting==============

#tipo y = e^(kx)
#o x vai ser o mesmo

y_array_exp = np.exp(-x_array*0.7) #se colocar mais, o valor explode
print(y_array_exp) 
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
plt.title("Single exponential random data")
plt.grid(True)
plt.show()

#fittando a curva: 

def exponential(x, a, k, b):
    return a*np.exp(x*k) + b

#popt_exponential, pcov_exponential = scipy.optimize.curve_fit(exponential, x_array, y_array_exp, p0=[1,-0.5, 1])
popt_exponential, pcov_exponential = sc.optimize.curve_fit(exponential, x_array, y_array_exp, p0=[1,-0.5, 1]) #sem o p0 nao funcionou da primeira vez
#erro:
perr_exponential = np.sqrt(np.diag(pcov_exponential))

print(pcov_exponential)
print("\n", popt_exponential)

print("Pre-exponential factor = %0.2f (+/-) %0.2f" % (popt_exponential[0], perr_exponential[0]))
print("Rate constant = %0.2f (+/-) %0.2f" % (popt_exponential[1], perr_exponential[1]))

#plotando:

fig = plt.figure(figsize=(4,3))
gs = gridspec.GridSpec(1,1)
ax1 = fig.add_subplot(gs[0])
ax1.plot(x_array, y_array_exp, "ro")
ax1.plot(x_array, exponential(x_array, *popt_exponential), 'k--', label="%0.2f$e^{%0.2fx}$ + %0.2f" % (popt_exponential[0], popt_exponential[1], popt_exponential[2]))
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
fig.savefig("fittedExponential.png", format="png", dpi=1000)
plt.title("Single exponential fitted line")
plt.grid(True)
#plt.show()

#===========Bi-exponential fitting================

#quando se tem algo do tipo y = e^(ax) + e^(bx)

y_array_2exp = (np.exp(-x_array*0.1) + np.exp(-x_array*1))
y_noise_2exp = (np.exp((np.random.ranf(10))))/30
#noise
y_array_2exp += y_noise_exp

#plotando:
fig = plt.figure(figsize=(4,3))
gs = gridspec.GridSpec(1,1)
ax1 = fig.add_subplot(gs[0])
ax1.plot(x_array, y_array_2exp, "ro")
ax1.set_xlim(0,11)
ax1.set_ylim(0.2,1.5)
ax1.set_xlabel("x_array", family="arial", fontsize=12)
ax1.set_ylabel("y_array", family="arial", fontsize=12)

ax1.xaxis.set_major_locator(ticker.MultipleLocator(2))  
ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

ax1.xaxis.set_minor_locator(AutoMinorLocator(2))    
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

ax1.tick_params(axis="both",which="major", direction="out", top="on", right="on", bottom="on", length=8, labelsize=8)
ax1.tick_params(axis="both",which="minor", direction="out", top="on", right="on", bottom="on", length=5, labelsize=8)

fig.tight_layout()
fig.savefig("rawBiExponential.png", format="png", dpi=1000)
plt.title("Double exponential data")
plt.grid(True)
plt.show()

def _2exponential(x, a, k1, b, k2, c):
    return a*np.exp(x*k1) + b*np.exp(x*k2) + c
    
popt_2exponential, pcov_2exponential = sc.optimize.curve_fit(_2exponential, x_array, y_array_2exp, p0=[1, -0.1, 1, -1, 1])
perr_2exponential = np.sqrt(np.diag(pcov_2exponential))    
print(*popt_2exponential)


#plotando:
fig = plt.figure(figsize=(4,3))
gs = gridspec.GridSpec(1,1)
ax1 = fig.add_subplot(gs[0])
ax1.plot(x_array, y_array_2exp, "ro")
ax1.plot(x_array, _2exponential(x_array, *popt_2exponential), 'k--', label="y= %0.2f$e^{%0.2fx}$ + \n %0.2f$e^{%0.2fx}$ + %0.2f" % (popt_2exponential[0], \
popt_2exponential[1], popt_2exponential[2], \
popt_2exponential[3], popt_2exponential[4])) 
ax1.set_xlim(0,11)
ax1.set_ylim(0.2,1.5)
ax1.set_xlabel("x_array", family="arial", fontsize=12)
ax1.set_ylabel("y_array", family="arial", fontsize=12)

ax1.xaxis.set_major_locator(ticker.MultipleLocator(2))  
ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

ax1.legend(loc="best")

ax1.xaxis.set_minor_locator(AutoMinorLocator(2))    
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

ax1.tick_params(axis="both",which="major", direction="out", top="on", right="on", bottom="on", length=8, labelsize=8)
ax1.tick_params(axis="both",which="minor", direction="out", top="on", right="on", bottom="on", length=5, labelsize=8)

fig.tight_layout()
fig.savefig("fitBiExponential.png", format="png", dpi=1000)
plt.title("Fitted BiExponential line")
plt.grid(True)
plt.show()






