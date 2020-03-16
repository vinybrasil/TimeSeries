#COD3D BY L1Z4RB
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import scipy as sc
from scipy import optimize 
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
import matplotlib.ticker as ticker

#vale lembrar que em 13/03 houve uma mudança na metodologia, ou seja, fodeu o modelo, mas né, é o que temos
def exponential(x, a, k, b):
    return a*np.exp(x*k) + b
        
def plotar(row):
    popt_exponential, pcov_exponential = sc.optimize.curve_fit(exponential, row[0], row[1], p0=[1, 1, 0])
    perr_exponential = np.sqrt(np.diag(pcov_exponential))
    print("Pre-exponential factor = %0.2f (+/-) %0.2f" % (popt_exponential[0], perr_exponential[0]))
    print("Rate constant = %0.2f (+/-) %0.2f" % (popt_exponential[1], perr_exponential[1]))
    fig = plt.figure(figsize=(4,3))
    gs = gridspec.GridSpec(1,1)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(row[0], row[1], "ro")
    ax1.plot(row[0], exponential(row[0], *popt_exponential), 'k--', label="y = %0.2f$e^{%0.2fx}$ %0.2f" % (popt_exponential[0], popt_exponential[1], popt_exponential[2]))
    ax1.set_xlim(0,20)
    ax1.set_ylim(0,250)
    ax1.set_xlabel("Dias desde a primeira infecção", family="arial", fontsize=12)
    ax1.set_ylabel("Casos de COVID até 14/03", family="arial", fontsize=12)
    ax1.legend(loc="best")
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(2))  #aqueles trequinhos grandes na reta
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(50))

    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))    #aqueles trequinhos pequenos na reta
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax1.tick_params(axis="both",which="major", direction="out", top="on", right="on", bottom="on", length=8, labelsize=8)
    ax1.tick_params(axis="both",which="minor", direction="out", top="on", right="on", bottom="on", length=5, labelsize=8)

    fig.tight_layout()
    fig.savefig("COVID1403brasil.png", format="png", dpi=1000)
    plt.title("COVID 19 em 14/03 no Brasil")
    plt.grid(True)
    plt.show()
    
    
def main():
    coronga = pd.read_csv('./data/corona.csv')
    
    br = np.delete(np.array(coronga.loc[0].array), 0,0)
    print(br) 
    dias = np.arange(1, len(br) + 1 , dtype=int)
    br = br.astype(int)
    data = np.array([dias, br])
    if len(data[0]) == len(data[1]):
        print("deu boa dms")
    
    plotar(data)
if __name__ == "__main__":
    main()
    

    
    
