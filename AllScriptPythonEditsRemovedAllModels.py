import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import shapely


plt.rcParams['font.size'] = 14

sns.set_style("ticks")
sns.set_context("paper")


from subprocess import check_output
#print(check_output(["ls", ".."]).decode("utf8"))




df_airbnb = pd.DataFrame({})
df_airbnb = pd.read_csv('../input/Airbnb_Milan.csv')

df_cdrs = pd.DataFrame({})
for i in range(1,8):
    df = pd.read_csv('../input/sms-call-internet-mi-2013-11-0{}.csv'.format(i), parse_dates=['datetime'])
    df_cdrs = df_cdrs.append(df)
    
df_cdrs=df_cdrs.fillna(0)
df_cdrs['sms'] = df_cdrs['smsin'] + df_cdrs['smsout']
df_cdrs['calls'] = df_cdrs['callin'] + df_cdrs['callout']
df_cdrs.head()

df_cdrs_internet = df_cdrs[['datetime', 'CellID', 'internet', 'calls', 'sms']].groupby(['datetime', 'CellID'], as_index=False).sum()
df_cdrs_internet['hour'] = df_cdrs_internet.datetime.dt.hour+24*(df_cdrs_internet.datetime.dt.day-1)
df_cdrs_internet = df_cdrs_internet.set_index(['hour']).sort_index()

df_airbnb.to_pickle('./completedInternet.pkl')
df_airbnb.to_csv('./completedInternet.csv')


import geojson
import matplotlib.colors as colors
import matplotlib.cm as cmx
#import matplotlib as mpl
from descartes import PolygonPatch





num = int(10000+1)
arr_cellID = np.zeros(num)
arr_mean = np.zeros(num)
for i in range(1,num):
    ydata = df_cdrs_internet[df_cdrs_internet.CellID==i]['internet']
    xdata = df_cdrs_internet[df_cdrs_internet.CellID==i]['internet'].index
    mean = np.mean(ydata)
    arr_cellID[i]=i
    arr_mean[i]=mean

arr_mean[arr_mean<=0] = 1 #replacing 0's with 1's for log calc
arr_mean_log = np.log(arr_mean)

#https://gis.stackexchange.com/questions/93136/how-to-plot-geo-data-using-matplotlib-python
with open("../input/milano-grid.geojson") as json_file:
    json_data = geojson.load(json_file)
    

fig = plt.figure() 
ax = fig.gca() 

coordlist = json_data.features[1]['geometry']['coordinates'][0]

jet = cm = plt.get_cmap('RdYlGn_r')
#cNorm  = colors.Normalize(vmin=0, vmax=np.max(arr_mean))
cNorm  = colors.Normalize(vmin=0, vmax=5000)
#cNorm  = colors.Normalize(vmin=0, vmax=np.max(arr_mean_log))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
#print(scalarMap.get_clim())

for i in range(1,10000):
    poly = json_data.features[i]['geometry']
    colorVal = scalarMap.to_rgba(arr_mean[i])
    ax.add_patch(PolygonPatch(poly, fc=colorVal, ec=colorVal, alpha=1, zorder=1 ))
ax.axis('scaled')

fig.set_size_inches(11,11)
plt.title("Grid Internet Connections (Green = 0 connections, Red ~ 5000 connections)")
plt.show()
print(coordlist)




#PortaRomana (downtown) data
ydata = df_cdrs_internet[df_cdrs_internet.CellID==6755]['internet']
xdata = df_cdrs_internet[df_cdrs_internet.CellID==6755]['internet'].index

f = plt.figure()
plt.plot(xdata, ydata, color='black', linewidth=1, linestyle='-', label='San Vittore - data')
plt.title("Internet Connections - San Vittore")
plt.xlabel("Time [hour]")
plt.ylabel("Internet Connections [#]")
plt.xlim([0,168])
plt.ylim([0,2000])
plt.legend()
sns.despine()
plt.show()




# Autocorellation (ACF) calc
from statsmodels.tsa.stattools import acf, pacf
acf_y = acf(ydata, nlags=48)

#Plot ACF: 
f = plt.figure()
plt.plot(acf_y, color='black', linewidth=1, linestyle='-', label='data')
plt.axhline(y=0,linestyle='--',color='gray')
plt.title('Autocorrelation Function')
plt.xlabel("Lag [hour]")
plt.ylabel("Autocorrelation [-]")
plt.xlim([0,48])
plt.ylim([-1,1])
plt.legend()
sns.despine()
plt.show()



import scipy
def func(xdata, a,b,c):
    return a*np.sin(2*np.pi*(1/24)*xdata+b)+c
    
    
    
    
    
    
popt,pcov = scipy.optimize.curve_fit(func, xdata, ydata)

print(popt)

f = plt.figure()
yfit = func(xdata, *popt)
#residual
residual = ydata - yfit
rss = np.sum(residual**2)
mean = np.mean(ydata)
b = popt[1] #phase shift (b) from curve_fit
#print('rss_navigli',rss_navigli,'mean_navigli',mean_navigli,'rss-norm',rss_navigli/mean_navigli)
#stddev = np.std(residual_navigli)
#print(np.std(residual_navigli))

yfit_duomo = yfit
b_duomo = b
ydata_duomo = ydata
xdata_duomo = xdata
T_peak_duomo = 18-(b_duomo%(2*np.pi))*(24/(2*np.pi))

ydata_moving_avg = ydata.rolling(window=24,center=False).mean()
residual_moving_avg = residual.rolling(window=24,center=False).mean()






f = plt.figure()
plt.plot(xdata, ydata, color='black', linewidth=1, linestyle='--', label='data')
plt.plot(xdata, yfit, color='black', linewidth=3, label='model')
plt.xlabel("Time [hour]")
plt.ylabel("Internet Connections [#]")
plt.xlim([0,168])
plt.ylim([0,2000])
plt.legend()
sns.despine()
plt.show()




ff = plt.figure()
plt.plot(xdata, residual, color='black', linewidth=1, linestyle='-', label='residual')
plt.axhline(y=0,linestyle='--',color='gray')
plt.title("Residual")
plt.xlabel("Time [hour]")
plt.ylabel("Internet Connections [#]")
plt.xlim([0,168])
plt.ylim([-2000,2000])
plt.legend()
sns.despine()
plt.show()





#Plot ACF: 
from statsmodels.tsa.stattools import acf, pacf
acf_r = acf(residual, nlags=48)

f = plt.figure()

plt.plot(acf_y, color='black', linewidth=1, linestyle='-', label='data')
plt.plot(acf_r, color='black', linewidth=1, linestyle='--', label='residual')

plt.axhline(y=0,linestyle='--',color='gray')
plt.title('Autocorrelation Function')
plt.xlabel("Lag [hour]")
plt.ylabel("Autocorrelation [-]")
plt.xlim([0,48])
plt.ylim([-1,1])
plt.legend()
sns.despine()
plt.show()






from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import zivot_andrews
def test_stationarity(timeseries,maxlag_input=None):
    #https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC', maxlag=maxlag_input)
    #print(dftest)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
    #Perform Kwiatkowski-Phillips-Schmidt-Shin test:
    print('Results of Kwiatkowski-Phillips-Schmidt-Shin test:')
    dftest2 = kpss(timeseries, regression='ct', nlags='auto')
    dfoutput2 = pd.Series(dftest2[0:3], index=['Test Statistic','p-value','#Trunc Lag Parameter'])
    for key,value in dftest2[3].items():
        dfoutput2['Critical Value (%s)'%key] = value
    print(dfoutput2)
    
    #Perform Zivot Andrews test:
    print('Results of Zivot Andrews test:')
    dftest3 = zivot_andrews(timeseries, trim=0.28, maxlag=maxlag_input,  regression='ct', autolag=None)
    dfoutput3 = pd.Series(dftest2[0:2], index=['Test Statistic','p-value'])
    for key,value in dftest3[2].items():
        dfoutput3['Critical Value (%s)'%key] = value
    print(dfoutput3)



test_stationarity(residual,1)




##AppliedAcrossAll - REMOVED

    

# Navigli
ydata = df_cdrs_internet[df_cdrs_internet.CellID==4456]['internet']
xdata = df_cdrs_internet[df_cdrs_internet.CellID==4456]['internet'].index


popt,pcov = scipy.optimize.curve_fit(func, xdata, ydata)

print(popt)

f = plt.figure()
yfit = func(xdata, *popt)
#residual
residual = ydata - yfit
rss = np.sum(residual**2)
mean = np.mean(ydata)
a = popt[0] #amplitude
b = popt[1] #phase shift (b) from curve_fit
c = popt[2] #mean

yfit_navigli = yfit
b_navigli = b
ydata_navigli = ydata
xdata_navigli = xdata

if (a>0) :
    T_peak = 6+24-(b%(2*np.pi))*(24/(2*np.pi))
elif (a<=0) :
    T_peak = 18-(b%(2*np.pi))*(24/(2*np.pi))
T_peak = T_peak%24 

a_norm = np.abs(a/c)

print (a_norm)

T_peak_navigli = T_peak    
print(T_peak_duomo)
print(T_peak_navigli)

###Plots


f = plt.figure()
plt.plot(xdata_duomo, ydata_duomo, color='black', linewidth=1, linestyle='--', label='Duomo - data')
plt.plot(xdata_duomo, yfit_duomo, color='black', linewidth=3, label='Duomo - model')
plt.plot(xdata_navigli, ydata_navigli, color='green', linewidth=1, linestyle='--', label='Navigli - data')
plt.plot(xdata_navigli, yfit_navigli, color='green', linewidth=3, label='Navigli - model')

plt.xlabel("Time [hour]")
plt.ylabel("Internet Connections [#]")
plt.xlim([0,168])
plt.legend()
sns.despine()






f = plt.figure()
#plt.plot(xdata_duomo, ydata_duomo, color='black', linewidth=1, linestyle='--', label='Duomo - data')
plt.plot(xdata_duomo, yfit_duomo, color='black', linewidth=3, label='Duomo - model')
#plt.plot(xdata_navigli, ydata_navigli, color='green', linewidth=1, linestyle='--', label='Navigli - data')
plt.plot(xdata_navigli, yfit_navigli, color='green', linewidth=3, label='Navigli - model')
plt.axvline(x=T_peak_duomo, color='red')
plt.axvline(x=T_peak_navigli, color='red')

plt.xlabel("Time [hour]")
plt.ylabel("Internet Connections [#]")
plt.xlim([0,24])
plt.ylim([0,10000])
plt.legend()
sns.despine()



f = plt.figure()
#plt.plot(xdata_duomo, ydata_duomo, color='black', linewidth=1, linestyle='--', label='Duomo - data')
plt.plot(xdata_duomo, yfit_duomo, color='black', linewidth=3, label='Duomo - model')
#plt.plot(xdata_navigli, ydata_navigli, color='green', linewidth=1, linestyle='--', label='Navigli - data')
plt.plot(xdata_navigli, yfit_navigli, color='green', linewidth=3, label='Navigli - model')

plt.xlabel("Time [hour]")
plt.ylabel("Internet Connections [#]")
plt.xlim([0,168])
plt.ylim([0,10000])
plt.legend()
sns.despine()



#Bocconi
ydata = df_cdrs_internet[df_cdrs_internet.CellID==4259]['internet']
xdata = df_cdrs_internet[df_cdrs_internet.CellID==4259]['internet'].index

popt,pcov = scipy.optimize.curve_fit(func, xdata, ydata)
print(popt)

yfit = func(xdata, *popt)
residual = ydata - yfit






f = plt.figure()
plt.plot(xdata, ydata, color='blue', linewidth=1, linestyle='--', label='Bocconi - data')
plt.plot(xdata, yfit, color='blue', linewidth=3, label='Bocconi - model')

plt.xlabel("Time [hour]")
plt.ylabel("Internet Connections [#]")
plt.xlim([0,168])
plt.ylim([0,10000])
plt.legend()
sns.despine()

f = plt.figure()
plt.plot(xdata, residual, color='blue', linewidth=1, linestyle='-', label='Bocconi - residual')
plt.axhline(y=0,linestyle='--',color='gray')
plt.title("Residual - Bocconi")
plt.xlabel("Time [hour]")
plt.ylabel("Residual Internet Connections [#]")
plt.xlim([0,168])
plt.ylim([-10000,10000])
plt.legend()
sns.despine()

plt.show()


test_stationarity(residual,1)