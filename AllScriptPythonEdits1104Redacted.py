import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import shapely

import scipy

import geojson
import matplotlib.colors as colors
import matplotlib.cm as cmx
plt.rcParams.update({'figure.max_open_warning': 0})
import matplotlib as mpl
from descartes import PolygonPatch
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import zivot_andrews

plt.rcParams['font.size'] = 14

sns.set_style("ticks")
sns.set_context("paper")


from subprocess import check_output
#print(check_output(["ls", ".."]).decode("utf8"))




df_airbnb = pd.read_pickle("./completedAirbnb.pkl")
df_cdrs_internet = pd.read_pickle("./completedInternetSet.pkl")


#df_yDataFrame = pd.DataFrame({})


#num = int(10000+1)
#arr_cellID = np.zeros(num)
#arr_mean = np.zeros(num)
#for i in range(1,num):
#    ydata = df_cdrs_internet[df_cdrs_internet.CellID==i]['internet']
#    xdata = df_cdrs_internet[df_cdrs_internet.CellID==i]['internet'].index
#    mean = np.mean(ydata)
#    arr_cellID[i]=i
#    arr_mean[i]=mean
    #df_yDataFrame[i] = df_cdrs_internet[df_cdrs_internet.CellID==i]['internet']

#arr_mean[arr_mean<=0] = 1 #replacing 0's with 1's for log calc
#arr_mean_log = np.log(arr_mean)

#https://gis.stackexchange.com/questions/93136/how-to-plot-geo-data-using-matplotlib-python
#with open("../input/milano-grid.geojson") as json_file:
#    json_data = geojson.load(json_file)
    
#coordlist = json_data.features[1]['geometry']['coordinates'][0]


#centro/duomo (downtown) data
#y1data = df_yDataFrame[5061]
#y2data = df_yDataFrame[5061]

#df1['score1'].sub(df1['score2']

#dfAverage = pd.concat((ydata, y2data))
#dfAverage = pd.DataFrame({})
#dfAverage = y1data['internet'].mean(y2data['internet'], axis = 0)


ydata = df_cdrs_internet[df_cdrs_internet.CellID==5060]['internet']
xdata = df_cdrs_internet[df_cdrs_internet.CellID==5060]['internet'].index

f = plt.figure()
plt.plot(xdata, ydata, color='#fff2cc', linewidth=1, linestyle='-', label='Data Usage')
plt.title("Internet Connections in the Historical City Centre")
plt.xlabel("Time of Week (in 24h intervals)")
plt.ylabel("Number of Connections")
plt.xlim([0,168])
plt.ylim([0,10000])
plt.legend()
sns.despine()
#plt.show()

dataframe_collectionIndexed = pd.read_pickle("./dataframe_collectionIndexed")
#df_neighbourhood = dataframe_collectionIndexed[1]
#neighbourhoodCell = df_neighbourhood.loc[5]
lengthofDict = len(dataframe_collectionIndexed)
#print(dataframe_collectionIndexed[3])
#print(dataframe_collectionIndexed[3].head())


#for key in dataframe_collectionIndexed.items():
#    print(key)
    
#for j in range(lengthofDict)

#ydata = df_cdrs_internet[df_cdrs_internet.CellID==5060]['internet']
#y2data = df_cdrs_internet[df_cdrs_internet.CellID==5061]['internet']
#newAve = pd.concat([ydata, y2data]).groupby(level=0).mean()
#y2data = df_cdrs_internet[df_cdrs_internet.CellID==5061]['internet'][164]

myNum = 1
myString = ''
arrdata = []
df_storeDataPerCellRegion = {}
df_storeDataforSideBySidePlots = {}
myfloat = 1

storeDataPerCellRegion = pd.DataFrame({})
#df_storeDataPerCellRegion['means']
#df_storeDataPerCellRegion['meansIndex']




storedWeeklyVals =  []
storedName = []
storedPeak = []
storedIndex = []
#df_neighbourhoodCells = pd.DataFrame({})

limitNum = lengthofDict 
for j in range(limitNum):
    df_neighbourhood = dataframe_collectionIndexed[j]
    arrdata = []
    for k in range(len(df_neighbourhood)):
        df_neighbourhoodCell = df_neighbourhood.loc[k]
        myString = str(df_neighbourhood.loc[k])   
        myfloat = df_neighbourhoodCell.median().round().astype(int)
        ydata = df_cdrs_internet[df_cdrs_internet.CellID==myfloat]['internet']
        xdata = df_cdrs_internet[df_cdrs_internet.CellID==myfloat]['internet'].index
        arrdata.append(ydata)
    #print(arrdata) gives array of week per cell in neighbourhood
    ydataMean = pd.concat(arrdata).groupby(level=0).mean()  
    #print(ydataMean) gives Series of week that has mean for all cells in neighbourhood
    #print(df_storeDataPerCellRegion)
    myNewString = myString.split(' ', 1)[0].title()
    #df_storeDataPerCellRegionTemp =  pd.DataFrame({myNewString: ydataMean})
    #df_storeDataPerCellRegion[myNewString] = pd.DataFrame({j: [ydataMean, xdata]})
    #df_airbnb.at[j, 'means'] = ydataMean
    #df_airbnb.at[j, 'meansIndex'] = xdata
    
    
    storedWeeklyVals.append(ydataMean)
    storedName.append(myNewString)
    
    #CODE FOR DRAWING THINGS PER NEIGHBOURHOOD

    ##PLOT FOR INTERNET USAGE
        #Changes
    typeofPlot = 'Data_Usage'
    
    
        #StaysConstant
    plotName =  myNewString     
    f = plt.figure()
    colourChoice = '#1f4e78'
    plt.plot(xdata, ydataMean, color=colourChoice, linewidth=2, linestyle='--', label=typeofPlot)    
    
        #Stays constant
    plt.title(plotName)    
    plt.xlabel("Time of day in hours")
    
        #Changes
    plt.ylabel("Number of internet connections")    
    plt.xlim([0,168])
    
        #StaysConstant
    plt.legend()
    sns.despine()
    fileSavePath = './output/' + myNewString + '_' + typeofPlot + '.png'
    plt.savefig(fileSavePath)
    
    
    ##PLOT FOR ACF Calc
    typeofPlot = 'ACF_Calc'
    from statsmodels.tsa.stattools import acf, pacf
    acf_y = acf(ydataMean, nlags=48)

    #Plot ACF: 
    f = plt.figure()
    plt.plot(acf_y, color=colourChoice, linewidth=2, linestyle='--', label=typeofPlot)  
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.title('ACF on ' + plotName)
    plt.xlabel("Lag in hours")
    plt.ylabel("ACF Outcome")
    plt.xlim([0,48])
    #plt.ylim([-1,1])
    plt.legend()
    sns.despine()
    fileSavePath = './output/' + myNewString + '_' + typeofPlot + '.png'
 
    
    
    ###PLOT FOR SCIPY CURVE FIT
    def func(xdata, a,b,c):
        return a*np.sin(2*np.pi*(1/24)*xdata+b)+c
        
    popt,pcov = scipy.optimize.curve_fit(func, xdata, ydataMean)

    #print(popt)

    f = plt.figure()
    yfit = func(xdata, *popt)
    #residual
    residual = ydataMean - yfit
    rss = np.sum(residual**2)
    mean = np.mean(ydataMean)    
    rss_norm = rss/mean
    a = popt[0] #amplitude
    b = popt[1] #phase shift (b) from curve_fit
    c = popt[2] #mean

    
    #phase shift (b) from curve_fit
    #print('rss_navigli',rss_navigli,'mean_navigli',mean_navigli,'rss-norm',rss_navigli/mean_navigli)
    #stddev = np.std(residual_navigli)
    #print(np.std(residual_navigli))

    yfit_this = yfit
    b_this = b
    ydata_this = ydataMean
    xdata_this = xdata
    
    
    if (a>0) :
        T_peak = 6+24-(b%(2*np.pi))*(168/(2*np.pi))
    elif (a<=0) :
        T_peak = 18-(b%(2*np.pi))*(168/(2*np.pi))
        T_peak_this = 18-(b_this%(2*np.pi))*(24/(2*np.pi))
    T_peak = T_peak%168 

    a_norm = np.abs(a/c)
    
    #print(T_peak)
    storedPeak.append(T_peak)
    #print(T_peak.type())
    minval = min(ydataMean)    
    maxval = max(ydataMean)
    outprintVals = myNewString + '\n' + ' Peak Hour: ' + str(T_peak) + '\n'  + ' Min: ' + str(minval) + '\n'+ ' Max: ' + str(maxval)
    print(outprintVals, file=open("./minMax.txt", "a"))
    
    #storedWeeklyVals.append(yfit_this)

    ydataMean_moving_avg = ydataMean.rolling(window=24,center=False).mean()
    residual_moving_avg = residual.rolling(window=24,center=False).mean()

    
    ###Plot Fit
    #Changes
    typeofPlot = 'Fit_Model'
    
    
        #StaysConstant
    plotName =  myNewString     
    f = plt.figure()
    colourChoice = '#1f4e78'
    plt.plot(xdata, ydataMean, color=colourChoice, linewidth=2, linestyle='--', label=typeofPlot)  
    plt.plot(xdata, yfit, color='black', linewidth=3, label='model')    
    
    
    
    
        #Stays constant
    plt.title(plotName)    
    plt.xlabel("Time of day in hours")
    
        #Stays Constant
    plt.ylabel("Number of internet connections")    
    plt.xlim([0,168])
    
        #StaysConstant
    plt.legend()
    sns.despine()
    fileSavePath = './output/' + myNewString + '_' + typeofPlot + '.png'
    plt.savefig(fileSavePath)
    
    
    
    ###Plot Residuals
    #Changes
    typeofPlot = 'Residuals'
    
    
        #StaysConstant
    plotName =  myNewString     
    f = plt.figure()
    colourChoice = '#1f4e78'
    plt.plot(xdata, residual, color=colourChoice, linewidth=1, linestyle='-', label=typeofPlot)  
    
    
    
    
        #Stays constant
    plt.title(plotName)    
    plt.xlabel("Time of day in hours")
    
        #Stays Constant
    plt.ylabel("Number of internet connections residuals")    
    plt.xlim([0,168])
    plt.ylim([-10000,10000])
    
        #StaysConstant
    plt.legend()
    sns.despine()
    fileSavePath = './output/' + myNewString + '_' + typeofPlot + '.png'
    plt.savefig(fileSavePath)
    
    

    ###RESULTS FOR STATIONARY TESTS
    

    def test_stationarity(timeseries,maxlag_input=None):
        f = plt.figure()
        typeofPlot = 'Rolling_Mean'
        fileSavePath = './deviation/' + myNewString + '_' + typeofPlot + '_Deviation'+'.png'
        #https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
        #Determing rolling statistics
        rolmean = timeseries.rolling(12).mean()
        rolstd = timeseries.rolling(12).std()
        #Plot rolling statistics:
        orig = plt.plot(timeseries, color='blue',label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label = 'Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation') 
        plt.savefig(fileSavePath)
        #plt.show(block=False)
        
        
        
        #Perform Dickey-Fuller test:
        print('Results of Dickey-Fuller Test: '+myNewString, file=open("./stattests.txt", "a"))
        dftest = adfuller(timeseries, autolag='AIC', maxlag=maxlag_input)
        #print(dftest)
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        print(dfoutput, file=open("./stattests.txt", "a"))
        
        
        
        #Perform Kwiatkowski-Phillips-Schmidt-Shin test:
        print('Results of Kwiatkowski-Phillips-Schmidt-Shin test: '+myNewString, file=open("./stattests.txt", "a"))
        dftest2 = kpss(timeseries, regression='ct', nlags='auto')
        dfoutput2 = pd.Series(dftest2[0:3], index=['Test Statistic','p-value','#Trunc Lag Parameter'])
        for key,value in dftest2[3].items():
            dfoutput2['Critical Value (%s)'%key] = value
        print(dfoutput2, file=open("./stattests.txt", "a"))
    
        #Perform Zivot Andrews test:
        print('Results of Zivot Andrews test: '+myNewString, file=open("./stattests.txt", "a"))
        dftest3 = zivot_andrews(timeseries, trim=0.28, maxlag=maxlag_input,  regression='ct', autolag=None)
        dfoutput3 = pd.Series(dftest2[0:2], index=['Test Statistic','p-value'])
        for key,value in dftest3[2].items():
            dfoutput3['Critical Value (%s)'%key] = value
        print(dfoutput3, file=open("./stattests.txt", "a"))
        
    test_stationarity(residual,1)
    
    
    
# CODE FOR CREATING NEW DATAFRAME FOR SIDE BY SIDE PLOTS
    #Neighbourhoods_dfTemp =  pd.DataFrame()
    
    #calchere = ydataMean.values
    #print(calchere)
    #storedWeeklyVals[j] = ydataMean.values
    #storedName[j] = storedName.append(myNewString)
    #storedPeak[j] = storedPeak.append(T_peak)
    #storedIndex[j] = ydataMean.index
    
    #print(storedName, storedPeak)
    #print(storedWeeklyVals, storedName, storedPeak, storedIndex)
    #Neighbourhoods_df = Neighbourhoods_df.append(T_peak)


#print(ydataMean)


#for i in range(1,5):
#    ydata = df_cdrs_internet[df_cdrs_internet.CellID==i]['internet']
#    xdata = df_cdrs_internet[df_cdrs_internet.CellID==i]['internet'].index
#    print(ydata)
#    print(xdata)

#print(Neighbourhoods_df)
#print(Neighbourhoods_df.get(myNewString))

#print(df_storeDataPerCellRegion.get(myNewString))

#print(xdata)
#print(storedWeeklyVals, storedName, storedPeak)
#print(storedWeeklyVals[2])

#neigh1


f = plt.figure()

ydata1 = storedWeeklyVals[4]
xpeak1 = storedPeak[4]
xname1 = storedName[4]
#print(xname1)

plt.plot(xdata, ydata1, color='orange', linewidth=2, linestyle='--', label=xname1)
plt.axvline(x=xpeak1, color='orange')

#print(len(yfit_this))

#neigh2

ydata2 = storedWeeklyVals[5]
xpeak2 = storedPeak[5]
xname2 = storedName[5]


plt.plot(xdata, ydata2, color='blue', linewidth=2, linestyle='--', label=xname2)
plt.axvline(x=xpeak2, color='blue')

#print(yfit_this, yfit_this, xname1, xpeak1)

#print(xname2)



#neigh2

ydata3 = storedWeeklyVals[6]
xpeak3 = storedPeak[6]
xname3 = storedName[6]


plt.plot(xdata, ydata3, color='green', linewidth=2, linestyle='--', label=xname3)
plt.axvline(x=xpeak3, color='green')


plt.xlabel("Time [hour]")
plt.ylabel("Internet Connections [#]")
plt.xlim([0,168])
#plt.ylim([0,10000])
plt.legend()
sns.despine()
fileSavePath = './output/' + xname1 + xname2 + xname3+'_latest_Combined.png'
plt.savefig(fileSavePath)








###SET OF PLOTS 2

#neigh1
plot1 = 9
plot2 = 3
plot3 = 8

f = plt.figure()

ydata1 = storedWeeklyVals[plot1]
xpeak1 = storedPeak[plot1]
xname1 = storedName[plot1]
#print(xname1)

plt.plot(xdata, ydata1, color='orange', linewidth=2, linestyle='--', label=xname1)
plt.axvline(x=xpeak1, color='orange')

#print(len(yfit_this))

#neigh2

ydata2 = storedWeeklyVals[plot2]
xpeak2 = storedPeak[plot2]
xname2 = storedName[plot2]


plt.plot(xdata, ydata2, color='blue', linewidth=2, linestyle='--', label=xname2)
plt.axvline(x=xpeak2, color='blue')

#print(yfit_this, yfit_this, xname1, xpeak1)

#print(xname2)



#neigh2

ydata3 = storedWeeklyVals[plot3]
xpeak3 = storedPeak[plot3]
xname3 = storedName[plot3]


plt.plot(xdata, ydata3, color='green', linewidth=2, linestyle='--', label=xname3)
plt.axvline(x=xpeak3, color='green')


plt.xlabel("Time [hour]")
plt.ylabel("Internet Connections [#]")
plt.xlim([0,168])
#plt.ylim([0,10000])
plt.legend()
sns.despine()
fileSavePath = './output/' + xname1 + xname2 + xname3+'_latest_Combined.png'
plt.savefig(fileSavePath)




###SET OF PLOTS 3

#neigh1
plot1 = 10
plot2 = 11

f = plt.figure()

ydata1 = storedWeeklyVals[plot1]
xpeak1 = storedPeak[plot1]
xname1 = storedName[plot1]
#print(xname1)

plt.plot(xdata, ydata1, color='orange', linewidth=2, linestyle='--', label=xname1)
plt.axvline(x=xpeak1, color='orange')

#print(len(yfit_this))

#neigh2

ydata2 = storedWeeklyVals[plot2]
xpeak2 = storedPeak[plot2]
xname2 = storedName[plot2]


plt.plot(xdata, ydata2, color='blue', linewidth=2, linestyle='--', label=xname2)
plt.axvline(x=xpeak2, color='blue')

#print(yfit_this, yfit_this, xname1, xpeak1)

#print(xname2)




plt.xlabel("Time [hour]")
plt.ylabel("Internet Connections [#]")
plt.xlim([0,168])
#plt.ylim([0,10000])
plt.legend()
sns.despine()
fileSavePath = './output/' + xname1 + xname2+'_latest_Combined.png'
plt.savefig(fileSavePath)





###SET OF PLOTS 4

#neigh1
plot1 = 1
plot2 = 12

f = plt.figure()

ydata1 = storedWeeklyVals[plot1]
xpeak1 = storedPeak[plot1]
xname1 = storedName[plot1]
#print(xname1)

plt.plot(xdata, ydata1, color='orange', linewidth=2, linestyle='--', label=xname1)
plt.axvline(x=xpeak1, color='orange')

#print(len(yfit_this))

#neigh2

ydata2 = storedWeeklyVals[plot2]
xpeak2 = storedPeak[plot2]
xname2 = storedName[plot2]


plt.plot(xdata, ydata2, color='blue', linewidth=2, linestyle='--', label=xname2)
plt.axvline(x=xpeak2, color='blue')

#print(yfit_this, yfit_this, xname1, xpeak1)

#print(xname2)





plt.xlabel("Time [hour]")
plt.ylabel("Internet Connections [#]")
plt.xlim([0,168])
#plt.ylim([0,10000])
plt.legend()
sns.despine()
fileSavePath = './output/' + xname1 + xname2 +'_latest_Combined.png'
plt.savefig(fileSavePath)




###SET OF PLOTS 5

#neigh1
plot1 = 2
plot2 = 0

f = plt.figure()

ydata1 = storedWeeklyVals[plot1]
xpeak1 = storedPeak[plot1]
xname1 = storedName[plot1]
#print(xname1)

plt.plot(xdata, ydata1, color='orange', linewidth=2, linestyle='--', label=xname1)
plt.axvline(x=xpeak1, color='orange')

#print(len(yfit_this))

#neigh2

ydata2 = storedWeeklyVals[plot2]
xpeak2 = storedPeak[plot2]
xname2 = storedName[plot2]


plt.plot(xdata, ydata2, color='blue', linewidth=2, linestyle='--', label=xname2)
plt.axvline(x=xpeak2, color='blue')

#print(yfit_this, yfit_this, xname1, xpeak1)

#print(xname2)





plt.xlabel("Time [hour]")
plt.ylabel("Internet Connections [#]")
plt.xlim([0,168])
#plt.ylim([0,10000])
plt.legend()
sns.despine()
fileSavePath = './output/' + xname1 + xname2 +'_latest_Combined.png'
plt.savefig(fileSavePath)


###SET OF PLOTS 6

#neigh1
plot1 = 7

f = plt.figure()

ydata1 = storedWeeklyVals[plot1]
xpeak1 = storedPeak[plot1]
xname1 = storedName[plot1]
#print(xname1)

plt.plot(xdata, ydata1, color='orange', linewidth=2, linestyle='--', label=xname1)
plt.axvline(x=xpeak1, color='orange')

#print(len(yfit_this))





plt.xlabel("Time [hour]")
plt.ylabel("Internet Connections [#]")
plt.xlim([0,168])
#plt.ylim([0,10000])
plt.legend()
sns.despine()
fileSavePath = './output/' + xname1 +'_latest_Combined.png'
plt.savefig(fileSavePath)


