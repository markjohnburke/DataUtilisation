import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import geopandas as gpd

import geojson
import matplotlib.colors as colors
import matplotlib.cm as cmx
#import matplotlib as mpl
from descartes import PolygonPatch


from mpl_toolkits.mplot3d import Axes3D




plt.rcParams['font.size'] = 14

sns.set_style("ticks")
sns.set_context("paper")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))


df_airbnb = pd.read_pickle("./completedAirbnb.pkl")
df_cdrs_internet = pd.read_pickle("./completedInternet.pkl")


print(df_airbnb.head(5))
print(df_airbnb.columns.tolist())

df_airbnb['cellY'] = 1
df_airbnb['cellX'] = 1
df_airbnb['countFreq'] = 1
print(df_airbnb.head(5))

print(df_airbnb['longitude'].dtypes)

print(df_airbnb['cellID'].dtypes)


df_airbnb['countFreq'] = df_airbnb.groupby('cellID')['countFreq'].transform('count')


print(df_airbnb.head(5))

valN = 100

print("CODE HERE")
for j, row in df_airbnb.iterrows():
    cellIDofThisRange = df_airbnb['cellID'][j]
    updatedY = cellIDofThisRange // valN
    updatedX = cellIDofThisRange % valN
    df_airbnb.at[j, 'cellY'] = updatedY    
    df_airbnb.at[j, 'cellX'] = updatedX

print(df_airbnb.head(5))
print(df_airbnb['countFreq'].max())


df_new_airbnb = df_airbnb.filter(['cellID','cellY','cellX','countFreq'])

print(df_new_airbnb.head(5))

df_newest_airbnb = df_new_airbnb.drop_duplicates(['cellID'])

print(df_newest_airbnb)

with open("../input/milano-grid.geojson") as json_file:
    json_data = geojson.load(json_file)
	
point1 = Point(9.1871018, 45.4723561)
#print(point1.dtypes)
print("SECOND END HERE")

coordlist = json_data.features[1]['geometry']['coordinates'][0]

print(coordlist)

poly = json_data.features[1]['geometry']
poly2 = Polygon(coordlist)

print(poly)
print(poly2)
print(point1)



df_newest_airbnb2 = df_new_airbnb

for i in range(1,10000):    
    #cellIDofRangeHere = df_newest_airbnb['cellID'][i]
    #internetData = df_cdrs_internet[df_cdrs_internet.CellID==i]['internet']
    if i not in df_newest_airbnb2['cellID']:
        print('not found')       
        newUpdatedY = i // valN
        newUpdatedX = i % valN     
        tempdf = {'cellID': i, 'cellX': newUpdatedX, 'cellY': newUpdatedY, 'countFreq': 0}
        df_newest_airbnb2 = df_newest_airbnb2.append(tempdf, ignore_index=True)
    else:
        print(df_newest_airbnb2['cellID'][i])
        
        
print(df_newest_airbnb2.head())  
print(df_newest_airbnb2)  

ypos = df_newest_airbnb2['cellY'].values
xpos = df_newest_airbnb2['cellX'].values


#df_newest_airbnb3 = df_newest_airbnb2.drop_duplicates(['cellX'])
#print(df_newest_airbnb3)

#ypos = df_newest_airbnb3['cellY'].values
#xpos = df_newest_airbnb3['cellX'].values

import sys
np.set_printoptions(threshold=sys.maxsize)
print(xpos)
print(ypos)
print(ypos)

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')


num_elements = len(xpos)
zpos = np.ones(10000)

dx = np.ones(10000)
dy = np.ones(10000)
#dz = df_newest_airbnb3['countFreq'].values

dz = df_newest_airbnb2['countFreq'].values

ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#00ceaa')

ax1.set_xlabel('Grid X Position')
ax1.set_ylabel('Grid Y Position')
ax1.set_zlabel('Number of Properties')

plt.show()

#mean of hours
num = int(10000)
arr_cellID = np.zeros(num)
arr_mean = np.zeros(num)
eightAmMondayCells = np.zeros(num)
twoAmMondayCells = np.zeros(num)
twoPmWednesdayCells = np.zeros(num)
tenAmThursdayCells = np.zeros(num)
twoPmFridayCells = np.zeros(num)
ninePmSaturdayCells = np.zeros(num)
maxCell = 1
for i in range(1,num):
    ydata = df_cdrs_internet[df_cdrs_internet.CellID==i]['internet']   
    xdata = df_cdrs_internet[df_cdrs_internet.CellID==i]['internet'].index
    mean = np.mean(ydata)
    print(mean)
    arr_cellID[i]=i    
    arr_mean[i]=mean
    eightAmMondayCells[i]=ydata[8]
    twoAmMondayCells[i]=ydata[2]    
    twoPmWednesdayCells[i]=ydata[62]     
    tenAmThursdayCells[i]=ydata[82]    
    twoPmFridayCells[i]=ydata[110]         
    ninePmSaturdayCells[i]=ydata[141]
    xpos[i] = i // valN
    ypos[i] = i % valN

maxMean = np.argmax(arr_mean)  
maxMeanVal = np.max(arr_mean)  
print(maxMean)
print(maxMeanVal)
eightAmMax = np.argmax(eightAmMondayCells)  
eightAmMaxVal = np.max(eightAmMondayCells)  
print(eightAmMax)
print(eightAmMaxVal)
twoAmMondayMax = np.argmax(twoAmMondayCells) 
twoAmMondayMaxVal = np.max(twoAmMondayCells)   
print(twoAmMondayMax)
print(twoAmMondayMaxVal)
twoPmWednesdayMax = np.argmax(twoPmWednesdayCells)  
twoPmWednesdayMaxVal = np.max(twoPmWednesdayCells)  
print(twoPmWednesdayMax)
print(twoPmWednesdayMaxVal)
tenAmThursdayMax = np.argmax(tenAmThursdayCells)  
tenAmThursdayMaxVal = np.max(tenAmThursdayCells) 
print(tenAmThursdayMax)
print(tenAmThursdayMaxVal)
twoPmFridayMax = np.argmax(twoPmFridayCells)  
twoPmFridayMaxVal = np.max(twoPmFridayCells)  
print(twoPmFridayMax)
print(twoPmFridayMaxVal)
ninePmSaturdayMax = np.argmax(ninePmSaturdayCells)  
ninePmSaturdayMaxVal = np.max(ninePmSaturdayCells)  
print(ninePmSaturdayMax)
print(ninePmSaturdayMaxVal)

#maxMeanCell = df_cdrs_internet['cellID']['internet'][maxMean]
#print(maxMeanCell)

#ypos = df_newest_airbnb2['cellY'].values
#xpos = df_newest_airbnb2['cellX'].values


#df_newest_airbnb3 = df_newest_airbnb2.drop_duplicates(['cellX'])
#print(df_newest_airbnb3)

#ypos = df_newest_airbnb3['cellY'].values
#xpos = df_newest_airbnb3['cellX'].values



# MEAN Plot
import sys
np.set_printoptions(threshold=sys.maxsize)
print(xpos)
print(ypos)
print(ypos)

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')


num_elements = len(xpos)
zpos = np.ones(10000)

dx = np.ones(10000)
dy = np.ones(10000)
#dz = df_newest_airbnb3['countFreq'].values

dz = arr_mean

ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#ce4300')

ax1.set_xlabel('Grid X Position')
ax1.set_ylabel('Grid Y Position')
ax1.set_zlabel('Number of Connections Per Grid')

plt.show()


#ypos = df_newest_airbnb3['cellY'].values
#xpos = df_newest_airbnb3['cellX'].values



# 8 AM Monday Plot
import sys
np.set_printoptions(threshold=sys.maxsize)
print(xpos)
print(ypos)
print(ypos)

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')


num_elements = len(xpos)
zpos = np.ones(10000)

dx = np.ones(10000)
dy = np.ones(10000)
#dz = df_newest_airbnb3['countFreq'].values

dz = eightAmMondayCells

ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#ce0024')

ax1.set_xlabel('Grid X Position')
ax1.set_ylabel('Grid Y Position')
ax1.set_zlabel('Number of Connections Per Grid')

plt.show()


# Two AM Monday Plot


#dz = df_newest_airbnb3['countFreq'].values

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

dz = twoAmMondayCells

ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#ce0024')

ax1.set_xlabel('Grid X Position')
ax1.set_ylabel('Grid Y Position')
ax1.set_zlabel('Number of Connections Per Grid')

plt.show()


# Ten Am Thursday Plot


#dz = df_newest_airbnb3['countFreq'].values

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

dz = tenAmThursdayCells

ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#ce0024')

ax1.set_xlabel('Grid X Position')
ax1.set_ylabel('Grid Y Position')
ax1.set_zlabel('Number of Connections Per Grid')

plt.show()


# Two Pm Wednesday Plot

#dz = df_newest_airbnb3['countFreq'].values

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

dz = twoPmWednesdayCells

ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#ce0024')

ax1.set_xlabel('Grid X Position')
ax1.set_ylabel('Grid Y Position')
ax1.set_zlabel('Number of Connections Per Grid')

plt.show()



# Two Pm Friday Plot


#dz = df_newest_airbnb3['countFreq'].values

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

dz = twoPmFridayCells

ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#ce0024')

ax1.set_xlabel('Grid X Position')
ax1.set_ylabel('Grid Y Position')
ax1.set_zlabel('Number of Connections Per Grid')

plt.show()




# Nine Pm Saturday Plot


#dz = df_newest_airbnb3['countFreq'].values

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

dz = ninePmSaturdayCells

ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#ce0024')

ax1.set_xlabel('Grid X Position')
ax1.set_ylabel('Grid Y Position')
ax1.set_zlabel('Number of Connections Per Grid')

plt.show()

