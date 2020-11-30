import shapely.geometry as sg
import shapely.ops as so
import matplotlib.pyplot as plt
from pyproj import Proj, transform

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import pandas as pd

import geopandas as gpd

df_gdf = pd.read_pickle("./completedLandUse.pkl")
milanoGrid = gpd.read_file('milano-grid.geojson')

p1 = Point(4267400.2195, 2484605.758399999)

#LAND USE IS EXACT OPPOSITE OF AIRBNB SET
inProj = Proj('epsg:4326')
outProj = Proj('epsg:3035')
#desired 4258
x1,y1 = 45.471613, 9.315573
#x1,y1 = 2484607.7584, 4267400.2196
x2,y2 = transform(inProj,outProj,x1,y1)


#print(df_gdf.head(8))
#polyInRange = df_gdf['geometry'][1]
#print(polyInRange)
polyInRange = pd.DataFrame()
subset_milano_landuse = pd.DataFrame()
outcome = False

for i in range(0,85167):
    polyInRange = df_gdf['geometry'][i]
    outcome = polyInRange.contains(p1)
    if outcome:
        print("Found it:")
        print(df_gdf['identifier'][i])
        

# to change all: df_gdf.to_crs({'init':'epsg:4326'})

#fig, ax = plt.subplots()
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
#gdf.plot(column = 'code_2012', ax=ax, legend = True)
#plt.title('Milan Land Use')
#ax.set_aspect('equal')
#plt.show()

#print(gdf.crs)

#gdf.to_pickle('./completedLandUse.pkl')




import shapely.geometry as sg
import shapely.ops as so
import matplotlib.pyplot as plt
from pyproj import Proj, transform

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import pandas as pd

import geopandas as gpd

df_gdf = pd.read_pickle("./completedLandUse.pkl")
milanoGrid = gpd.read_file('milano-grid.geojson')

df_gdf.to_crs({'init':'epsg:4326'})

polyInRange = pd.DataFrame()
subset_milano_landuse = pd.DataFrame()
outcome = False

for i in range(0, len(milanoGrid)):
    checkVal = milanoGrid.geometry[i].centroid
    for j in range(0,85167):
        polyInRange = df_gdf['geometry'][j]
        outcome = polyInRange.contains(checkVal)
        if outcome:
            print("Found it:")
            landUseUpdate = df_gdf['code_2012'][j]
            print(landUseUpdate)
            subset_milano_landuse = subset_milano_landuse.append({'Cell': i, 'LandUse': landUseUpdate}, ignore_index=True)