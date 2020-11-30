import shapely.geometry as sg
import shapely.ops as so
import matplotlib.pyplot as plt

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import pandas as pd

import geojson
import geopandas as gpd

#df_gdf = pd.read_pickle("./completedLandUse.pkl")
nameArrPoints = ["p1", "p2Test", "centro", "brera", "navigli", "sanvittore", "ticinese", "deangeliwagnerbuonarotti", "sansiro", "portanuova", "cittastudi", "portavenezia", "bicocca", "sandonato"]
print(nameArrPoints)

with open("../input/milano-grid.geojson") as json_file:
    json_data = geojson.load(json_file)
	
p1 = Point(9.1871018, 45.4723561)
p2Test = Point(9.151665, 45.470606)
centro = Point(9.188896, 45.464196)
brera = Point(9.187640, 45.470708)
navigli = Point(9.168886,45.448230)
sanvittore = Point(9.171999, 45.459901)
ticinese = Point(9.185778, 45.454232)
deangeliwagnerbuonarotti = Point(9.151665, 45.470606)
sansiro = Point(9.129824, 45.478174)
portanuova = Point(9.189952, 45.483430)
cittastudi = Point(9.223925, 45.478484)
portavenezia = Point(9.205348, 45.474231)
bicocca = Point(9.208425,45.519124)
sandonato = Point(9.264526,45.416563)

#nameArrPoints = [p1, p2Test, centro, brera, navigli, sanvittore, ticinese, deangeliwagnerbuonarotti, sansiro, portanuova, cittastudi, portavenezia, bicocca, sandonato]

arrPoints = [p1, p2Test, centro, brera, navigli, sanvittore, ticinese, deangeliwagnerbuonarotti, sansiro, portanuova, cittastudi, portavenezia, bicocca, sandonato]
#print(arrPoints)
length = len(arrPoints)
print(length)



#p1 = Point(4267400.2195, 2484605.758399999)

#LAND USE IS EXACT OPPOSITE OF AIRBNB SET
#inProj = Proj('epsg:4326')
#outProj = Proj('epsg:3035')
#desired 4258
#x1,y1 = 45.471613, 9.315573
#x1,y1 = 2484607.7584, 4267400.2196
#x2,y2 = transform(inProj,outProj,x1,y1)


#print(df_gdf.head(8))
#polyInRange = df_gdf['geometry'][1]
#print(polyInRange)
for j in range(length):
    point = arrPoints[j]
    #print(point)
    for i in range(1,10000):
        coordlistInRange = json_data.features[i]['geometry']['coordinates'][0]
        polyInRange = Polygon(coordlistInRange)
        outcome = polyInRange.contains(point)
        if outcome:
            #print("Found it:")
            print(nameArrPoints[j])
            prVal = json_data.features[i]['properties']['cellId']
            print(json_data.features[i]['properties']['cellId'])
            


#fig, ax = plt.subplots()
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
#gdf.plot(column = 'code_2012', ax=ax, legend = True)
#plt.title('Milan Land Use')
#ax.set_aspect('equal')
#plt.show()

#print(gdf.crs)

#gdf.to_pickle('./completedLandUse.pkl')