import shapely.geometry as sg
import shapely.ops as so
import matplotlib.pyplot as plt


import geopandas as gpd
gdf = gpd.read_file(r'IT002L3_MILANO_UA2012_revised_020.gpkg')
geomot = gdf['geometry']
geomot1 = gdf['geometry'][1]
#print(geomot1)



#fig, ax = plt.subplots()
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
#gdf.plot(column = 'code_2012', ax=ax, legend = True)
#plt.title('Milan Land Use')
#ax.set_aspect('equal')
#plt.show()

print(gdf.crs)

gdf.to_pickle('./completedLandUse.pkl')