import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pickle

fileSavePath = ''
colname = ''
milanoGrid = gpd.read_file('milano-grid.geojson')
milanNeighbourhood = gpd.read_file('MilanNeighbourhoodsJSON.geojson')
#milano.plot()
#plt.show()

df = pd.read_pickle('dataframe_collectionIndexed')

#Hardcode Array for neighbourhood
#a = [5437,5438,5536,5537,5538,5539,5635,5636,5637,5638,5639,5640,5641,5642,5643,5644,5735,5736,5737,5738,5739,5740,5741,5742,5743,5744,5835,5836,5837,5838,5839,5840,5841,5842,5843,5935,5936,5937,5938,5939,5940,5941,5942,6036,6037,6038,6039]
subset_milano = pd.DataFrame()
Neighbourhoods_df =  pd.DataFrame()

subset_milano_landuse = pd.DataFrame()


#First FOR loop is to extract columns per neighbourhood
for y in range(0, len(df)):
    #f = plt.figure()
    colnameDf =  df[y]
    colname = ''
    for col in colnameDf.columns:
        colname = col
    print(colname)
    cellsToConsider = colnameDf[colname]
    cellsArray = colnameDf[colname].values
    #Plot actual neighbourhood boundaries using the name of neighbourhood and the geojson created from the kml google earth file
    milanNeighbourhoodSub =  milanNeighbourhood[milanNeighbourhood.name==colname]
    #create new dataframe of each neighbourhood for future use/pickling
    Neighbourhoods_df = Neighbourhoods_df.append(milanNeighbourhoodSub)
    print(milanNeighbourhoodSub)
    #Check which cells are part of the neighbourhood and those to plotting dataframe
    for i in range(1,len(milanoGrid)):
        if i in cellsArray:            
            #plot neighbourhood as grid
            subset_milano = subset_milano.append(milanoGrid[milanoGrid.cellId==i])
            
            #edits here for neighbourhood land use: subset_milano = subset_milano.append(
            #print(subset_milano)
    #turn into Geopandas
    subset_milano_gpd = gpd.GeoDataFrame(subset_milano)    
    #print(subset_milano_gpd)
    


    #ax = milanNeighbourhood.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
    
    ax = subset_milano_gpd.plot()
    milanNeighbourhoodSub.plot(color='red', alpha=0.7, ax=ax)
    fileSavePath = './output/' + colname + '.png'
    plt.savefig(fileSavePath)
    subset_milano = pd.DataFrame()
    #plt.show()    

print(Neighbourhoods_df)
#print(Neighbourhoods_df[Neighbourhoods_df.name==colname])
Neighbourhoods_df.to_pickle('neighbourhoodDF')

#f = plt.figure()
#df_milano = milano['geometry']

#milano['geometry']
#print(milano)
#print(df_milano)
#ff = plt.figure()
#plt = milano.plot(color='blue')
#plot = ax.plt()
#plt.show()
#geoplot.polyplot(milano, figsize=(8, 4))