from pyproj import Proj, transform
from shapely.geometry import Point
from coordinates import Coordinate

inProj = Proj('epsg:4326')
outProj = Proj('epsg:3035')
#desired 4258

x1,y1 = 45.471613, 9.315573
x1y1 = Point(9.315573, 45.471613)
x2y2 =Point(x1,y1)
print(x2y2)
#x1,y1 = 45.471613, 9.315573
#x1,y1 = 2484607.7584, 4267400.2196
#x2y2 = transform(inProj,outProj,x1y1)



coord_1 = x1y1.x
coord_2 = x1y1.y

outy2,outx2 = transform(inProj,outProj,coord_2,coord_1)

#print(coord_1)
print(outx2)
print(outy2)

allout = Point(outx2, outy2)

print(allout)