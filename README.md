# GMAPTILETOOLS

The library is created to facilitate easy creation, manipulation, visualization of maps offline. It has many additional empowering tools for freedom-seeking developers.

## Features:

1) Completely reverse-engineered Google's map-storage engine. With this library, you can seek google's own building blocks of its maps infrastructure, called the 'tiles'. This gives your more power over Google Maps Static Image API, as this has the ability to let you create your own maps precisely as how you need them. You can create them of any shapes, sizes or locations because you have the basic building blocks. For reference about google maps' terminology and organization: 'https://developers.google.com/maps/documentation/javascript/coordinates'.

2) Tools to manipulate indivudal and groups of tiles. These tools help you to merge/scale/draw-over individual or group of tiles. With the help of the 'canvas' class, you can easily manipulate an ordered set of tiles to create larger map images. You can also draw over them, to create visualizations like heatmaps. 

3) Full caching for reduced network use. You don't have to download these map tiles everytime you want to use them. Once they are downloaded, they are written to storage so that they can be reused. Anytime a tile is sought, the library looks for offline cached-tiles and then goes on to seek them off the API.

4) Pixel level mathematical conversion functions. As a by product of reverse engineering the GMAPS API, the library facilitates deep information about each pixel. You can identify the pixel's lat-lng in the real world, or do the reverse. It enables conversion to and fro from WGS84 standard, Tile coordinates, Pixel cordinates and Latitude Longitude co-ordinates. This helps one keep track of the real location of every part of the map and vice-versa. 

5) Pixel level data assosciation: As each pixel has conversion functions that you can use to know it's real co-ordinates and vice-versa, one can assosciate various other forms data to each pixel for later use. This is common in GIS with data like depth, IR, water-levels etc.

6) Depth information and 3D visualization: As a bonus, the library fetches depth-information and interpolates them to give you perfect depth information at any place you seek in your map. If you have MAYAVI-3D, you can also render the map in 3D. Cool, right!?



## Dependencies:

1)motionless

2)mayavi-3d

3)requests

4)tvtk

5)numpy

6)os

7)PIL

8)cStringIO



## Contact:
1) I know the documentation is non-existent and is underway to be published. In the mean-time, if you need any assistance or help, feel free to reach out at 
manish97ravula at gmail dot com.


