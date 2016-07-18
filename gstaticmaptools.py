from __future__ import division
from motionless import CenterMap
import numpy as np
import requests
from PIL import Image
import cStringIO


__TILE_SIZE = 256
__INV_TILE_SIZE = 1 / __TILE_SIZE
TILE_SIZE = __TILE_SIZE  # So that external functions can read, but not write to this variable's value.
MAX_ZOOM = 23
D2R = np.pi / 180
R2D = 180 / np.pi
PI4 = np.pi * 4
INV_PI4 = 1 / PI4
EXP2 = np.array([np.power(2, i) for i in range(0, 32)])
INVEXP2 = np.array([np.power(np.float(2), -i) for i in range(0, 32)], dtype='float32')

API_KEY = '' #SET API KEY.



def latlngToWorld((lat, lon)):

	#Accepts tuple of Lat,Lon numpy 1D arrays/ or just a tuple of values. Recomended to use numpy arrays.
	"""
		:parameter lat: single latitude value, or a numpy 1D array of values
		:parameter lon: single longitude value, or a numpy 1D array of values
		:return Returns the array/value of World coordinate in the web mercator projection (Google), of the given latlngs.
		 		Definitions from https://developers.google.com/maps/documentation/javascript/examples/map-coordinates
	"""
	siny = np.sin(lat*np.pi/180)
	siny = np.clip(siny,-.9999,.9999) #Clipping to limit all lat longs. Refer to the above link for more details.
	mercpoint = ((0.5 + (lon / 360))*__TILE_SIZE,( 0.5 - ((np.log((1 + siny) / (1 - siny))) * (INV_PI4)))*__TILE_SIZE)  # print mercpoint*__TILE_SIZE
	return mercpoint
	#Works!

def latlngToPixel((lat,lon),z):
	"""
		:parameter lat: single latitude value, or a numpy 1D array of values
		:parameter lon: single longitude value, or a numpy 1D array of values
		:parameter z: single/array as the same size of lat/lon of zoom values. If z is single value and worldX, worldY are arrays, the same, given z value applies to all,
			     	else if z is array equal to size of worldX or worldY, then each zoom is individually applied.
		:return Returns the array/value pixel cordinates of the given latlngs at the given zoom level.

	"""
	(worldX,worldY) = latlngToWorld((lat,lon))
	(pixelX,pixelY) = worldToPixel((worldX,worldY),z)
	return (pixelX,pixelY)


def latlngToTile((lat,lng),z):
	"""
		:parameter lat: single latitude value, or a numpy 1D array of values
		:parameter lon: single longitude value, or a numpy 1D array of values
		:parameter z: single/array as the same size of lat/lon of zoom values. If z is single value and worldX, worldY are arrays, the same, given z value applies to all,
			     	else if z is array equal to size of worldX or worldY, then each zoom is individually applied.
		:return Returns the tile cordinates of the tiles  which contain the given latlngs at the given zoom level.
	"""
	(worldX,worldY) = latlngToWorld((lat,lng))
	(pixelX,pixelY) = worldToPixel((worldX,worldY),z)
	(tileX,tileY) = pixelToTile((pixelX,pixelY))
	return (tileX,tileY)


def worldToPixel((worldX,worldY),z):
	"""
		:parameter worldX: single world xcordinate or a 1D numpy array of values
		:parameter worldY: single world ycordinate or a 1D numpy array of values
		:parameter if z is single value and worldX, worldY are arrays, the same, given z value applies to all,
			     	else if z is array equal to size of worldX or worldY, then each zoom is individually applied.
		:return Returns converted world coordinates at a zoom level to pixel coordinates.
	"""
	_validrequest = False
	_validrequest = True if z<=MAX_ZOOM else False
	if _validrequest:
		xcord = np.floor(worldX * EXP2[z])
		ycord = np.floor(worldY * EXP2[z])
		return np.asarray((xcord,ycord),dtype=int)
	else:
		raise ValueError('Invalid zoom value. Must be under 21.')


def worldToLatLon((world_x,world_y)):
	"""
		:parameter worldX: single world xcordinate or a 1D numpy array of values
		:parameter worldY: single world ycordinate or a 1D numpy array of values

		:returns Returns the Inverse Projected world cordinates, i.e from webmercator to LatLong. Reverse engineered from
		  		link: https://developers.google.com/maps/documentation/javascript/examples/map-coordinates
	"""

	# Is vectorized already! Awaiting performance tests.
	lng = ((world_x * __INV_TILE_SIZE) * 360) - 180
	p = np.exp((-(world_y * __INV_TILE_SIZE) + .5) * PI4)
	lat = np.float(R2D) * (np.arcsin((p - 1) / (1 + p)))
	return ((lat,lng))



def tileToPixel((tileX,tileY)):
	"""
		:parameter tileX: single tile xcordinate or 1D numpy array of values
		:parameter tileY: single tile ycordinate or 1D numpy array of values
		:return Returns top left corner pixel of tile's cordinate.
	"""

	pixelX = tileX * __TILE_SIZE
	pixelY = tileY * __TILE_SIZE
	return (pixelX,pixelY)


def tileToCenterPixel((tileX,tileY)):
	"""
		:parameter tileX: single tile xcordinate or 1D numpy array of values
		:parameter tileY: single tile ycordinate or 1D numpy array of values
		:return Returns center pixel cordinate of tile.
	"""
	(pixelX, pixelY) = tileToPixel((tileX, tileY))
	return (pixelX + 128, pixelY + 128)


def tileToLatlng((tileX, tileY), z):
	"""
		:parameter tileX: single tile x-cordinate or a 1D numpy array of values
		:parameter tileY: single tile y-cordinate or a 1D numpy array of values
		:parameter if z is single value and pixel_x, pixel_y are arrays, the same, given z value applies to all,
			     	else if z is array equal to size of pixel_x or pixel_y, then each zoom is individually applied.
		:return    Returns corresponding latitude, longitude of the center of the center pixel of the tile.
	"""
	if np.any(np.array((tileX, tileY) > EXP2[z + 8])):
		raise ValueError('Invalid tile value for the given zoom')
	centrepixel = tileToCenterPixel((tileX,tileY))
	latlng = pixelToLatLng_center(centrepixel, z)
	print ('Latitude and Longitude of the centre pixel %s are %s')%(centrepixel,latlng)
	return latlng


def pixelToWorld((pixel_x,pixel_y),z):
	"""
		:parameter pixel_x: single pixel_x cordinate or a 1D numpy array of values
		:parameter pixel_y: single pixel_y cordinate or a 1D numpy array of values
		:parameter if z is single value and pixel_x, pixel_y are arrays, the same, given z value applies to all,
			     	else if z is array equal to size of pixel_x or pixel_y, then each zoom is individually applied.
		:return Returns converted pixel coordinates at a zoom level to world coordinates.
	"""
	z = np.array(z).astype('int')
	if np.any(np.abs(np.array((pixel_x, pixel_y))) > EXP2[z + 8]):
		raise ValueError('Invalid Pixel value for given zoom')

	scale_factor = INVEXP2[z]
	world_x = np.asarray(pixel_x, dtype='float') * scale_factor
	world_y = np.asarray(pixel_y, dtype='float') * scale_factor
	return (world_x,world_y)


def pixelToLatlng_corner((pixelX, pixelY), z):
	"""
		:parameter pixelX: single pixelx cordinate or a 1D numpy array of values
		:parameter pixelX: single pixely cordinate or a 1D numpy array of values
		:parameter if z is single value and pixel_x, pixel_y are arrays, the same, given z value applies to all,
			     	else if z is array equal to size of pixel_x or pixel_y, then each zoom is individually applied.
		:return    Returns corresponding latitude, longitude of the top left corner of the pixel/s.
	"""
	# latlong coordinate of a pixel's top left corner.
	(worldX,worldY) = pixelToWorld((pixelX,pixelY),z)
	return worldToLatLon((worldX,worldY))


def pixelToTile((pixelX,pixelY)):
	"""
		:parameter pixel_x: single pixel_x cordinate or a 1D numpy array of values
		:parameter pixel_y: single pixel_y cordinate or a 1D numpy array of values
		:return Returns the tile cordinates of the tiles  which contain the given pixels.
	"""
	tileX = np.floor(pixelX * __INV_TILE_SIZE)
	tileY = np.floor(pixelY * __INV_TILE_SIZE)
	return np.asarray((tileX,tileY),dtype=int)





def pixelToLatLng_center(pixelXY, zoom):
	"""
	:param pixelXY: Expects a nby2 numpy array or a single tuple of lat,lng in that order.
	:param z: Expects a integer zoom value, of range 0-MAX_ZOOM
	:return: Returns the latitude, longitude of the center of the pixel at the specified zoom level
			 in the same data object type they arrived in. Tuple->Tuple, or Array->Array.
	.. seealso:: pixelToLatLng_corner()

	By design, the pixeltoLatLng converts to the Latitude and Longitude corresponding to the top left corner of the pixel.
	So for better results, we calculate all the latlongs of all four corners (Different pixels' shared corners, infact) and
	 average them up, to get the lat long of the center of the pixel instead of a corner.
	"""



	z = np.array(zoom).astype(
		'int')  # Just to make it a int. If it is a single number, it can still be used as a scalar
	# after converting it to an array. It's alright!
	try:
		if isinstance(pixelXY, tuple):
			pixelX = pixelXY[0]
			pixelY = pixelXY[1]

		elif hasattr(pixelXY, '__iter__'):
			# It is very likely a numpy array or a list of lists.
			pixelX = np.array(pixelXY[:, 0]).astype('int')
			pixelY = np.array(pixelXY[:, 1]).astype('int')

			if np.size(zoom) == 1:
				zoom = np.ones(np.size(pixelX)) * zoom
			elif np.size(zoom) != np.size(pixelX):
				raise RuntimeError("Invalid number of zoom values. Does'n't match the size of Pixel Array.")



	except TypeError:
		raise ('Input must be a tuple or a numpy array.')

	pdb.set_trace()
	w = pixelToWorld((pixelX, pixelY), z)
	list_sides = []
	list_sides.append(w)
	list_sides.append(pixelToWorld((pixelX, pixelY + 1), z))
	list_sides.append(pixelToWorld((pixelX + 1, pixelY), z))
	list_sides.append(pixelToWorld((pixelX + 1, pixelY + 1), z))
	ele1 = 0
	ele2 = 0
	for i in range(4):
		# print(list_sides[i])
		ele1 += list_sides[i][0]
		ele2 += list_sides[i][1]
	w_new = (ele1 / 4, ele2 / 4)
	# print w_new
	# Vectorized already.
	latlng_final = worldToLatLon(w_new)
	# print(latlng_final)
	return (latlng_final)







class Tile():
	# Object to help retrieve specific tiles.
	def __init__(self, (tileX, tileY), zoom, **kwargs):
		self.tileX = tileX
		self.tileY = tileY
		self.zoom = zoom

		self.maptype = kwargs.get('maptype','satellite')
		self.getCenterLatLng()
		self.get_tile()
		self.__tilesize__ = TILE_SIZE #Set the TILE_SIZE in gmapshelpers.py
		self.__requestsize__ = self.__tilesize__ + 95 #Trimming purposes. Requests this size from gmaps static API.
		# self.Image.show()
		return

	def getCenterLatLng(self):
		"""
		Method to get the center point's latitude and longitude, we use this to request GMAPS Static API the tile, which when returned would
		actually be a tile and not a combination of tiles.
		"""
		self.centerlatlng = tileToLatlng((self.tileX, self.tileY), self.zoom)
		print('The assosicated Latitude and Longitude are %s and %s') % (self.centerlatlng[0], self.centerlatlng[1])
		return

	def get_tile(self):
		"""
		Method to request the google maps the tile using the coordinates obtained from the getCenterLatLng method.
		"""
		if API_KEY:
			self.cmap = CenterMap(lat=self.centerlatlng[0], lon=self.centerlatlng[1], zoom=self.zoom,
							  maptype=self.maptype, size_x=351, size_y=351, key=API_KEY)
		else:
			self.cmap = CenterMap(lat=self.centerlatlng[0], lon=self.centerlatlng[1], zoom=self.zoom,
								  maptype=self.maptype, size_x=351, size_y=351)
		# If we ask for a tile of even length and width, there is no center pixel per se, and we have
		# to request the tiles with the latlng of a point between the center 4 pixels. Instead, it is better to
		# request a tile with odd number of pixels, so that we have an actual center pixel, and then later trim it
		# to get the square tile we want.
		self.url = self.cmap.generate_url()
		print('Requesting Static Tile from GMAPS at %s') % (self.url)
		self.r = requests.get(self.url)
		print('Respose recieved with status code: %s and size:%s') % (
		self.r.status_code, self.r.headers['content-length'])
		self.image_fileobject = cStringIO.StringIO(self.r.content)
		self.fullimage = Image.open(self.image_fileobject)
		self.image = (self.fullimage.copy()).crop((47, 47, 303, 303))
		return self.image

	def writetofile(self, filename):
		self.image.save(filename) #PNG is the default format.
		return

	def pixelToLatLng(self, pixel):
		"""
		Method to get any pixel/list of pixels's Latitude Longitude in the Image.
		:param pixel: expects a numpy array of nby2 size of pixels, but a single pixel tuple is also fine.
						Anything else, it throws an error.
		:return Returns, corresponding Latitude and Longitude of the center of the pixel/list of pixels in the current tile.
		"""
		self.pixel_offset=tileToPixel((self.tileX,self.tileY))
		if isinstance(pixel,tuple):
			#Means a tuple. Adding is exclusive.
			self.pixelXY=(pixel[0]+self.pixel_offset[0],pixel[1]+self.pixel_offset[1])
			return pixelToLatLng_center(self.pixelXY,self.zoom)
		elif hasattr(pixel,'__iter__'):
			#Probably means a numpy array. We hope it is a numpy array.
			self.pixelXY = np.array(pixel)
			self.pixelXY = self.pixelXY+self.pixel_offset
		return pixelToLatLng_center(self.pixelXY,self.zoom)

