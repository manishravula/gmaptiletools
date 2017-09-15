from __future__ import division

import cStringIO
import errno
import numpy as np
import os
import pdb
import requests
from PIL import Image
from PIL import ImageDraw
from mayavi import mlab
from tvtk.api import tvtk

import googlemaps
from motionless import CenterMap

import gis

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


geography = gis.geography()

API_KEY = ''  # SET API KEY.


# todo: ditch tuples. Make numpy inputs compulsory.
# todo: test suite. No randomness. Just plain hardcoded tests. Estimated time: 1hr
# todo: Make a graphical window with a same option, which displays a GIF of sorts and asks if they both are the same. Estimated time: 2hrs
# todo: Obtain 10 tiles at each zoom level. Necessary for testing. Total: 10*20 = 200. Hard! Estimated time: 4hrs+
# todo: Make the docs perfect. Estunate time: 1 hr.
# todo: Make setup.py
# Estimated time: 8 hours.


def latlngToWorld(latlng):
	"""
		:parameter latlng: nby2 numpy array of latitude and longitudes.
		:return: Returns the world coordinates of the given latitude and longitudes in the web mercator projection (Google).
		 		Definitions from https://developers.google.com/maps/documentation/javascript/examples/map-coordinates
	"""
	if latlng.ndim > 1:
		lat = latlng[:, 0]
		lng = latlng[:, 1]
	else:
		lat = latlng[0]
		lng = latlng[1]
	siny = np.sin(lat * np.pi / 180)
	siny = np.clip(siny, -.9999, .9999)  # Clipping to limit all lat longs. Refer to the above link for more details.
	mercpoints = ((0.5 + (lng / 360)) * __TILE_SIZE,
				  (0.5 - ((np.log((1 + siny) / (1 - siny))) * (INV_PI4))) * __TILE_SIZE)  # print mercpoint*__TILE_SIZE
	return np.vstack(mercpoints)


def latlngToPixel(latlng, z):
	"""
		:parameter latlng: nby2 numpy array of latitude and longitudes.
		:parameter z: single/array as the same size of lat/lon of zoom values. If z is single value and worldX, worldY are arrays, the same, given z value applies to all,
			     	else if z is array equal to size of worldX or worldY, then each zoom is individually applied.
		:return: Returns the array/value pixel cordinates of the given latlngs at the given zoom level.

	"""
	worldXY = latlngToWorld(latlng)
	pixelXY = worldToPixel(worldXY, z)
	return pixelXY


def latlngToTile(latlng, z):
	"""
		:parameter latlng: nby2 numpy array of latitude and longitudes.
		:parameter z: single/array as the same size of lat/lon of zoom values. If z is single value and latlng is an array, the same, given z value applies to all,
			     	else if the length of z array is equal to the length of latlng, then each zoom is individually applied.
		:return: Returns a nby2 numpy array of tile cordinates of the tiles which contain the given latlngs at the given zoom level.
	"""
	worldXY = latlngToWorld(latlng)
	pixelXY = worldToPixel(worldXY, z)
	tileXY = pixelToTile((pixelXY))
	return tileXY


def worldToPixel(worldXY, z):
	"""
		:parameter worldXY: nby2 numpy array of world X and Y coordinates.
		:parameter z: if z is single value and worldXY is an array, the same, given z value applies to all,
			     	else if the length z array is equal to length of worldXY, then each zoom is individually applied.
		:return: Returns nby2 numpy array of converted world coordinates at a zoom level to pixel coordinates.
	"""
	_validrequest = False
	_validrequest = True if z <= MAX_ZOOM else False
	if _validrequest:
		return np.asarray(worldXY * EXP2[z], dtype=int)
	else:
		raise ValueError('Invalid zoom value. Must be under 21.')


def worldToLatLon(worldXY):
	"""
		:parameter worldXY: nby2 numpy array of world X and Y coordinates.
		:return:s Returns the Inverse Projected world cordinates, i.e from webmercator to LatLong. Reverse engineered from
		  		link: https://developers.google.com/maps/documentation/javascript/examples/map-coordinates
	"""
	if worldXY.ndim > 1:
		worldX = worldXY[:, 0]
		worldY = worldXY[:, 1]
	else:
		worldX = worldXY[0]
		worldY = worldXY[1]
	lng = ((worldX * __INV_TILE_SIZE) * 360) - 180
	p = np.exp((-(worldY * __INV_TILE_SIZE) + .5) * PI4)
	lat = np.float(R2D) * (np.arcsin((p - 1) / (1 + p)))
	lat = lat.reshape(-1,1)
	lng = lng.reshape(-1,1)

	latlng = np.hstack((lat,lng))
	if(latlng.shape[0]==1):
		return latlng[0]
	else:
		return latlng


def tileToPixel(tileXY):
	"""
		:parameter tileXY: nby2 numpy array of tile XY cordinates, in XY order.
		:return: Returns nby2 numpy array of top left corner pixel locations of tile's cordinate.
	"""
	return tileXY * __TILE_SIZE


def tileToCenterPixel(tileXY):
	"""
		:parameter tileXY: nby2 numpy array of tile XY cordinates, in XY order.
		:return: Returns nby2 numpy array of center pixels' cordinates' of tile.
	"""
	pixelXY = tileToPixel(tileXY)
	return pixelXY + 128


def tileToLatlng((tileXY), z):
	"""
		:parameter tileXY: nby2 numpy array of tile XY cordinates, in XY order.
		:parameter z: if z is single value and pixel_x, pixel_y are arrays, the same, given z value applies to all,
			     	else if z is array equal to size of pixel_x or pixel_y, then each zoom is individually applied.
		:return:    Returns corresponding latitude, longitude of the center of the center pixel of the tile.
	"""
	if np.any((tileXY) > EXP2[z + 8]):
		raise ValueError('Invalid tile value for the given zoom')
	centrepixel = tileToCenterPixel(tileXY)
	latlng = pixelToLatLng_center(centrepixel, z)
	print ('Latitude and Longitude of the centre pixel %s are %s') % (centrepixel, latlng)
	return latlng


def pixelToWorld(pixelXY, z):
	"""
		:parameter tileXY: nby2 numpy array of tile XY cordinates, in XY order.
		:parameter z: if z is single value and pixel_x, pixel_y are arrays, the same, given z value applies to all,
			     	else if z is array equal to size of pixel_x or pixel_y, then each zoom is individually applied.
		:return: Returns converted pixel coordinates at a zoom level to world coordinates.
	"""
	z = np.array(z).astype('int')
	if np.any(np.abs(pixelXY) > EXP2[z + 8]):
		raise ValueError('Invalid Pixel value for given zoom')

	scale_factor = INVEXP2[z]
	# world_x = np.asarray(pixelX, dtype='float') * scale_factor
	# world_y = np.asarray(pixel_y, dtype='float') * scale_factor
	worldXY = pixelXY * scale_factor
	return pixelXY * scale_factor


def pixelToLatLng_corner(pixelXY, z):
	"""
		:param pixelXY: nby2 numpy array or a single tuple of pixel cordinates (pixelx, pixely) in that order.
		:parameter if z is single value and pixel_x, pixel_y are arrays, the same, given z value applies to all,
		    	else if z is array equal to size of pixel_x or pixel_y, then each zoom is individually applied.
		:return:    Returns corresponding latitude, longitude of the top left corner of the pixel/s.
	"""
	# latlong coordinate of a pixel's top left corner.


	worldXY = pixelToWorld(pixelXY, z)
	return worldToLatLon(worldXY)


def pixelToTile(pixelXY):
	"""
		:param pixelXY: nby2 numpy array of pixel cordinates (pixelx, pixely) in that order.
		:return: Returns the tile cordinates of the tiles  which contain the given pixels.
	"""
	# pixelX = pixelXY[:,0]
	# pixelY = pixelXY[:,1]
	tileXY = pixelXY * __INV_TILE_SIZE
	# tileY = np.floor(pixelY * __INV_TILE_SIZE)
	return np.asarray(tileXY, dtype='int')


def pixelToLatLng_center(pixelXY, zoom):
	"""
		:param pixelXY: nby2 numpy array  of pixel cordinates (pixelx, pixely) in that order.
		:param zoom: Zoom value/s in the range 0-MAX_ZOOM. If z is single value and pixel_x, pixel_y are arrays, the same, given z value applies to all,
			     	else if z is array equal to size of pixel_x or pixel_y, then each zoom is individually applied.
		:return: Returns the latitude, longitude of the center of the pixel at the specified zoom level
			 in the same data object type they arrived in. Tuple->Tuple, or Array->Array.
		.. seealso:: pixelToLatLng_corner((pixelX, pixelY), z)

			By design, the pixeltoLatLng converts to the Latitude and Longitude corresponding to the top left corner of the pixel.
			So for better results, we calculate all the latlongs of all four corners (Different pixels' shared corners, infact) and
	 		average them up, to get the lat long of the center of the pixel instead of a corner.
	"""

	z = np.array(zoom).astype(
		'int')  # Just to make it a int. If it is a single number, it can still be used as a scalar
	# after converting it to an array. It's alright!

	if np.size(zoom) == 1:
		zoom = np.ones(np.shape(pixelXY)[0]) * zoom
	elif np.size(zoom) != np.shape(pixelXY)[0]:
		raise RuntimeError("Invalid number of zoom values. Does'n't match the size of Pixel Array.")

	# pdb.set_trace()
	w = pixelToWorld(pixelXY, z)
	list_sides = []
	list_sides.append(w)
	list_sides.append(pixelToWorld(pixelXY + (0, 1), z))
	list_sides.append(pixelToWorld(pixelXY + (1, 0), z))
	list_sides.append(pixelToWorld(pixelXY + (1, 1), z))
	ele1 = 0
	for i in range(4):
		# print(list_sides[i])
		ele1 += list_sides[i]
	w_new = ele1/4
	# print w_new
	# Vectorized already.
	latlng_final = worldToLatLon(w_new)
	# print(latlng_final)
	return (latlng_final)



class Tile():
	# Object to help retrieve specific tiles.
	def __init__(self, (tileX, tileY), zoom, get_image=0, get_elevdata=0, maptype='satellite'):
		self.tileX = tileX
		self.tileY = tileY
		self.tileXY = np.array((self.tileX, self.tileY))
		self.zoom = zoom
		self.get_image = get_image
		self.maptype = maptype

		self.getCenterLatLng()
		self.tile_cache_dir = './' #Define your tile cache directory. This is default. The directory needs to be created beforehand.
		if self.get_image:
			self.get_tile()
		if self.get_elevdata:
			self.get_elevdata()
		self.__tilesize__ = TILE_SIZE  # Set the TILE_SIZE in gmapshelpers.py
		self.__requestsize__ = self.__tilesize__ + 95  # Trimming purposes. Requests this size from gmaps static API.
		# self.Image.show()
		self.gmapsclient = googlemaps.Client(API_KEY)
		return

	@classmethod
	def fromLatLng(cls, latlng, z, get_image=0, get_elevdata=0,maptype='satellite'):
		"""
		:param latlng: A nby2 numpy array of Latlngs or a single tuple (Latitude, Longitude).
		:param z: Zoom value/s in the range 0-MAX_ZOOM. If z is single value and pixel_x, pixel_y are arrays, the same, given z value applies to all,
			     	else if z is array equal to size of pixel_x or pixel_y, then each zoom is individually applied.
		:param maptype: String denoting the map type to request. Could be one of "satellite" or "terrain" or "roadmap".
		:return: a Tile instance/list of tile instances containing the given Latitude Longitude points in order. If multiple Latlngs belong
				to the same tile, the same instance is returned for all of them, in their respective positions in the final list.
		"""
		pdb.set_trace()
		self.maptype = maptype
		tileXY = latlngToTile(np.array(latlng), z)
		return cls((tileXY[0], tileXY[1]), zoom=z)




	def getCenterLatLng(self):
		"""
		Method to get the center point's latitude and longitude, we use this to request GMAPS Static API the tile, which when returned would
		actually be a tile and not a combination of tiles.
		"""
		#pdb.set_trace()
		self.centerlatlng = tileToLatlng(self.tileXY, self.zoom)
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

		filename = self.tile_cache_dir + '/' + str(self.zoom) + '/' + str((self.tileX, self.tileY)) + '.png'
		if (os.path.isfile(filename)):
			# So we have the image, hence we don't request it, but still create a tile class
			# This will help us with pixel related geo-calculations because of the tile methods.
			#t = Tile((i, j), z, get_image=0)  # We already have the image.
			self.image = Image.open(filename, 'r')
			self.image = self.image.convert(mode="RGB")

		# t.image = cv2.imread

		else:
			# So we don't have the image and hence we request it from GMAPS and save it for future use.
			if not (os.path.exists(os.path.dirname(filename))):
				try:
					os.makedirs(os.path.dirname(filename))
				except OSError as exc:  # Guard against race condition
					if exc.errno != errno.EEXIST:
						raise
			self.url = self.cmap.generate_url()
			print('Requesting Static Tile (%d,%d) from GMAPS at %s') % (self.tileXY[0], self.tileXY[1], self.url)
			self.r = requests.get(self.url)
			print('Respose recieved with status code: %s and size:%s') % (
				self.r.status_code, self.r.headers['content-length'])
			self.image_fileobject = cStringIO.StringIO(self.r.content)
			self.fullimage = Image.open(self.image_fileobject).convert(mode="RGB")
			self.image = self.fullimage.convert(mode="RGB")
			self.image = (self.fullimage.copy()).crop((47, 47, 303, 303))
			self.writetofile(filename)
		return self.image

	def writetofile(self, filename):
		if self.image:
			self.image.save(filename)  # PNG is the default format.
		else:
			raise Exception('Image not requested from server yet.')
		return

	def pixelToLatLng(self, pixel):
		"""
		Method to get any pixel/list of pixels's Latitude Longitude in the Image.
		:param pixel: expects a numpy array of nby2 size of pixels, but a single pixel tuple is also fine.
						Anything else, it throws an error.
						Also, it is in the format of pixelx, pixelY, which is opposite to original numpy convention which represents y axis first and x axis later.

		:return: Returns, corresponding Latitude and Longitude of the center of the pixel/list of pixels in the current tile.
		"""
		pixel = np.fliplr(pixel)#because first is y axis and second is x axis.
		self.pixel_offset = tileToPixel(np.array([self.tileX, self.tileY]))
		self.pixelXY = np.array(pixel)
		self.pixelXY = self.pixelXY + self.pixel_offset
		return pixelToLatLng_center(self.pixelXY, self.zoom)
	def get_elevdata_gmaps(self):
		"""
		Method to retrieve elevation data of the entire tile's pixels.
		:return: nothing
		"""
		pdb.set_trace()
		self.pixels = np.indices((TILE_SIZE,TILE_SIZE)).T.reshape(-1,2)
		#Generates list of indices like ((0,0),
		# array([[0, 0],
		# 	   [1, 0],
		# 	   [2, 0],
		# 	   [3, 0],
		# 	   [0, 1],
		# 	   [1, 1],
		# 	   [2, 1],
		# 	   [3, 1]])
		self.latlngOfpixels = self.pixelToLatLng(self.pixels)
		no_requests = self.latlngOfpixels.shape[0]
		batch_size = 500
		current_index = 0
		elevdata = []
		batch_indx = 0
		while(no_requests):
			current_batch_size = batch_size if no_requests>batch_size else no_requests
			latlng_sublist = self.latlngOfpixels[current_index:current_index+current_batch_size]
			elevdata_sublist = self.gmapsclient.elevation(latlng_sublist)
			elevdata_sublist = [ele['elevation'] for ele in elevdata_sublist]
			elevdata.extend(elevdata_sublist)
			no_requests-=current_batch_size
			current_index += current_batch_size
			print("The current batch index is %d \n", batch_indx)
			batch_indx+=1

		self.elevdata = np.array(elevdata)
		self.elevdata = self.elevdata.reshape(TILE_SIZE,TILE_SIZE).T
	def get_elevdata(self):
		"""
		Method to retrieve elevation data of the entire tile's pixels.
		:return: nothing
		"""

		pdb.set_trace()
		self.pixels = np.indices((TILE_SIZE,TILE_SIZE)).T.reshape(-1,2)
		#Generates list of indices like ((0,0),
		# array([[0, 0],
		# 	   [1, 0],
		# 	   [2, 0],
		# 	   [3, 0],
		# 	   [0, 1],
		# 	   [1, 1],
		# 	   [2, 1],
		# 	   [3, 1]])

		filename = self.tile_cache_dir + '/' + str(self.zoom) + '/' + str((self.tileX, self.tileY)) + '_elevdata.npy'
		if (os.path.isfile(filename)):
			#We have the data stored
			self.elevdata= np.load(filename)
		# t.image = cv2.imread
			print ("Loading elevation data from file "+str(filename)+"\n")

		else:
			#We don't have the data stored

			#Remember that pixelsToLatLng and everyother similar routine expects x and y cordinates of the pixels in that order.
			#and numpy, rather every other indexing scheme has the opposite convention, starting with y cordinate first and then x cordinate.
			print ("Loading elevation data from TIF. \n")

			self.latlngOfpixels = self.pixelToLatLng(self.pixels)


			#Because we are sending in the list with varying first elements which are interpreted as the x-cordinates by the converter
			#so, we would recieve the result with varying in x-direction first, which is longitudinal in that sense. Hence, our second co-ordinate(lng)
			#will differ first in the result. It would look something like,
			# lat,lng
			# lat, lng +1
			# lat, lng +2
			# lat+1, lng
			# lat+1, lng+1
			# lat+1, lng+2

			self.elevdata = geography.elevation(self.latlngOfpixels)

			#plain reshaping to get the data back in the format.
			self.elevdata = self.elevdata.reshape(TILE_SIZE, TILE_SIZE)
			if not (os.path.exists(os.path.dirname(filename))):
				try:
					os.makedirs(os.path.dirname(filename))
				except OSError as exc:  # Guard against race condition
					if exc.errno != errno.EEXIST:
						raise
			np.save(filename,self.elevdata)

		return

class Canvas():
	def __init__(self,latlng, maptype='satellite'):
		"""
		The canvas class is used to make manipulate combinations of multiple tiles.

		:param latlng: List of Latitude Longitudes for creating the canvas from individual tiles.
		:param maptype: Default map type to use.
		"""
		self.latlng = latlng
		self.maptype = maptype
		self.MAXSIZE = 1024
		self.LEAST_PIXELDISTANCE = 20  # 20px apart.
		self.canvas_cache_dir = './' #Define your canvas cache directory. You need to make a folder by yourself.

	def show_canvas(self):
		self.canvas.show()

	def render_3d(self):
		# MAYAVI seems to rotate 90 by counterclockwise and then flipleftright. So we do the opposite. We, flipleftright and ro
		# tate 90 by clockwise.
		# a = image
		# plt.imshow(a)
		# plt.show()
		#     r = a[:,:,0]
		#     g = a[:,:,1]
		#     b = a[:,:,2]
		#     rnew = np.fliplr(r.T)
		#     gnew = np.fliplr(g.T)
		#     bnew = np.fliplr(b.T)
		#     newimg = np.hstack((rnew.
		#     re = np.vstack((rnew.reshape(-1),gnew.reshape(-1), bnew.reshape(-1)))
		#     newimg = re.T.reshape((1024,1024,3))
		# 	plt.imshow(np.array(imageobject))
		# 	plt.show()
		#newimg = imageobject.transpose(Image.FLIP_LEFT_RIGHT)
		# plt.imshow(np.array(newimg))
		# plt.show()
		# plt.figure()

		# plt.imshow(np.array(newimg))
		# plt.show()
		self.canvas_mayavi = self.canvas.rotate(90)
		# self.canvas_mayavi = self.canvas
		self.canvas_mayavi_filename = self.canvas_cache_dir + '/' + str(self.zoom)+'/'+str(((self.top_tileSpan,self.left_tileSpan),(self.bottom_tileSpan,self.right_tileSpan)))+'_formayavi.png'
		self.canvas_mayavi.save(self.canvas_mayavi_filename)
		self.bmp1 = tvtk.PNGReader()
		self.bmp1.file_name = self.canvas_mayavi_filename

		self.texture = tvtk.Texture()
		self.texture.interpolate = 0
		self.texture.set_input(0, self.bmp1.get_output())

		mlab.figure(size=(640, 800), bgcolor=(0.16, 0.28, 0.46))

		surf = mlab.surf(self.DEM, color=(1, 1, 1), warp_scale=1)
		surf.actor.enable_texture = True
		surf.actor.tcoord_generator_mode = 'plane'
		surf.actor.actor.texture = self.texture
		mlab.show()

	def mark_latlngs(self,latlng_points):

		pdb.set_trace()
		pixel_cords = self.LatlngtoPixel(latlng_points)
		imdraw = ImageDraw.Draw(self.canvas)
		mark_size = 5 #in pixels.
		for pixel in pixel_cords:
			imdraw.rectangle(((pixel[0]-mark_size/2,pixel[1]-mark_size/2),(pixel[0]+mark_size/2,pixel[1]+mark_size/2)), fill='Red')
		self.show_canvas()


	def pixelToLatLng(self,pixelXY):
		"""
		:param pixelXY: nd-by-2 array of pixelXY cordinates to be converted into latitude longitude
		:return: LatLng: nd-by-2 array of LatLang cordinates converted from pixels.
		If pixel cordinates exceed the canvas size, IndexError exception is raised.
		"""


		global_pixelXY = np.fliplr(pixelXY) + self.topleft_pixel #So we are making all the pixels in the list to be aligned to the
		#Global pixel frame by adding the offset of the origin.
		return pixelToLatLng_center(global_pixelXY)

	def LatlngtoPixel(self, latlng):
		"""

		:param latlng: latitudes and longitudes to get the pixels locations in the canvas. in nby2 numpy array format.
		:return: Pixel cordinates in the canvas of the given latlng.
		"""
		global_pixelXY = latlngToPixel(latlng, self.zoom)
		return (global_pixelXY.T-self.topleft_pixel)

	#def Markpoints(self,latlng):

	def drawCanvas_auto(self):
		pdb.set_trace()
		self.calc_autozoom()
		self.drawCanvas(self.zoom_auto)

	def drawCanvas(self,z,with_DEM=0):
		"""
		:param z: zoom level at which the canvas must be built.
		Draws the canvas from individual elements to encompass the given set of latitudes and longitudes at the specified zoom.
		"""



		self.zoom = z
		self.tiles_XY = latlngToTile(self.latlng, z)  # Tile cordinates of all latlngs in specified zoom.
		# pdb.set_trace()

		# TileCordinates of edge borders of tiles.

		self.left_tileSpan = min(self.tiles_XY[0])
		self.right_tileSpan = max(self.tiles_XY[0])
		self.top_tileSpan = min(self.tiles_XY[1])
		self.bottom_tileSpan = max(self.tiles_XY[1])

		self.canvas_size = ((self.right_tileSpan - self.left_tileSpan + 1) * TILE_SIZE,(+1 + self.bottom_tileSpan - self.top_tileSpan) * TILE_SIZE)
		self.DEM_size = self.canvas_size

		canvas_filename = self.canvas_cache_dir + '/' + str(z)+'/'+str(((self.top_tileSpan,self.left_tileSpan),(self.bottom_tileSpan,self.right_tileSpan)))+'.png'
		self.canvas_filename = canvas_filename
		DEM_filename = self.canvas_cache_dir + '/' + str(z)+'/'+str(((self.top_tileSpan,self.left_tileSpan),(self.bottom_tileSpan,self.right_tileSpan)))+ '_DEM.npy'
		self.DEM_filename = DEM_filename
		tileList_filename = canvas_filename[0:-4] +'tileList'
		topLeftPixel_filename = canvas_filename[0:-4] +'topLeft.npy'
		if(os.path.isfile(canvas_filename)):
			canvas = Image.open(canvas_filename,"r")
			DEM  = np.load(DEM_filename)

			self.topleft_pixel = np.load(topLeftPixel_filename)
		else:
			#canvas = np.empty(self.canvas_size)
			canvas = Image.new('RGB',self.canvas_size)
			DEM = np.empty(self.DEM_size)


			for j in range(self.top_tileSpan, self.bottom_tileSpan + 1, 1):
				#pdb.set_trace()
				for i in range(self.left_tileSpan, self.right_tileSpan + 1, 1):  # Because we want the tile array to follow the same indexing as a numpy array.
					# Building tile blocks each column from top to bottom
					#pdb.set_trace()
					t = Tile((i, j), z, get_image=1,get_elevdata=1)
					q = i - self.left_tileSpan
					p = j - self.top_tileSpan
					t.image = t.image.convert(mode="RGB")
					t.numpyimage = np.array(t.image)
					#t.numpyimage = color.rgb2lab(np.array(t.image))
					loc = (q*256,p*256)
					#canvas[p * 256:(p + 1) * 256, q * 256:(q + 1) * 256,0] = t.numpyimage[:, :, 0]
					#canvas[p * 256:(p + 1) * 256, q * 256:(q + 1) * 256,1] = t.numpyimage[:, :, 1]
					#canvas[p * 256:(p + 1) * 256, q * 256:(q + 1) * 256,2] = t.numpyimage[:, :, 2]
					canvas.paste(t.image,loc)
					DEM[loc[0]:loc[0]+256,loc[1]:loc[1]+256] = t.elevdata


			self.topleft_pixel = tileToPixel(np.array([self.left_tileSpan,self.top_tileSpan]))
			#THis is also the pixel cordinate of the top left pixel in the canvas. Could be useful later on.
			if not (os.path.exists(os.path.dirname(canvas_filename))):
				try:
					os.makedirs(os.path.dirname(canvas_filename))
				except OSError as exc:  # Guard against race condition
					if exc.errno != errno.EEXIST:
						raise
			canvas.save(canvas_filename)
			np.save(topLeftPixel_filename,np.array(self.topleft_pixel))
			np.save(DEM_filename,DEM)



		self.canvas = canvas
		self.DEM = DEM


	def calc_autozoom(self):
		# Function to auto calculate the zoom for a given set of latlng points.

		pdb.set_trace()

		# Calculating maximum possible zoom to fit into the boundary size.
		self.worldCords = latlngToWorld(self.latlng)
		self.max_worldXDistance = (max(self.worldCords[1]) - min(self.worldCords[1]))
		if self.max_worldXDistance:
			self.max_zoomAcrossLength = np.log2(self.MAXSIZE/ self.max_worldXDistance)
		else:  # Means that only one point exists across the longitude
			self.max_zoomAcrossLength = np.log2(self.MAXSIZE)  # Because only one point exists, we set the zoom to be maximum.

		self.max_worldYDistance = (max(self.worldCords[1]) - min(self.worldCords[1]))
		if self.max_worldYDistance:
			self.max_zoomAcrossBreadth = np.log2(self.MAXSIZE / self.max_worldYDistance)
		else:  # Means that only one point exists across the latitude
			self.max_zoomAcrossBreadth = np.log2(self.MAXSIZE)  # Because only one point exists, we set the zoom to be maximum.
		self.max_zoom = np.floor(min(self.max_zoomAcrossBreadth, self.max_zoomAcrossLength))

		# Calculating minimum needed zoom

		# The closest elements need to be atleast least_distance number of pixels apart.


		# Calculating the closest distance along latitudes.
		self.Xdiffarray = self.worldCords[1][np.nonzero(np.diff(np.sort(self.worldCords[1])))[0]]
		if self.Xdiffarray.size == 0:  # Meaning only one Longitude exists. So no point worrying about zoom in that direction.
			self.min_zoomAcrossLength = 0
		else:
			self.least_worldXDistance = np.min(self.Xdiffarray)
			self.min_zoomAcrossLength = np.log2(self.LEAST_PIXELDISTANCE / self.least_worldXDistance)

		# Calculating the closest distance along longitudes.
		self.Ydiffarray = self.worldCords[0][np.nonzero(np.diff(np.sort(self.worldCords[0])))[0]]
		if self.Ydiffarray.size == 0:  # Meaning only one latitude exists. No point worrying about zoom in that direction
			self.min_zoomAcrossBreadth = 0
		else:
			self.least_worldYDistance = np.min(self.Ydiffarray)
			self.min_zoomAcrossBreadth = np.log2(self.LEAST_PIXELDISTANCE / self.least_worldYDistance)

		self.min_zoom = np.ceil(max(self.min_zoomAcrossBreadth, self.min_zoomAcrossLength))

		if (self.max_zoom < self.min_zoom):
			print('Size exceeding the limit due to close points')
			self.zoom_auto  = self.min_zoom
		else:
			self.zoom_auto = (np.floor((self.min_zoom + self.max_zoom) / 2))
		self.zoom_auto = int(self.zoom_auto)






