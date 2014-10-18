import sys
import getopt
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

class HistogramEqualization:
	def __init__(self, img):
		"""Create threshold image"""
		self.equalized_image = Image(img, None, None, None)
		self.rows,self.cols = self.equalized_image.shape()
		self.max_val = 256
		self.histogram = Histogram(self.equalized_image.matrix, self.max_val)
							
	def equalize(self):
		for i in xrange(self.rows):
			for j in xrange(self.cols):
				current = self.equalized_image.get_pixel(i,j)
				current.color = self.histogram.new_val(current.color)
				self.equalized_image.color_pixel(current)

	def plot(self):
		self.equalized_image.plot()

	def save(self, output_file):
		self.equalized_image.save(output_file)

	def save_text(self):
		"""Save matrix in csv file for debugging"""
		np.savetxt('in.csv', self.original_image.matrix, delimiter=',', fmt='%d')
		np.savetxt('out.csv', self.equalized_image.matrix, delimiter=',', fmt='%d')	
	

class Image:
	def __init__(self, matrix, rows, cols, background):
		"""Store matrix"""
		if matrix is None: 
			self.matrix = np.zeros((rows,cols), dtype=np.int)
		else:	
			self.matrix = matrix
		self.background = background

	def shape(self):
		return self.matrix.shape	

	def get_pixel(self, row, col):
		"""Return a specific pixel"""
		return Pixel(self.matrix.item(row,col), row, col)

	def get_pixels(self, coords):
		"""Return all pixels specified by coordinates array"""
		pixels = []
		row_max, col_max = self.shape()
		for c in coords:
			row_temp = c[0]
			col_temp = c[1]
			if row_temp >= 0 and col_temp >= 0 and row_temp < row_max and col_temp < col_max:
				px = self.get_pixel(row_temp,col_temp)
				pixels.append(px)
		return pixels		

	def color_pixel(self, pixel):
		"""color a specific pixel"""
		self.matrix.itemset((pixel.row,pixel.col), pixel.color)

	def plot(self):
		plt.imshow(self.matrix, cmap = 'gray', interpolation = 'nearest')
		plt.xticks([]), plt.yticks([])
		plt.show()

	def save(self, output_file):
		plt.imshow(self.matrix, cmap = 'gray', interpolation = 'nearest')
		plt.xticks([]), plt.yticks([])
		plt.savefig(output_file, bbox_inches='tight')		
	

class Pixel:
	def __init__(self, color, row, col):
		"""Create a pixel with color and coordinates"""
		self.color = color
		self.row = row
		self.col = col

	def is_color(self, color):
		"""Test if pixel has a color"""
		if self.color == color:
			return True
		else:
			return False

	def is_not_color(self, color):
		"""Test if pixel doesn't have a color"""
		if self.color != color:
			return True
		else:
			return False


class Histogram:
	def __init__(self, img, max_val):
		self.max_val = max_val
		self.hist = cv2.calcHist([img],[0],None,[self.max_val],[0,self.max_val])
		self.cumsum = self.hist.cumsum()
		self.cumsum_max = self.cumsum.max()

	def cdf(self, i):
		return self.cumsum[i]/self.cumsum_max

	def new_val(self, i):
		return math.floor((self.max_val - 1)*self.cdf(i))

						
def main():
	def usage():
		print 'python histogram_equalization.py -i <inputf> [-o <outputf>]'

	inputf = None
	outputf = None

	"""Process command line inputs"""
	try:
		opts, args = getopt.getopt(sys.argv[1:], "hi:o:", ["inputf=", "outputf="])
	except getopt.GetoptError:
		usage()
		sys.exit(2)
	for opt, arg in opts:
		if opt in ('-h', '--help'):
			usage()
			sys.exit()	
		elif opt in ("-i", "--inputf"):
			inputf = arg
		elif opt in ("-o", "--outputf"):
			outputf = arg

	"""Required arguments"""
	if not inputf:
		usage()
		sys.exit()
			 
	img = cv2.imread(inputf, 0)
	he = HistogramEqualization(img)
	he.equalize()

	"""Save or plot image"""	
	if outputf:
		he.save(outputf)
	else:
		he.plot()

if __name__ == "__main__":
	main()
		