import sys
import getopt
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

class HistogramEqualization:
	def __init__(self, img):
		"""Create image and histogram"""
		self.equalized_image = Image(matrix=img)
		self.rows,self.cols = self.equalized_image.shape()
		self.max_val = 256
		self.histogram = Histogram(self.equalized_image.matrix, self.max_val)
							
	def equalize(self):
		"""Equalize image"""
		for i in xrange(self.rows):
			for j in xrange(self.cols):
				"""Get current pixel and set color to equalized color"""
				current = self.equalized_image.get_pixel(i,j)
				current.color = self.histogram.new_val(current.color)
				self.equalized_image.color_pixel(current)

	def plot(self):
		"""Plot equalized_image"""
		self.equalized_image.plot()

	def save(self, output_file):
		"""Save equalized image"""
		self.equalized_image.save(output_file)

	def transfer_function(self, output_file):
		self.histogram.transfer_function(output_file)

	def save_text(self):
		"""Save matrix in csv file for debugging"""
		np.savetxt("out.csv", self.equalized_image.matrix, delimiter=",", fmt="%d")	
	

class Image:
	def __init__(self, **kwargs):
		"""Save or create matrix"""
		if kwargs.has_key("matrix"):
			self.matrix = kwargs["matrix"]
		elif kwargs.has_key("rows")	and kwargs.has_key("cols"):
			self.matrix = np.zeros((kwargs["rows"], kwargs["cols"]), dtype=np.int)

		if kwargs.has_key("background"):
			self.background = kwargs["background"]	

	def shape(self):
		"""Return number of rows and columns"""
		return self.matrix.shape	

	def get_pixel(self, row, col):
		"""Return a specific pixel"""
		return Pixel(self.matrix.item(row,col), row, col)

	def color_pixel(self, pixel):
		"""Color a specific pixel"""
		self.matrix.itemset((pixel.row,pixel.col), pixel.color)

	def plot(self):
		"""Plot matrix"""
		plt.imshow(self.matrix, cmap = "gray", interpolation = "nearest")
		plt.xticks([]), plt.yticks([])
		plt.show()

	def save(self, output_file):
		"""Save matrix"""
		plt.imshow(self.matrix, cmap = "gray", interpolation = "nearest")
		plt.xticks([]), plt.yticks([])
		plt.savefig(output_file, bbox_inches="tight")		
	

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
		"""Create histogram"""
		self.max_val = max_val
		self.hist = cv2.calcHist([img],[0],None,[self.max_val],[0,self.max_val])
		"""Save cummulative sum and it's maximum"""
		self.cumsum = self.hist.cumsum()
		self.cumsum_max = self.cumsum.max()

	def cdf(self, i):
		"""Return value of cummulative distribution function at i"""
		return self.cumsum[i]/self.cumsum_max

	def new_val(self, i):
		"""Use cdf to calculate new gray level"""
		return math.floor((self.max_val - 1)*self.cdf(i))

	def transfer_function(self, output_file):
		"""Create CDF array then plot"""
		transfer = self.cumsum / self.cumsum_max
		plt.plot(transfer)
		plt.xlim(xmax=self.max_val-1)
		plt.ylim(ymax=1.2)
		plt.xlabel("Gray Value")
		plt.ylabel("CDF")
		plt.title("Transfer Function")
		plt.savefig(output_file, bbox_inches="tight")
		plt.clf()	

						
def main():
	def usage():
		print "python histogram_equalization.py -i <inputf> [-o <outputf> -t <transferf>]"

	inputf = None
	outputf = None
	transferf = None

	"""Process command line inputs"""
	try:
		opts, args = getopt.getopt(sys.argv[1:], "hi:o:t:", ["inputf=", "outputf=", 
								 "transferf="])
	except getopt.GetoptError:
		usage()
		sys.exit(2)
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			usage()
			sys.exit()	
		elif opt in ("-i", "--inputf"):
			inputf = arg
		elif opt in ("-o", "--outputf"):
			outputf = arg
		elif opt in ("-t", "--transferf"):
			transferf = arg			

	"""Required arguments"""
	if not inputf:
		usage()
		sys.exit()
			 
	img = cv2.imread(inputf, 0)
	he = HistogramEqualization(img)
	he.equalize()

	"""Save transfer function plot"""
	if transferf:
		he.transfer_function(transferf)

	"""Save or plot image"""	
	if outputf:
		he.save(outputf)
	else:
		he.plot()

if __name__ == "__main__":
	main()
		