import sys
import getopt
import cv2
import numpy as np
from matplotlib import pyplot as plt

class HistogramEqualization:
	def __init__(self, img):
		"""Create threshold image"""
		self.original_image = Image(img, None, None, None)
		self.equalized_image = Image(img, None, None, None)
		self.rows,self.cols = self.equalized_image.shape()
		self.histogram = Histogram(self.equalized_image.matrix)
							
	def equalize(self):
		print 'Use histogram to equalize image'

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

	def label_pixel(self, pixel):
		"""Label a specific pixel"""
		self.matrix.itemset((pixel.row,pixel.col), pixel.label)

	def plot(self):
		plt.imshow(self.matrix, cmap = 'gray', interpolation = 'nearest')
		plt.xticks([]), plt.yticks([])
		plt.show()

	def save(self, output_file):
		plt.imshow(self.matrix, cmap = 'gray', interpolation = 'nearest')
		plt.xticks([]), plt.yticks([])
		plt.savefig(output_file, bbox_inches='tight')		
	

class Pixel:
	def __init__(self, label, row, col):
		"""Create a pixel with label and coordinates"""
		self.label = label
		self.row = row
		self.col = col

	def is_label(self, label):
		"""Test if pixel has a label"""
		if self.label == label:
			return True
		else:
			return False

	def is_not_label(self, label):
		"""Test if pixel doesn't have a label"""
		if self.label != label:
			return True
		else:
			return False


class Histogram:
	def __init__(self, img):
		self.hist = cv2.calcHist([img],[0],None,[256],[0,256])
		self.cumsum = self.hist.cumsum()
		self.cumsum_max = self.cumsum.max()

	def cdf(self, i):
		return self.cumsum[i]/self.cumsum_max				

						
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
		