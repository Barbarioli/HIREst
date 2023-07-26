import numpy as np
import os
from timeit import default_timer as timer
import math
from core import *

import warnings
import PIL 
from PIL import Image, ImageOps
PIL.Image.MAX_IMAGE_PIXELS = 933120000
warnings.filterwarnings("ignore")

class EntropyCoding(CompressionAlgorithm):
	'''Applies a quantization + entropy coding to 
	   compress a dataset similar to Squish.
	'''


	'''
	The compression codec is initialized with a per
	attribute error threshold.
	'''
	def __init__(self, target, error_thresh=0.005):

		super().__init__(target, error_thresh)

		self.coderange = int(math.ceil(1.0/error_thresh))

		self.TURBO_CODE_LOCATION = "./../Turbo-Range-Coder/turborc" 
		self.TURBO_CODE_PARAMETER = "-20" #on my laptop run -e0 and find best solution


	"""The main compression loop
	"""
	def compress(self):
		start = timer()

		codes = np.ones((self.N, self.p))*-1#set all to negative one

		for i in range(self.N):
			for j in range(self.p):
				codes[i,j] = int(self.data[i,j]*self.coderange)

		codes = codes.astype(np.intc).flatten(order='F') #set as a c-integer type
		fname = self.CODES
		np.save(fname, codes)

		command = " ".join([self.TURBO_CODE_LOCATION, self.TURBO_CODE_PARAMETER, fname+".npy", fname+".npy"])

		os.system(command)

		self.DATA_FILES += [fname+".npy.rc"]


		self.compression_stats['compression_latency'] = timer() - start
		self.compression_stats['compressed_size'] = self.getSize()
		self.compression_stats['compressed_ratio'] = self.getSize()/self.compression_stats['original_size']

		

	def decompress(self, original=None):

		start = timer()

		command = " ".join([self.TURBO_CODE_LOCATION, "-d", self.CODES+".npy.rc", self.CODES+".npy"])
		os.system(command)
		codes = np.load(self.CODES+".npy", allow_pickle=False)
		
		#unpack_time = timer() - start
		#print('unpack time: ', unpack_time)

		normalization = np.load(self.NORMALIZATION)
		_, P2 = normalization.shape

		p = int(P2 - 1)
		N = int(normalization[0,p])

		codes = codes.reshape(N,p, order='F').astype(np.float64)
		coderange = np.max(codes)

		for i in range(p):
			codes[:,i] = (codes[:,i]/coderange)*(normalization[0,i] - normalization[1,i]) + normalization[1,i]


		self.compression_stats['decompression_latency'] = timer() - start

		if not original is None:
			self.compression_stats['errors'] = self.verify(original, codes)

		return codes



####
"""
Test code here
"""
####

#data = np.loadtxt('/Users/sanjaykrishnan/Downloads/HT_Sensor_UCIsubmission/HT_Sensor_dataset.dat')[:2000,1:]
"""
data = np.load('/Users/sanjaykrishnan/Downloads/ts_compression/l2c/data/electricity.npy')
print(data.shape)
#data = np.nan_to_num(data)
#normalize this data
N,p = data.shape
nn = EntropyCoding('quantize')
nn.load(data)
nn.compress()
nn.decompress(data)
print(nn.compression_stats)
"""
# data = np.load('/Users/gabemersy/Desktop/ts_compression/l2c/data/solar.npy')
# data = np.load('/Users/gabemersy/Desktop/bitcoin.npy')
# data = np.load('/Users/gabemersy/Desktop/data/solar.npy')
# data = np.load('/Users/gabemersy/Desktop/data/exchange_rate.npy')
# data = np.load('/Users/gabemersy/Desktop/data/taxi.npy')
# data = np.load('/Users/gabemersy/Desktop/data/exchange_rate.npy')
# data = np.load('/Users/gabemersy/Desktop/ts_compression/synthetic_TS_gen/MA1/MA_1_corrcoef_0.97.npy')
# data = np.load('/Users/gabemersy/Desktop/ts_compression/synthetic_TS_gen/MA1/MA_1_corrcoef_0.19.npy')
# # data = np.load('/Users/gabemersy/Desktop/ts_compression/l2c/data/exchange_rate.npy')
# data = np.load('/Users/gabemersy/Desktop/ts_compression/synthetic_TS_gen/trios/sigma2_10.npy')
# data = np.load('/Users/brunobarbarioli/Documents/Research/ts_compression/l2c/data/gas.npy')
# data = np.load('/Users/brunobarbarioli/Documents/Research/ts_compression/l2c/data/house.npy')
# data = np.load('/Users/brunobarbarioli/Documents/Research/ts_compression/l2c/data/gas.npy')
#data = np.load('/Users/brunobarbarioli/Documents/Research/ts_compression/l2c/data/crop_images.npy', allow_pickle = True)
#data = np.load('/Users/brunobarbarioli/Documents/Research/ts_compression/l2c/data/sensor.npy')

#data = np.load('/Users/brunobarbarioli/Documents/Research/ts_compression/l2c/data/watch.npy')[:4096*64,-1:]

"""
nn = EntropyCoding('quantize', error_thresh = 0.001)
print(nn.TURBO_CODE_PARAMETER)
nn.load(data)
nn.compress()
nn.decompress(data)
print(nn.compression_stats)
"""
# data = np.load('/Users/gabemersy/Desktop/data/gas.npy')
# data = np.load('/Users/gabemersy/Desktop/data/house.npy')
# data = np.load('/Users/gabemersy/Desktop/data/gas.npy')
# data = np.load('/Users/gabemersy/Desktop/data/house.npy')
# data = np.load('/Users/gabemersy/Desktop/data/gas.npy')
# data = np.load('/Users/gabemersy/Desktop/ts_compression/synthetic_TS_gen/trios/sigma2_10.npy')
#data = np.load('/Users/brunobarbarioli/Documents/Research/ts_compression/l2c/data/phones_accelerometer.npy')[:4096*64]
#data = np.load('/Users/brunobarbarioli/Documents/Research/ts_compression/l2c/data_old/sensor.npy')[:4096*64,:]
#data = np.load('/Users/brunobarbarioli/Documents/Research/ts_compression/l2c/data/phones_accelerometer.npy')[:4096*64,1:2]
#data = np.load('/Users/brunobarbarioli/Documents/Research/ts_compression/l2c/data_old/gas.npy')[:4096*64,0:1]

# file = np.fromfile('CLDLOW_1_1800_3600.f32', dtype=float)
# file = file.reshape(1800, 1800)
# data = file[0:1024,0:1024]

# img = Image.open('chicago.tif')
# img = ImageOps.grayscale(img)
# data = np.array(img)
# data = data.astype(np.float32)
# data = data[:1024,:1024]

data = np.load('park_full.npy')
print('original shape:', data.shape)
data = data[0:1024,0:1024]
print('new shape:', data.shape)

N,p = data.shape
nn = EntropyCoding('quantize', error_thresh = 0.1)
nn.load(data)
nn.compress()
#nn.decompress(data)
print(nn.compression_stats)
