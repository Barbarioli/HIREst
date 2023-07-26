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

PIL.Image.MAX_IMAGE_PIXELS = 933120000
warnings.filterwarnings("ignore")

class Quantize(CompressionAlgorithm):
	'''Applies a basic quantization algorithm to compress the data
	'''


	'''
	The compression codec is initialized with a per
	attribute error threshold.
	'''
	def __init__(self, target, error_thresh=0.005):

		super().__init__(target, error_thresh)

		self.coderange = int(math.ceil(1.0/error_thresh))


	"""The main compression loop
	"""
	def compress(self):
		start = timer()

		codes = np.ones((self.N, self.p))*-1#set all to negative one

		for i in range(self.N):
			for j in range(self.p):
				codes[i,j] = int(self.data[i,j]*self.coderange)

		#bit_start = timer()
		struct = iarray_bitpacking(codes)
		#bit_total = timer() - bit_start
		struct.flush(self.CODES)
		struct.flushmeta(self.METADATA)

		self.compression_stats['compression_latency'] = timer() - start
		self.compression_stats['compressed_size'] = self.getSize()
		self.compression_stats['compressed_ratio'] = self.getSize()/self.compression_stats['original_size']

		

	def decompress(self, original=None):

		start = timer()

		struct = BitPackedStruct.loadmeta(self.METADATA)
		codes = struct.load(self.CODES)
		coderange = np.max(codes)

		normalization = np.load(self.NORMALIZATION)
		_, P2 = normalization.shape

		p = int(P2 - 1)
		N = int(normalization[0,p])
		bit_length = struct.bit_length

		for i in range(p):
			codes[:,i] = (codes[:,i]/coderange + normalization[1,i])*(normalization[0,i] - normalization[1,i])

		self.compression_stats['decompression_latency'] = timer() - start

		if not original is None:
			self.compression_stats['errors'] = self.verify(original, codes)

		return codes



class QuantizeGZ(CompressionAlgorithm):
	'''Applies a basic quantization algorithm to compress the data
	'''


	'''
	The compression codec is initialized with a per
	attribute error threshold.
	'''
	def __init__(self, target, error_thresh=0.005):

		super().__init__(target, error_thresh)

		self.coderange = int(math.ceil(1.0/error_thresh))

	def quantize(self, x):
		x = x*self.coderange
		return x
		
	def dequantize(self, x_quant):
		x_quant = x_quant/self.coderange
		return x_quant

	"""The main compression loop
	"""
	def compress(self):
		start = timer()

		codes = np.ones((self.N, self.p))*-1#set all to negative one

		#for i in range(self.N):
		#	for j in range(self.p):
		#		codes[i,j] = int(self.data[i,j]*self.coderange)
		codes = self.quantize(self.data)

		#bit_start = timer()
		struct = iarray_bitpacking(codes)
		#bit_total = timer() - bit_start
		struct.flushz(self.CODES)
		struct.flushmeta(self.METADATA)

		self.compression_stats['compression_latency'] = timer() - start
		self.compression_stats['compressed_size'] = self.getSize()
		self.compression_stats['compressed_ratio'] = self.getSize()/self.compression_stats['original_size']
		self.compression_stats.update(struct.additional_stats)
		

	def decompress(self, original=None):

		start = timer()

		struct = BitPackedStruct.loadmeta(self.METADATA)
		codes = struct.loadz(self.CODES)
		coderange = np.max(codes)

		normalization = np.load(self.NORMALIZATION)
		_, P2 = normalization.shape

		p = int(P2 - 1)
		N = int(normalization[0,p])
		#bit_length = struct.bit_length

		# for i in range(p):
		# 	codes[:,i] = (codes[:,i]/coderange + normalization[1,i])*(normalization[0,i] - normalization[1,i])
		codes = self.dequantize(codes)
		for i in range(p):
			codes[:,i] = (codes[:,i])*(normalization[0,i] - normalization[1,i]) + normalization[1,i]

		self.compression_stats['decompression_latency'] = timer() - start

		if not original is None:
			self.compression_stats['errors'] = self.verify(original, codes)


		self.compression_stats.update(struct.additional_stats)

		return codes





####
"""
Test code here
"""
####
# '/Users/brunobarbarioli/Documents/Research/learning-to-compress-master/l2c/data/electricity_nips/electricity.npy'
# '/Users/brunobarbarioli/Documents/Research/learning-to-compress-master/l2c/data/exchange_rate_nips/exchange_rate.npy'
# '/Users/brunobarbarioli/Documents/Research/learning-to-compress-master/l2c/data/solar_nips/solar.npy'
# '/Users/brunobarbarioli/Documents/Research/learning-to-compress-master/l2c/data/taxi_30min/taxi.npy'
# '/Users/brunobarbarioli/Documents/Research/learning-to-compress-master/l2c/data/traffic_nips/traffic.npy'
# data = np.load('/Users/brunobarbarioli/Documents/Research/learning-to-compress-master/l2c/data/wiki-rolling_nips/wiki.npy')
# data = np.load('/Users/gabemersy/Desktop/ts_compression/l2c/data/solar.npy')
# data = np.load('/Users/brunobarbarioli/Documents/Research/ts_compression/l2c/data/house.npy')
#data = np.load('/Users/brunobarbarioli/Documents/Research/ts_compression/l2c/data/sensor.npy')
# #normalize this data
#N,p = data.shape

# file = np.fromfile('CLDLOW_1_1800_3600.f32', dtype=float)
# file = file.reshape(1800, 1800)
# data = file[0:1024,0:1024]

#data = np.load('/Users/brunobarbarioli/Documents/Research/ts_compression/l2c/data/phones_accelerometer.npy')[:4096*24,:]

img = Image.open('chicago.tif')
img = ImageOps.grayscale(img)
data = np.array(img)
data = data.astype(np.float32)
data = data[:1024,:1024]

nn = QuantizeGZ('Q', 0.001)
nn.load(data)
nn.compress()
nn.decompress(data)
print(nn.compression_stats)


