import numpy as np
import os
from timeit import default_timer as timer
import math
import struct
from core import *

import warnings
import PIL 
from PIL import Image, ImageOps
PIL.Image.MAX_IMAGE_PIXELS = 933120000
warnings.filterwarnings("ignore")

class Identity(CompressionAlgorithm):
	'''Baseline lossless compression
	'''


	'''
	The compression codec is initialized with a per
	attribute error threshold.
	'''
	def __init__(self, target, error_thresh=0.005):

		super().__init__(target, error_thresh)

		self.RAW = self.target + '/data'
		self.DATA_FILES += [self.RAW + '.npy']


	"""The main compression loop
	"""
	def compress(self):
		start = timer()

		np.save(self.RAW, self.data)

		self.compression_stats['compression_latency'] = timer() - start
		self.compression_stats['compressed_size'] = self.getSize()
		self.compression_stats['compressed_ratio'] = self.getSize()/self.compression_stats['original_size']

		

	def decompress(self, original=None):

		start = timer()

		codes = np.load(self.RAW + '.npy')

		normalization = np.load(self.NORMALIZATION)
		_, P2 = normalization.shape

		p = int(P2 - 1)
		N = int(normalization[0,p])

		for i in range(p):
			codes[:,i] = (codes[:,i] + normalization[1,i])*(normalization[0,i] - normalization[1,i])


		self.compression_stats['decompression_latency'] = timer() - start

		if not original is None:
			self.compression_stats['errors'] = self.verify(original, codes)

		return codes



class IdentityGZ(CompressionAlgorithm):
	'''Baseline lossless compression
	'''


	'''
	The compression codec is initialized with a per
	attribute error threshold.
	'''
	def __init__(self, target, error_thresh=0.005):

		super().__init__(target, error_thresh)

		self.RAW = self.target + '/data'
		self.DATA_FILES += [self.RAW+ '.npyz']


	"""The main compression loop
	"""
	def compress(self):
		start = timer()

		np.save(self.RAW, self.data)
		compressz(self.RAW + '.npy', self.RAW+'.npyz')

		self.compression_stats['compression_latency'] = timer() - start
		self.compression_stats['compressed_size'] = self.getSize()
		self.compression_stats['compressed_ratio'] = self.getSize()/self.compression_stats['original_size']

		

	def decompress(self, original=None):

		start = timer()

		decompressz(self.RAW + '.npyz', self.RAW+'.npy')
		codes = np.load(self.RAW + '.npy')

		normalization = np.load(self.NORMALIZATION)
		_, P2 = normalization.shape

		p = int(P2 - 1)
		N = int(normalization[0,p])

		for i in range(p):
			codes[:,i] = (codes[:,i] + normalization[1,i])*(normalization[0,i] - normalization[1,i])


		self.compression_stats['decompression_latency'] = timer() - start

		if not original is None:
			self.compression_stats['errors'] = self.verify(original, codes)

		return codes


class BitStripGZ(CompressionAlgorithm):
	'''Baseline lossy compression
	'''


	'''
	The compression codec is initialized with a per
	attribute error threshold.
	'''
	def __init__(self, target, error_thresh=0.005):

		super().__init__(target, error_thresh)

		self.RAW = self.target + '/data'
		self.DATA_FILES += [self.RAW+ '.npyz']


	"""The main compression loop
	"""
	def compress(self):
		start = timer()

		N, _ = self.data.shape

		for i in range(N):
			self.data[i,:] = self._strip_code(self.data[i,:])

		np.save(self.RAW, self.data)
		compressz(self.RAW + '.npy', self.RAW+'.npyz')

		self.compression_stats['compression_latency'] = timer() - start
		self.compression_stats['compressed_size'] = self.getSize()
		self.compression_stats['compressed_ratio'] = self.getSize()/self.compression_stats['original_size']

		

	def decompress(self, original=None):

		start = timer()

		decompressz(self.RAW + '.npyz', self.RAW+'.npy')
		codes = np.load(self.RAW + '.npy')

		normalization = np.load(self.NORMALIZATION)
		_, P2 = normalization.shape

		p = int(P2 - 1)
		N = int(normalization[0,p])

		for i in range(p):
			codes[:,i] = (codes[:,i] + normalization[1,i])*(normalization[0,i] - normalization[1,i])


		self.compression_stats['decompression_latency'] = timer() - start

		if not original is None:
			self.compression_stats['errors'] = self.verify(original, codes)

		return codes


	#zero out as many bits as possible
	def _strip_code(self, vector):
		p = vector.shape[0]

		for i in range(p): #go component by component
			value = vector[i]
			ba = bytearray(struct.pack("d", value))

			for j in range(len(ba)):
				tmp = ba[j]
				ba[j] = int('00000000')
				newvalue = struct.unpack("d", ba)[0]

				if np.abs(newvalue - value) > self.error_thresh:
					ba[j] = tmp
					vector[i] = struct.unpack("d", ba)[0]
					break

		return vector
		

####
"""
Test code here
"""
####


#data = np.loadtxt('/Users/sanjaykrishnan/Downloads/test_comp/ColorHistogram.asc')[:,1:]
#normalize this data

# file = np.fromfile('CLDLOW_1_1800_3600.f32', dtype=float)
# file = file.reshape(1800, 1800)
# data = file[0:1024,0:1024]

img = Image.open('chicago.tif')
img = ImageOps.grayscale(img)
data = np.array(img)
data = data.astype(np.float32)
data = data[:1024,:1024]

N,p = data.shape
nn = BitStripGZ('quantize')
nn.load(data)
nn.compress()
nn.decompress(data)
print(nn.compression_stats)

