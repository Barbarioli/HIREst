"""This module defines core subroutines used in a number 
of L2C techniques.
"""
from bitstring import BitArray
from timeit import default_timer as timer
import gzip
import os
import numpy as np
import shutil
import pickle
import scipy.stats
import matplotlib.pyplot as plt
import bz2
from PIL import Image, ImageOps
#plt.style.use('fivethirtyeight')
plt.rcParams["figure.figsize"] = (12,5)

class BitPackedStruct(object):
	'''Implements a data structure that holds a bitpacked integer matrix
	'''

	def __init__(self, bit_length, order, dims, \
					   bit_array = None, additional_stats={}):
		'''Constructs a bit-packed encoding object
				bit_array: data
				bit_length: number of bits per data item
				dims: the dimensions of the array
				order: 'row major' or 'column major' ordering
		'''
		
		self.bit_length = bit_length
		self.order = order
		self.dims = dims
		self.additional_stats = additional_stats
		self.cardinality = np.prod(self.dims)

		if not bit_array is None:
			self.data = bit_array
			self.sizeInBytes = len(self.data)//8
		else:
			self.data = None
			self.sizeInBytes = 0



	def flush(self, file):
		'''Flushes the bitpacked encoding to a file and frees
		   working memory.
		'''
		start = timer()
		self.data.tofile(open(file, 'wb'))
		del self.data #free
		self.data = None

		self.additional_stats['size_on_disk'] = os.path.getsize(file)
		self.additional_stats['flush_time'] = timer() - start


	def flushz(self, file):
		'''Flushes the bitpacked encoding to a file and frees
		   working memory. 

		   Applies gzip to the flushed file
		'''
		start = timer()

		self.data.tofile(open(file+'.tmp', 'wb'))
		del self.data #free

		with open(file+'.tmp', 'rb') as f_in, gzip.open(file, 'wb') as f_out:
			f_out.writelines(f_in)

		os.remove(file+'.tmp')

		self.additional_stats['size_on_disk'] = os.path.getsize(file)
		self.additional_stats['flush_time'] = timer() - start


	def flushbz2(self, file):
		'''Flushes the bitpacked encoding to a file and frees
		   working memory. 

		   Applies bz2 to the flushed file
		'''
		start = timer()

		self.data.tofile(open(file+'.tmp', 'wb'))
		del self.data #free

		with open(file+'.tmp', 'rb') as f_in, bz2.open(file, 'wb') as f_out:
			f_out.writelines(f_in)

		os.remove(file+'.tmp')

		self.additional_stats['size_on_disk'] = os.path.getsize(file)
		self.additional_stats['flush_time'] = timer() - start



	def load(self, file):
		'''Load takes a bitpacked binary file and returns 
		   an nd numpy array
		'''
		start = timer()

		with open(file, 'rb') as fin:
			byte_array = fin.read()
			a = BitArray(byte_array)

		decoding_array = np.zeros((self.cardinality, 1))

		for k, i in enumerate(range(self.cardinality)):
			bit_slice = a[i*self.bit_length:(i+1)*self.bit_length]
			datum = bit_slice.unpack('uint:'+str(self.bit_length))
			decoding_array[i] = datum[0]

		self.additional_stats['load_time'] = timer() - start

		return decoding_array.reshape(self.dims[0],self.dims[1], order=self.order)



	def loadz(self, file):
		'''Load takes a bitpacked binary (gziped) file and returns 
		   an nd numpy array
		'''
		start = timer()

		with gzip.open(file, 'rb') as fin:
			byte_array = fin.read()
			a = BitArray(byte_array)

		decoding_array = np.zeros((self.cardinality, 1))

		for k, i in enumerate(range(self.cardinality)):
			bit_slice = a[i*self.bit_length:(i+1)*self.bit_length]
			datum = bit_slice.unpack('uint:'+str(self.bit_length))
			decoding_array[i] = datum[0]

		
		self.additional_stats['load_time'] = timer() - start

		return decoding_array.reshape(self.dims[0],self.dims[1], order=self.order)


	def loadbz2(self, file):
		'''Load takes a bitpacked binary (bziped) file and returns 
		   an nd numpy array
		'''
		start = timer()

		with bz2.open(file, 'rb') as fin:
			byte_array = fin.read()
			a = BitArray(byte_array)

		decoding_array = np.zeros((self.cardinality, 1))

		for k, i in enumerate(range(self.cardinality)):
			bit_slice = a[i*self.bit_length:(i+1)*self.bit_length]
			datum = bit_slice.unpack('uint:'+str(self.bit_length))
			decoding_array[i] = datum[0]

		
		self.additional_stats['load_time'] = timer() - start

		return decoding_array.reshape(self.dims[0],self.dims[1], order=self.order)


	#flushes the meta data to the folder
	def flushmeta(self, metafile):
		pickle.dump([self.bit_length, self.order, self.dims], open(metafile,'wb'))


	#flushes the meta data to the folder
	def loadmeta(metafile):
		bit_length, order, dims = pickle.load(open(metafile,'rb'))
		return BitPackedStruct(bit_length, order, dims)


'''The abstract class for all compression algorithms
'''
class CompressionAlgorithm:

	def __init__(self, target, error_thresh=0.005):
		''' Constructs a basic compression algorithm
			Target is a folder that contains all of the data
		'''
		self.target = target
		self.error_thresh = error_thresh
		self.compression_stats = {}

		
		#if the directory already exists
		try:
			os.mkdir(target)
		except:
			shutil.rmtree(target)
			os.mkdir(target)

		self.NORMALIZATION = target + '/normalization'
		self.CODES = target + '/codes'
		self.METADATA =  target + '/meta'
		self.DATA_FILES = [self.CODES, self.NORMALIZATION + '.npy']




	'''Loads a dataset in for compression, data is assumed
	   to be a numpy array. Normalizes the data between 0 and
	   1 automatically. 
	'''
	def load(self, data):
		start = timer()

		#store the variables
		self.data = data.copy()

		self.N,self.p = data.shape

		self.normalization = np.array([np.max(self.data), np.min(self.data), self.N, self.p])

		#print(self.normalization.shape)
		#get all attrs between 0 and 1
		# for i in range(self.p):
		# 	self.data[:,i] = (self.data[:,i] - self.normalization[1,i])/(self.normalization[0,i] - self.normalization[1,i])
		self.data = (self.data - self.normalization[1])/(self.normalization[0]-self.normalization[1])
		#print(np.count_nonzero(np.isnan(data)))
		self.NORMALIZATION += '.npy'
		np.save(self.NORMALIZATION, self.normalization)
		self.compression_stats['load_time'] = timer() - start
		self.compression_stats['original_size'] = data.size * data.itemsize


	def getSize(self):
		total = 0

		for file in self.DATA_FILES:

			try:
				total += os.path.getsize(file)
				# to check that compression ratio is computed correctly
				# print(file, os.path.getsize(file))
			except:
				continue

		return total


	def getModelSize(self):
		total = 0

		for file in self.DATA_FILES:

			try:
				if 'learned' in file:
					total += os.path.getsize(file)
			except:
				continue

		return total


	def verify(self, original, decompressed, plot = False):
		data = original.copy()
		codec = decompressed.copy()
		N,p = data.shape
		data = (data - self.normalization[1])/(self.normalization[0] - self.normalization[1])
		codec = (codec - self.normalization[1])/(self.normalization[0] - self.normalization[1])

		"""
		if plot == True:
			plt.title('Original vs reconstructed')
			plt.plot(data[:10000], alpha = 0.5, label = 'original', color = 'black')
			plt.plot(codec[:10000], alpha = 0.5, label = 'reconstructed', color = 'r')
			plt.legend()
			plt.show()
		"""
		return {'Linfty': np.max(np.abs(data - codec)), 'L1':np.mean(np.abs(data - codec))}
		

def iarray_bitpacking(codes, order='C'):
	'''Implements a bit-packed encoding for an nd-integer array
	   		codes: an nd integer array
	   		order: 'C' row-major flattening, 'F' column-major flattening
	'''
	start = timer()
	code_range = np.max(codes)#calculate the highest value
	dims = codes.shape

	#error checking
	#codes = np.abs(codes) #remove this one
	assert(np.min(codes) >= 0)
	
	#calculates the number of bits to allocate per int
	bit_length = int(np.ceil(np.log2(code_range)))

	codes = codes.flatten(order=order)

	code_array = BitArray(bit_length * len(codes)) #codes for the data

	#iterates through codes and assigns tthem to the array 
	for i, c in enumerate(codes):
		c = int(c)
		bs = bin(c)[2:].zfill(bit_length)
		bl = list(map(lambda x: int(x), bs))
		code_array[i*bit_length:(i+1)*bit_length] = bl

	return BitPackedStruct(bit_length, order, dims, code_array, {'bitpacktime': timer() - start})


def sarray_conversion(codes):
	'''Implements a sparse encoding for an nd-integer array
	   		codes: an nd integer array
	'''
	N,p = codes.shape
	buffer = []

	for i in range(N):
		for j in range(p):
			if codes[i,j] != 0.0:
				buffer.append(np.array([i,j, codes[i,j]]))

	return np.array(buffer)



def compressz(filein, fileout):
	#applies gzip to a file

	with open(filein, 'rb') as f_in, gzip.open(fileout, 'wb') as f_out:
		f_out.writelines(f_in)

	os.remove(filein)


def decompressz(filein, fileout):
	#applies gzip -d to a file

	with gzip.open(filein, 'rb') as f_in, open(fileout, 'wb') as f_out:
		f_out.writelines(f_in)

	os.remove(filein)

def entropy(labels, base=None):
	value,counts = np.unique(labels, return_counts=True)
	return scipy.stats.entropy(counts, base=2)


def delta_xform(data):
	xform = np.zeros(data.shape)
	N,p = xform.shape

	for i in range(1,N):
		for j in range(p):
			xform[i,j] = data[i,j] - data[i-1,j]

	metadata = np.zeros((1,p+1))
	metadata[0,0:p] = np.array(data[0,:])
	metadata[0,p] = np.min(xform)

	#print(xform)

	return xform - np.min(xform), metadata


def i_delta_xform(data, metadata):
	xform = data.copy()

	N,p = xform.shape
	first_row = metadata[0,0:p]
	minv = metadata[0,p]
	xform += minv
	xform[0,:] = first_row

	for i in range(1,N):
		for j in range(p):
			xform[i,j] = xform[i,j] + xform[i-1,j]

	return xform

