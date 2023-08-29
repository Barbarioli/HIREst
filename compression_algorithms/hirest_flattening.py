#from curses import window
import numpy as np
from timeit import default_timer as timer
from scipy.interpolate import interp1d
import os
from core import *
import torchvision.transforms as T
import torch.nn as nt
import torch
import warnings
import PIL 
from PIL import Image, ImageOps
from scipy.stats import multivariate_normal
from flattening_algos import *
import matplotlib.pyplot as plt

PIL.Image.MAX_IMAGE_PIXELS = 933120000
warnings.filterwarnings("ignore")


#implements a univariate sketch
class HierarchicalSketch():                                                  

    def __init__(self, quantize_thresh,window_thresh, blocksize, pfn, sfn, start_level, flattening_algo='row', w=8):
        self.error_thresh = quantize_thresh
        self.coderange = np.ceil(1.0/(quantize_thresh*2))
        self.window_error = window_thresh
        self.blocksize = blocksize #must be a power of 2
        self.d = int(np.log2(blocksize))
        self.pfn = pfn
        self.sfn = sfn
        self.start_level = start_level

       

        self.w = w 
        if flattening_algo == "row":
            self.flattener = BaseFlatten()
        elif flattening_algo == "window":
            self.flattener = WindowFlatten(w=self.w)
        elif flattening_algo == "windowContiguous":
            self.flattener = WindowContiguousFlatten(w=self.w)
    
    def quantize(self, x):
        #x = x.copy()
        x = np.rint(x*self.coderange)
        return x
    
    def dequantize(self, x_quant):
        #x_quant = x_quant.copy()
        x_quant = x_quant/self.coderange
        #print(self.coderange)
        #x_quant = x_quant.astype(np.float16)
        return x_quant

    def pool_max(self, x, width):
        return np.max(x.reshape(-1,width), axis=1)
    
    def pool_mean(self, x, width):
        return np.mean(x.reshape(-1,width), axis=1)
    
    def window_error_2d(self, x, width, error, first_element = True, pool_function = False):
        w,l = x.shape[0], x.shape[1]
        #print(w, l)
        new_x = np.copy(x)
        #print(new_x)
        for i in range(0, w-width+1, int(width)):
            #print('i: ',i)
            #print(width)
            for j in range(0, l-width+1, int(width)):
                #print('j: ',j)
                if first_element:
                    #print(new_x[i,j])
                    pivot = x[i,j]

                    #print(pivot)
                    #print('pivot: ', pivot)
                else:
                    pivot  = np.mean(x[i:i+width,j:j+width])

                for k in range(i,i+width):
                    #print('k: ', k)
                    for y in range(j,j+width):
                        #print('y: ', y)
                        if np.abs(x[k,y] - pivot)< error:
                            #print(k,y)
                            new_x[k,y] = pivot
                        else:
                            continue
                    
        return new_x
    
    #image version
    def encode(self, data):
        cpy = data.copy()
        v = self.window_error_2d(cpy, self.w, self.window_error)
        return self.quantize(v)

    def decode(self, sketch, error_thresh=0):
        return self.dequantize(sketch)

    #packs all of the data into a single array
    def pack(self, sketch):
        return self.flattener.flatten(sketch)

    #unpack all of the data
    def unpack(self, array, w=2, error_thresh=0):
        return self.flattener.unflatten(array)


class MultivariateHierarchical(CompressionAlgorithm):

    '''
    The compression codec is initialized with a per
    attribute error threshold.
    '''
    def __init__(self, target,pfn = np.mean, quantize_thresh=1e-5, window_thresh = 0.0, blocksize=4096, start_level = 0, trc = False, flattening_algo='row', flatten_window=2):

        super().__init__(target, quantize_thresh)
        self.trc = trc
        self.blocksize = blocksize
        self.window_error = window_thresh
        self.start_level =start_level
        self.TURBO_CODE_PARAMETER = "20"
        self.TURBO_CODE_LOCATION = "./../Turbo-Range-Coder/turborc" 
        #self.TURBO_CODE_PARAMETER = "-20" #on my laptop run -e0 and find best solution

        self.pfn = pfn

        self.sketch = HierarchicalSketch(self.error_thresh, blocksize=blocksize,window_thresh=self.window_error, start_level = self.start_level, pfn=self.pfn, sfn='nearest', flattening_algo=flattening_algo, w=8)
        cr = self.sketch.coderange
        self.dtype = np.int8 if cr <= 256 else np.int16 if cr < 65536 else np.int32
    
    def compress(self):
        start = timer()
        
        #print(self.data.dtype)
        codes = self.sketch.pack(self.sketch.encode(self.data)).astype(self.dtype)
        trc_flag = '-' + self.TURBO_CODE_PARAMETER
        # flush to .npy file
        self.path = self.CODES + '.npy'
        # np.save(self.path, codes.flatten(order='F'))
        np.save(self.path, codes)
        self.CODES += '.rc'
        self.DATA_FILES[0] = self.CODES


        print('\n')
        
        # run TRC [compression]
        # best performing function should be the the int trc_flag after run of turborc -e0
        #subprocess.run(['./../Turbo-Range-Coder/turborc', trc_flag, self.path, self.CODES])
        command = " ".join(['./../Turbo-Range-Coder/turborc', trc_flag, self.path, self.CODES])
        os.system(command)

        self.compression_stats['compression_latency'] = timer() - start
        self.compression_stats['compressed_size'] = self.getSize()
        self.compression_stats['compressed_ratio'] = self.getSize()/self.compression_stats['original_size']
        #self.compression_stats.update(struct.additional_stats)

    def decompress(self, original=None, error_thresh=1e-4):
        start = timer()
        
        #subprocess.run(['./../Turbo-Range-Coder/turborc', '-d', self.CODES, self.path])
        command = " ".join([self.TURBO_CODE_LOCATION, "-d", self.CODES, self.path])
        os.system(command)

        normalization = np.load(self.NORMALIZATION)
        #self.normalization = np.array([np.max(self.data), np.min(self.data), self.N, self.p])

        # N, p = self.normalization[2], self.normalization[3]
        
        codes = self.sketch.unpack(self.sketch.decode(np.load(self.path)))

        codes = codes*(normalization[0] - normalization[1]) + normalization[1]


        self.compression_stats['decompression_latency'] = timer() - start
        if not original is None:
            self.compression_stats['errors'] = self.verify(original, codes)

        return codes

####
"""
Test code here
"""
####

""""""

# packing algo experiments
# if __name__ == "__main__":
#     # for w in [2,4,8,16,32]:
#     for w in [8]:
#         ratios, latencies, = [], []
#         for data in os.listdir('data'):
#             if data.endswith('.f32'):
#                 file = np.fromfile('data/' + data, dtype=float)
#                 file = file.reshape(1800, 1800)
#                 data = file[0:1024,0:1024]
#                 shape = 1024
#                 ratios_1ds = []
#                 latencies_1ds = []
#                 for flattener_type in ['row', 'window', 'windowContiguous']:
#                     nn = MultivariateHierarchical('hier', quantize_thresh = 0.005,window_thresh = 0.005, blocksize=shape, start_level = 10, trc = True, flattening_algo=flattener_type, flatten_window=w)
#                     nn.load(data)
#                     nn.compress()
#                     ratios_1ds.append(nn.compression_stats['compressed_ratio'])
#                     latencies_1ds.append(nn.compression_stats['compression_latency'])
#                 ratios.append(ratios_1ds)
#                 latencies.append(latencies_1ds)

#         ratios = np.array(ratios)
#         latencies = np.array(latencies)
#         np.savetxt("packing_algo_results/ratios_{}.csv".format(w), ratios, delimiter=',', fmt='%f')
#         np.savetxt("packing_algo_results/latencies_{}.csv".format(w), latencies, delimiter=',', fmt='%f')   

if __name__ == "__main__":
    file = np.fromfile('data/CLDLOW_1_1800_3600.f32', dtype=float)
    file = file.reshape(1800, 1800)
    data = file[0:1024,0:1024]

    shape = 1024
    for flattener_type in ['row', 'window', 'windowContiguous']:
        print("Flattner type: ", flattener_type)
        nn = MultivariateHierarchical('hier', quantize_thresh = 0.005, window_thresh = 0.005, blocksize=shape, start_level = 10, trc = True, flattening_algo=flattener_type, flatten_window=8)
        nn.load(data)
        nn.compress()
        nn.decompress(data) 
        print(nn.compression_stats)

    # nn = MultivariateHierarchical('hier', quantize_thresh = 0.005, window_thresh = 0.005, blocksize=shape, start_level = 10, trc = True, flattening_algo='row', flatten_window=8)
    # nn.load(data)
    # nn.compress()
    # nn.decompress(data) 
    # print(nn.compression_stats)