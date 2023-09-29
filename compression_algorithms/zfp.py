import zfpy
import numpy as np
import os
import sys
from timeit import default_timer as timer

path = './../data/array_of_things_final_1024.npy'
file = np.load(path)
original_size = os.path.getsize(path)
print(original_size)

#confirm lossless compression/decompression
start = timer()
compressed_data = zfpy.compress_numpy(file, tolerance=0.001)
print('compression latency: ', timer()- start)
print('compression ratio: ', sys.getsizeof(compressed_data)/original_size)
start_d = timer()
decompressed_array = zfpy.decompress_numpy(compressed_data)
print('decompression latency: ',  timer()- start_d)
