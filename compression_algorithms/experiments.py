from identity import *
#from quantize import Quantize, QuantizeGZ
from hirest_experiments import *
from quantize_bit import QuantizeGZ_bit
import numpy as np

error_thresholds = [0.15,0.1,0.05,0.01,0.005,0.001]

#ERROR_THRESHOLD = 0.00499

def initialize(ERROR_THRESH):
	#set up baslines
	BASELINES = []
	BASELINES.append(MultivariateHierarchical('rs', quantize_thresh = 0.875*ERROR_THRESH,window_thresh = 0.125*ERROR_THRESH, blocksize=1024, start_level = 10, trc = True, flattening_algo="row", w=8, pivot_selection='window'))
	BASELINES.append(MultivariateHierarchical('qtrc', quantize_thresh = ERROR_THRESH,window_thresh = 0, blocksize=1024, start_level = 10, trc = True, flattening_algo="row", w=8, pivot_selection='window'))
	BASELINES.append(IdentityGZ('gz', error_thresh=ERROR_THRESH))
	BASELINES.append(QuantizeGZ_bit('q+gz', error_thresh=ERROR_THRESH))
	return BASELINES

def run(BASELINES,\
		DATA_DIRECTORY = './../data/', \
		FILENAME = 'affordable_rentals_final_1024.npy',\
		):
	data = np.load(DATA_DIRECTORY + FILENAME, allow_pickle = True)

	bresults = {}
	for nn in BASELINES:
		#print(data.shape)
		nn.load(data)
		nn.compress()
		nn.decompress(data)
		bresults[nn.target] = nn.compression_stats

	return bresults

#plotting

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams["figure.figsize"] = (10,4)

for t in error_thresholds:

	ERROR_THRESHOLD = t
	print(t)
	BASELINES = initialize(t)
	FILENAME = 'ar.npy'
	#SIZE_LIMIT = 1000
	bresults = run(BASELINES)


	#compressed size

	try:
		with open('/home/nuc/HIREst/compression_algorithms/results/results_compression_r_' + FILENAME.split('.')[0] + '.txt', 'x') as f:
			f.write('[')
			[f.write(str(bresults[k]['compressed_ratio'])+',') if k != 'apca' else f.write(str(bresults[k]['compressed_ratio'])) for k in bresults]
			f.write(']')
			f.write(',')
			f.close()

	except:
		with open('/home/nuc/HIREst/compression_algorithms/results/results_compression_r_' +FILENAME.split('.')[0] + '.txt','a') as f:
			f.write('[')
			[f.write(str(bresults[k]['compressed_ratio'])+',') if k != 'apca' else f.write(str(bresults[k]['compressed_ratio'])) for k in bresults]
			f.write(']')
			f.write(',')
			f.close()

	#decompression throughput (subtract bitpacking time)
	try:
		with open('/home/nuc/HIREst/compression_algorithms/results/results_compression_l_' + FILENAME.split('.')[0] + '.txt', 'x') as f:
			f.write('[')
			[f.write(str(bresults[k]['compression_latency'])+',') if k != 'apca' else f.write(str(bresults[k]['compressed_ratio'])) for k in bresults]
			f.write(']')
			f.write(',')
			f.close()
	except:
		with open('/home/nuc/HIREst/compression_algorithms/results/results_compression_l_' +FILENAME.split('.')[0] + '.txt','a') as f:
			f.write('[')
			[f.write(str(bresults[k]['compression_latency'])+',') if k != 'apca' else f.write(str(bresults[k]['compressed_ratio'])) for k in bresults]
			f.write(']')
			f.write(',')
			f.close()

	try:
		with open('/home/nuc/HIREst/compression_algorithms/results/results_decompression_l_' + FILENAME.split('.')[0] + '.txt', 'x') as f:
			f.write('[')
			[f.write(str(bresults[k]['decompression_latency'])+',') if k != 'apca' else f.write(str(bresults[k]['compressed_ratio'])) for k in bresults]
			f.write(']')
			f.write(',')
			f.close()
	except:
		with open('/home/nuc/HIREst/compression_algorithms/results/results_decompression_l_' +FILENAME.split('.')[0] + '.txt','a') as f:
			f.write('[')
			[f.write(str(bresults[k]['decompression_latency'])+',') if k != 'apca' else f.write(str(bresults[k]['compressed_ratio'])) for k in bresults]
			f.write(']')
			f.write(',')
			f.close()
