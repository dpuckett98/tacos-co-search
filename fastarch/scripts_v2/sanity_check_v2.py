from dataflow_enc_dec import run_MM_dataflow
from build_models import Layer
from build_hardware import Hardware, _get_tile_parameters, _get_chunk_parameters

import itertools
import random

# TODO:
# Examine the affect of sparsity on the multiplication -- at what point does bandwidth become a bottleneck? How well does the encoder/decoder help with that, and in what situations?

# running params
test_descrip = "Sanity Check Full v1"
file_name = "sanity_check_test_v1.txt"
estimating = True

# list potential hardware params
potential_num_PEs = list(range(1, 30))
potential_num_RFs_per_PE = list(range(10, 20))
potential_size_RF = list(range(10, 20))
potential_off_chip_bandwidth = list(range(50, 150))
potential_on_chip_bandwidth = list(range(1, 4))
potential_buffer_sizes = list(range(1000, 10000))
potential_dataflows = ["Output-Stationary", "A-Stationary", "B-Stationary"]
number_tile_params = 5
number_chunk_params = 5
num_to_gen = 10

# list layers to validate on
layers = []
layers.append(Layer(A_rows=1000, A_cols_B_rows=1000, B_cols=1000))
layers.append(Layer(A_rows=100, A_cols_B_rows=10000, B_cols=1000))
layers.append(Layer(A_rows=100, A_cols_B_rows=1000, B_cols=10000))
layers.append(Layer(A_rows=10000, A_cols_B_rows=100, B_cols=1000))
layers.append(Layer(A_rows=1000, A_cols_B_rows=100, B_cols=10000))
layers.append(Layer(A_rows=10000, A_cols_B_rows=1000, B_cols=100))
layers.append(Layer(A_rows=1000, A_cols_B_rows=10000, B_cols=100))

# generate pairs of params & layers
#base_keys = [p for p in itertools.product(potential_num_PEs, potential_num_RFs_per_PE, potential_size_RF, potential_off_chip_bandwidth, potential_on_chip_bandwidth, potential_buffer_sizes, potential_dataflows)]
pairs = [] # index 0 is layer, index 1 is list of params
for i in range(num_to_gen):
	num_PEs = random.choice(potential_num_PEs)
	num_RFs = random.choice(potential_num_RFs_per_PE)
	size_RF = random.choice(potential_size_RF)
	off_chip_bandwidth = random.choice(potential_off_chip_bandwidth)
	on_chip_bandwidth = random.choice(potential_on_chip_bandwidth)
	buffer_size = random.choice(potential_buffer_sizes)
	dataflow = random.choice(potential_dataflows)
	for layer in layers:
		for i in range(number_tile_params):
			t_a, t_b, t_w = _get_tile_parameters(buffer_size, layer)
			for j in range(number_chunk_params):
				c_a, c_b, c_w = _get_chunk_parameters(t_a, t_b, t_w, num_RFs, size_RF, layer)
				params = [num_PEs, num_RFs, size_RF, off_chip_bandwidth, on_chip_bandwidth, buffer_size, dataflow, t_a, t_b, t_w, c_a, c_b, c_w]
				pairs.append([layer, params])

print("Number of tests to run:", len(pairs))

# calculate results
results = [] # index 0 is [cycles, dram_accesses]; index 1 is the layer; index 2 is a list of the parameters
for idx, pair in enumerate(pairs):
	layer = pair[0]
	params = pair[1]
	cycles, dram_accesses = run_MM_dataflow(params[0], params[1], params[2], params[3], params[4], params[5], layer.A_rows, layer.A_cols_B_rows, layer.B_cols, params[6], params[7], params[8], params[9], params[10], params[11], params[12], estimate=estimating, encode=layer.encode, decode=layer.decode, orig_heads=layer.orig_head_dim, comp_heads=layer.comp_head_dim, sparsity=layer.sparsity)
	res = [cycles, dram_accesses]
	results.append([res, layer, params])

# save results
f = open(file_name, 'a')
f.write(test_descrip + "\n")
for ((cycles, dram_accesses), layer, p) in results:
	params = [str(i) for i in p]
	f.write(str(cycles) + "\t" + str(dram_accesses) + "\t" + layer.get_string_no_nl() + "\t" + params[0] + "\t" + params[1] + "\t" + params[2] + "\t" + params[3] + "\t" + params[4] + "\t" + params[5] + "\t" + params[6] + "\t" + params[7] + "\t" + params[8] + "\t" + params[9] + "\t" + params[10] + "\t" + params[11] + "\t" + params[12] + "\n")