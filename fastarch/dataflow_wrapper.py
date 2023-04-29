import math
import random
import numpy as np

from fastarch.dataflow_enc_dec import run_MM_dataflow
import fastarch.build_models_v2 as models
import fastarch.build_hardware_v2 as hw
import fastarch.dataflow_estimator as de
import fastarch.conv_helper as ch
import fastarch.dataflow_convolution as dc
import fastarch.dataflow_estimator_conv as dec

# hardware is from build_hardware.py
# layer is from build_models_v2.py
# params: [split_dim, dataflow, ta, tb, tw, ca, cb, cw]
# 	split_dim is either "rows" or "cols" and controls whether the matrix mult is divided among the PE lanes by A rows or by B cols ("None" is the typical 1 PE lane option)
# returns [cycles, power, memory idle cycles, offload cycles]
def run_layer(hardware, params, layer, preload_cycles=0, pipeline_offloading=False, generate_sparse_map=True, estimate=False, memory_initial_size=0):
	hardware.print()
	layer.print()
	print(params)
	
	# handle convs separately
	if isinstance(layer, ch.ConvLayer):
		cycles, power = dec.estimate_performance(hardware, layer, params)
		if estimate:
			return [cycles, power, -1, -1, -1]
		res = dc.run_conv_dataflow(hardware, layer, params)
		return [res[0], power, res[4], res[5], res[6]]
	
	# handle estimate
	cycles, power = de.estimate_performance(hardware, layer, params)
	if estimate:
		return [cycles, power, -1, -1, -1]
	
	# inner matrix mult
	
	# divide layers among PE lanes
	A_rows = math.ceil(layer.A_rows / hardware.num_PE_lanes) if params[0] == "rows" else layer.A_rows
	A_cols_B_rows = layer.A_cols_B_rows
	B_cols = math.ceil(layer.B_cols / hardware.num_PE_lanes) if params[0] == "cols" else layer.B_cols
	print(A_rows, B_cols, A_cols_B_rows)
	#print(A_rows * hardware.num_PE_lanes, layer.A_rows)
	#print(hardware.get_single_lane_bandwidth())
	
	# divide tiles among PE lanes
	t_a = math.ceil(params[2] / hardware.num_PE_lanes) if params[0] == "rows" else params[2]
	t_w = params[4]
	t_b = math.ceil(params[3] / hardware.num_PE_lanes) if params[0] == "cols" else params[3]
	print(t_a, t_b, t_w)
	share_load = "B" if params[0] == "rows" else "A" if params[0] == "cols" else "None"
	
	# outer matrix mult
	'''
	# divide layers among PE lanes
	A_rows = layer.A_rows #math.ceil(layer.A_rows / hardware.num_PE_lanes) if params[0] == "rows" else layer.A_rows
	A_cols_B_rows = math.ceil(layer.A_cols_B_rows / hardware.num_PE_lanes)
	B_cols = layer.B_cols #math.ceil(layer.B_cols / hardware.num_PE_lanes) if params[0] == "cols" else layer.B_cols
	
	# divide tiles among PE lanes
	t_a = params[2] #math.ceil(params[2] / hardware.num_PE_lanes) if params[0] == "rows" else params[2]
	t_w = math.ceil(params[4] / hardware.num_PE_lanes)
	t_b = params[3] #math.ceil(params[3] / hardware.num_PE_lanes) if params[0] == "cols" else params[3]
	'''
	
	# generate sparse map
	if generate_sparse_map and layer.sparsity != 0.0:
		#sparse_map = np.zeros((A_rows, B_cols))
		sparse_map = [[1 if random.random() > layer.sparsity else 0 for i in range(B_cols)] for j in range(A_rows)] # random generation
		sparse_map = np.array(sparse_map)
		# ensure it's relatively close to the right amount of sparsity
		num_iters = 0
		while abs(np.count_nonzero(sparse_map) / sparse_map.size - (1 - layer.sparsity)) > 0.01 + 0.000001 * num_iters:
			sparse_map = [[1 if random.random() > layer.sparsity else 0 for i in range(B_cols)] for j in range(A_rows)] # random generation
			sparse_map = np.array(sparse_map)
			num_iters += 1
		sparse_map[0,0] = 1
		
		# use this to account for head overlapping
		#while abs(np.count_nonzero(sparse_map) / (sparse_map.size / layer.num_heads) - (1 - layer.sparsity)) > 0.01:
		#	for row in range(A_rows):
		#		for col in range(B_cols):
		#			if row % layer.num_heads == col % layer.num_heads:
		#				sparse_map[row,col] = 1 if random.random() > layer.sparsity else 0
		#print(sparse_map)
		#print(np.count_nonzero(sparse_map) / (sparse_map.size / layer.num_heads))
		
		#print(np.count_nonzero(sparse_map) / sparse_map.size)
	else:
		sparse_map = None
	
	res = run_MM_dataflow(num_PEs=hardware.num_PEs_per_lane, num_RFs_per_PE=hardware.num_RFs_per_PE, size_RF=hardware.size_RF, off_chip_bandwidth=hardware.get_single_lane_bandwidth(), on_chip_bandwidth=hardware.on_chip_bandwidth, max_sram_size=hardware.get_single_lane_SRAM_size(), A_rows=A_rows, A_cols_B_rows=A_cols_B_rows, B_cols=B_cols, dataflow=params[1], t_a=t_a, t_b=t_b, t_w=t_w, c_a=params[5], c_b=params[6], c_w=params[7], estimate=estimate, sparsity=layer.sparsity, preload_cycles=preload_cycles, pipeline_offloading=pipeline_offloading, share_load=share_load, num_PE_lanes=hardware.num_PE_lanes, A_transfer_scale=layer.A_transfer_scale, B_transfer_scale=layer.B_transfer_scale, O_transfer_scale=layer.O_transfer_scale, num_heads=layer.num_heads, load_immediate=layer.load_immediate, store_immediate=layer.store_immediate, sparse_map=sparse_map) #, memory_target_size=hardware.total_sram_size, memory_initial_size=memory_initial_size)
	print(res[1], hardware.num_PE_lanes)
	return [res[0], power, res[4], res[5], res[6]]

def run_layer_set_no_pipelining(hardware, params, layer_set, estimate=False):
	for idx, (layer, _) in enumerate(layer_set.unique_layers):
		cycles, power, _, _, _ = run_layer(hardware, params[idx], layer, preload_cycles=0, pipeline_offloading=False, estimate=estimate)
		layer_set.update_layer_latency(layer, cycles)
		layer_set.power += power
		#layer_set.update_layer_dram_accesses(layer, dram_accesses)
	print(layer_set.get_string_stats(hardware.num_PE_lanes * hardware.num_PEs_per_lane, hardware.on_chip_bandwidth, params))

# hardware is from build_hardware.py
# layer_set is from build_models_v2.py
# params: list, where each element is a list: [split_dim, dataflow, ta, tb, tw, ca, cb, cw]
def run_layer_set(hardware, params, layer_set, init_preload_cycles=0):
	prev_preload_cycles = init_preload_cycles
	prev_layer = None
	prev_layer_cycles = 0
	prev_offload_cycles = 0
	for idx, layer in enumerate(layer_set.layers):
		print("***" * 10)
		print("Layer", idx, "out of", len(layer_set.layers))
		print("***" * 10)
		cycles, power, mem_idle, offload_cycles, offload_size = run_layer(hardware, params[idx], layer, preload_cycles=prev_preload_cycles, pipeline_offloading=True)
		# if the previous offloading cycles can be fit into the memory idle time of the current layer
		if prev_offload_cycles <= mem_idle:
			# give the rest of the mem idle cycles to the next layer to preload
			prev_preload_cycles = mem_idle - prev_offload_cycles
			print("Offload cycles from previous layer fit! Preload cycles for next layer:", prev_preload_cycles)
		else:
			# if the previous offloading cycles don't fit into the memory idle time of the current layer...
			# no preloading
			prev_preload_cycles = 0
			# increase the latency of the previous layer by the number of offloading cycles that didn't fit into the current layer's memory idle time
			prev_layer.actual_cycles = prev_layer_cycles + prev_offload_cycles - mem_idle
			#layer_set.update_layer_latency(prev_layer, prev_layer_cycles + prev_offload_cycles - mem_idle)
			print("Offload cycles from previous layer didn't fit :( Increased latency of previous layer by:", prev_offload_cycles - mem_idle)
		
		# update the latency of the current layer
		layer.actual_cycles = cycles
		layer_set.power += power
		#layer.actual_memory_accesses = dram_accesses
		#layer_set.update_layer_latency(layer, cycles)
		#layer_set.update_layer_dram_accesses(layer, dram_accesses)
		
		# update vars
		prev_layer = layer
		prev_layer_cycles = cycles
		prev_offload_cycles = offload_cycles
	
	# last layer can't offload cycles, so just sit and take it
	prev_layer.actual_cycles = prev_layer_cycles + prev_offload_cycles
	#layer_set.update_layer_latency(prev_layer, prev_layer_cycles + prev_offload_cycles)
	
	print(layer_set.get_string_stats(hardware.num_PE_lanes * hardware.num_PEs_per_lane, hardware.on_chip_bandwidth, params))
	
	return layer_set

# hardware is from build_hardware.py
# layer_set is from build_models_v2.py
# params: list, where each element is a list: [split_dim, dataflow, ta, tb, tw, ca, cb, cw]
# def run_layer_set(hardware, params, layer_set, init_preload_cycles=0):
	# prev_preload_cycles = init_preload_cycles
	# prev_layer = None
	# prev_layer_cycles = 0
	# prev_offload_size = 0
	# for idx, layer in enumerate(layer_set.layers):
		# print("***" * 10)
		# print("Layer", idx, "out of", len(layer_set.layers))
		# print("***" * 10)
		# cycles, dram_accesses, mem_idle, offload_cycles, offload_size = run_layer(hardware, params[idx], layer, preload_cycles=prev_preload_cycles, pipeline_offloading=True, memory_initial_size=prev_offload_size)
		
		# prev_offload_size = offload_size
		# prev_preload_cycles = mem_idle
		
		# # if the previous offloading cycles can be fit into the memory idle time of the current layer
		# #if prev_offload_cycles <= mem_idle:
			# # give the rest of the mem idle cycles to the next layer to preload
		# #	prev_preload_cycles = mem_idle - prev_offload_cycles
		# #	print("Offload cycles from previous layer fit! Preload cycles for next layer:", prev_preload_cycles)
		# #else:
			# # if the previous offloading cycles don't fit into the memory idle time of the current layer...
			# # no preloading
		# #	prev_preload_cycles = 0
			# # increase the latency of the previous layer by the number of offloading cycles that didn't fit into the current layer's memory idle time
		# #	prev_layer.actual_cycles = prev_layer_cycles + prev_offload_cycles - mem_idle
			# #layer_set.update_layer_latency(prev_layer, prev_layer_cycles + prev_offload_cycles - mem_idle)
		# #	print("Offload cycles from previous layer didn't fit :( Increased latency of previous layer by:", prev_offload_cycles - mem_idle)
		
		# # update the latency of the current layer
		# layer.actual_cycles = cycles
		# layer.actual_memory_accesses = dram_accesses
		# #layer_set.update_layer_latency(layer, cycles)
		# #layer_set.update_layer_dram_accesses(layer, dram_accesses)
		
		# # update vars
		# prev_layer = layer
		# prev_layer_cycles = cycles
		# prev_offload_cycles = offload_cycles
	
	# # last layer can't offload cycles, so just sit and take it
	# prev_layer.actual_cycles = prev_layer_cycles + prev_offload_cycles
	# #layer_set.update_layer_latency(prev_layer, prev_layer_cycles + prev_offload_cycles)
	
	# print(layer_set.get_string_stats(hardware.num_PE_lanes * hardware.num_PEs_per_lane, hardware.on_chip_bandwidth, params))
	
	# return layer_set

def test():
	hardware = hw.Hardware(num_PE_lanes=8, num_PEs_per_lane=64, num_RFs_per_PE=10, size_RF=10, off_chip_bandwidth=5, on_chip_bandwidth=10, total_sram_size=140*140*3)
	model = models.get_DeiT_Base(1, 1.0, 0.9)
	layer_set = models.model_to_layer_set(model)
	#layer = models.Layer(A_rows=198, A_cols_B_rows=198, B_cols=64, sparsity=0.9)
	params = [['rows', 'Output-Stationary', 140, 140, 140, 5, 5, 10] for i in range(len(layer_set.unique_layers))]
	#print("Result from run_layer:", run_layer(hardware, params, layer))
	run_layer_set(hardware, params, layer_set)

def run_attention_score_single_head(preload_cycles=0, bw=77, sparsity=0.9, lanes=8, PEs_per_lane=64, num_heads=12):
	hardware = hw.Hardware(num_PE_lanes=lanes, num_PEs_per_lane=PEs_per_lane, num_RFs_per_PE=11, size_RF=10, off_chip_bandwidth=bw, on_chip_bandwidth=10, total_sram_size=140*140*3)
	layer = models.Layer(A_rows=198, A_cols_B_rows=64, B_cols=198, sparsity=sparsity, num_heads=1)
	params = ['rows', 'Output-Stationary', 1072, 134, 134, 10, 1, 10] # for regular runs
	res = run_layer(hardware, params, layer, preload_cycles=preload_cycles, pipeline_offloading=False, generate_sparse_map=True)
	print("Attention Score, single head:", res, "all heads:", res[0] * num_heads)
	return res[0] * num_heads

def run_attention_score(preload_cycles=0, encdec=1.0, bw=77, sparsity=0.9, lanes=8, PEs_per_lane=64):
	hardware = hw.Hardware(num_PE_lanes=lanes, num_PEs_per_lane=PEs_per_lane, num_RFs_per_PE=11, size_RF=10, off_chip_bandwidth=bw, on_chip_bandwidth=10, total_sram_size=140*140*3)
	layer = models.Layer(A_rows=198*12, A_cols_B_rows=64, B_cols=198*12, sparsity=sparsity, num_heads=12, A_transfer_scale=encdec, B_transfer_scale=encdec, O_transfer_scale=1.0)
	params = ['rows', 'Output-Stationary', 1072, 134, 134, 5, 5, 10] # for regular runs
	res = run_layer(hardware, params, layer, preload_cycles=preload_cycles, pipeline_offloading=False)
	print("Attention Score:", res)
	return res[0]

def run_attention_score_decoder(bw=77):
	hardware = hw.Hardware(num_PE_lanes=8, num_PEs_per_lane=8, num_RFs_per_PE=11, size_RF=10, off_chip_bandwidth=bw, on_chip_bandwidth=10, total_sram_size=140*140*3)
	layer = models.Layer(A_rows=6, A_cols_B_rows=198*64, B_cols=12, sparsity=0.0, num_heads=1) # decoder
	params = ['rows', 'Output-Stationary', 134, 134, 1072, 10, 1, 10] # for encoder/decoder
	print("Decoder for Attention Score:", run_layer(hardware, params, layer, preload_cycles=1000, pipeline_offloading=True))

# just computes one of the query/key matrices; this is run twice to compute both of them
def run_query_key(encdec=1.0):
	hardware = hw.Hardware(num_PE_lanes=8, num_PEs_per_lane=63, num_RFs_per_PE=11, size_RF=10, off_chip_bandwidth=77, on_chip_bandwidth=10, total_sram_size=140*140*3)
	layer = models.Layer(A_rows=198, A_cols_B_rows=768, B_cols=1536 // 2, sparsity=0.0, num_heads=1, O_transfer_scale=encdec)
	params = ['rows', 'Output-Stationary', 134, 1072, 134, 1, 10, 10] # for regular runs
	print("Query/Key:", run_layer(hardware, params, layer, preload_cycles=0, pipeline_offloading=False))

def run_query_key_encoder():
	hardware = hw.Hardware(num_PE_lanes=8, num_PEs_per_lane=1, num_RFs_per_PE=11, size_RF=10, off_chip_bandwidth=77, on_chip_bandwidth=10, total_sram_size=140*140*3)
	layer = models.Layer(A_rows=12, A_cols_B_rows=198*64, B_cols=6, sparsity=0.9, num_heads=1) # encoder
	params = ['rows', 'Output-Stationary', 134, 134, 1072, 10, 1, 10] # for encoder/decoder
	print("Decoder for Attention Score:", run_layer(hardware, params, layer, preload_cycles=0, pipeline_offloading=False))

def run_deit_tiny(pipelining=True, comp_ratio=1.0, sparsity=0.0, bw=77):
	hardware = hw.Hardware(num_PE_lanes=8, num_PEs_per_lane=64, num_RFs_per_PE=11, size_RF=10, off_chip_bandwidth=bw, on_chip_bandwidth=100, total_sram_size=140*140*3)
	model = models.get_DeiT_Tiny(1, comp_ratio, comp_ratio, sparsity)
	layer_set = models.model_to_layer_set(model)
	params = [['rows', 'Output-Stationary', 1072, 134, 134, 10, 1, 10] for i in range(len(layer_set.layers))]
	for i in range(len(layer_set.layers)):
		if isinstance(layer_set.layers[i], ch.ConvLayer):
			params[i] = ["rows", "co", "ci", "y", "x", 120, 120, 5, 8, "x", "y", "ci", "kx", "ky", "co", 7, 1, 3, 4, 1, 1]
	if pipelining:
		run_layer_set(hardware, params, layer_set)
	else:
		run_layer_set_no_pipelining(hardware, params, layer_set)
	print("finished deit_tiny:", pipelining, comp_ratio, sparsity, bw)

def run_deit_small(pipelining=True, comp_ratio=1.0, sparsity=0.0, bw=77):
	hardware = hw.Hardware(num_PE_lanes=8, num_PEs_per_lane=64, num_RFs_per_PE=11, size_RF=10, off_chip_bandwidth=bw, on_chip_bandwidth=10, total_sram_size=140*140*3)
	model = models.get_DeiT_Small(1, comp_ratio, comp_ratio, sparsity)
	layer_set = models.model_to_layer_set(model)
	params = [['rows', 'Output-Stationary', 1072, 134, 134, 10, 1, 10] for i in range(len(layer_set.layers))]
	if pipelining:
		run_layer_set(hardware, params, layer_set)
	else:
		run_layer_set_no_pipelining(hardware, params, layer_set)
	print("finished deit_small:", pipelining, comp_ratio, sparsity, bw)

def run_deit_base(pipelining=True, comp_ratio=1.0, sparsity=0.0, bw=77):
	hardware = hw.Hardware(num_PE_lanes=8, num_PEs_per_lane=64, num_RFs_per_PE=11, size_RF=10, off_chip_bandwidth=bw, on_chip_bandwidth=10, total_sram_size=140*140*3)
	model = models.get_DeiT_Base(1, comp_ratio, comp_ratio, sparsity)
	layer_set = models.model_to_layer_set(model)
	params = [['rows', 'Output-Stationary', 1072, 134, 134, 10, 1, 10] for i in range(len(layer_set.layers))]
	if pipelining:
		run_layer_set(hardware, params, layer_set)
	else:
		run_layer_set_no_pipelining(hardware, params, layer_set)
	print("finished deit_base:", pipelining, comp_ratio, sparsity, bw)

def run_levit_128(pipelining=True, comp_ratio=1.0, sparsity=0.0, bw=77, ViTCoD=False, estimate=False):
	hardware = hw.Hardware(num_PE_lanes=8, num_PEs_per_lane=64, num_RFs_per_PE=11, size_RF=10, off_chip_bandwidth=bw, on_chip_bandwidth=10, total_sram_size=320000//2 - 512*11*10)
	model = models.get_LeViT_128(1, comp_ratio, comp_ratio, sparsity, ViTCoD)
	layer_set = models.model_to_layer_set(model)
	params = [['rows', 'Output-Stationary', 1072, 134, 134, 10, 1, 10] for i in range(len(layer_set.layers))]
	for i in range(len(layer_set.layers)):
		if isinstance(layer_set.layers[i], ch.ConvLayer):
			params[i] = ["rows", "co", "ci", "y", "x", 120, 120, 5, 8, "x", "y", "ci", "kx", "ky", "co", 7, 1, 3, 4, 1, 1]
	if pipelining:
		run_layer_set(hardware, params, layer_set)
	else:
		run_layer_set_no_pipelining(hardware, params, layer_set, estimate=estimate)
	print("finished levit_128:", pipelining, comp_ratio, sparsity, bw)

def run_levit_192(pipelining=True, comp_ratio=1.0, sparsity=0.0, bw=77, ViTCoD=False):
	hardware = hw.Hardware(num_PE_lanes=8, num_PEs_per_lane=64, num_RFs_per_PE=11, size_RF=10, off_chip_bandwidth=bw, on_chip_bandwidth=10, total_sram_size=140*140*3)
	model = models.get_LeViT_192(1, comp_ratio, comp_ratio, sparsity, ViTCoD)
	layer_set = models.model_to_layer_set(model)
	params = [['rows', 'Output-Stationary', 1072, 134, 134, 10, 1, 10] for i in range(len(layer_set.layers))]
	if pipelining:
		run_layer_set(hardware, params, layer_set)
	else:
		run_layer_set_no_pipelining(hardware, params, layer_set)
	print("finished levit_192:", pipelining, comp_ratio, sparsity, bw)

def run_levit_256(pipelining=True, comp_ratio=1.0, sparsity=0.0, bw=77, ViTCoD=False):
	hardware = hw.Hardware(num_PE_lanes=8, num_PEs_per_lane=64, num_RFs_per_PE=11, size_RF=10, off_chip_bandwidth=bw, on_chip_bandwidth=10, total_sram_size=140*140*3)
	model = models.get_LeViT_256(1, comp_ratio, comp_ratio, sparsity, ViTCoD)
	layer_set = models.model_to_layer_set(model)
	params = [['rows', 'Output-Stationary', 1072, 134, 134, 10, 1, 10] for i in range(len(layer_set.layers))]
	if pipelining:
		run_layer_set(hardware, params, layer_set)
	else:
		run_layer_set_no_pipelining(hardware, params, layer_set)
	print("finished levit_256:", pipelining, comp_ratio, sparsity, bw)

def run_nasvit_supernet(pipelining=True, comp_ratio=1.0, sparsity=0.0, bw=77):
	hardware = hw.Hardware(num_PE_lanes=8, num_PEs_per_lane=64, num_RFs_per_PE=11, size_RF=10, off_chip_bandwidth=bw, on_chip_bandwidth=10, total_sram_size=140*140*3)
	model = models.create_nasvit_supernet(1, comp_ratio, comp_ratio, sparsity)
	layer_set = models.model_to_layer_set(model)
	params = [['rows', 'Output-Stationary', 1072, 134, 134, 10, 1, 10] for i in range(len(layer_set.layers))]
	if pipelining:
		run_layer_set(hardware, params, layer_set)
	else:
		run_layer_set_no_pipelining(hardware, params, layer_set)
	print("finished nasvit supernet:", pipelining, comp_ratio, sparsity, bw)

def run_nasvit_smallest(pipelining=True, comp_ratio=1.0, sparsity=0.0, bw=77):
	hardware = hw.Hardware(num_PE_lanes=8, num_PEs_per_lane=64, num_RFs_per_PE=11, size_RF=10, off_chip_bandwidth=bw, on_chip_bandwidth=10, total_sram_size=140*140*3)
	model = models.create_nasvit_smallest(1, comp_ratio, comp_ratio, sparsity)
	layer_set = models.model_to_layer_set(model)
	params = [['rows', 'Output-Stationary', 1072, 134, 134, 10, 1, 10] for i in range(len(layer_set.layers))]
	if pipelining:
		run_layer_set(hardware, params, layer_set)
	else:
		run_layer_set_no_pipelining(hardware, params, layer_set)
	print("finished nasvit supernet:", pipelining, comp_ratio, sparsity, bw)

#def run_levit_128():
#	hardware = hw.Hardware(num_PE_lanes=8, num_PEs_per_lane=64, num_RFs_per_PE=11, size_RF=10, off_chip_bandwidth=77, on_chip_bandwidth=10, total_sram_size=140*140*3)
#	model = models.get_LeViT_128(1, 1.0, 0.0)
#	layer_set = models.model_to_layer_set(model)
#	params = [['rows', 'Output-Stationary', 1072, 134, 134, 10, 1, 10] for i in range(len(layer_set.layers))]
#	run_layer_set(hardware, params, layer_set)

def test_decoder_score(decode_preload_cycles=0, bw=77, sparsity=0.9, lanes=8, PEs_per_lane=64, num_heads=12):
	inst_const = 0.000001

	hardware = hw.Hardware(num_PE_lanes=lanes, num_PEs_per_lane=PEs_per_lane, num_RFs_per_PE=11, size_RF=10, off_chip_bandwidth=bw, on_chip_bandwidth=10, total_sram_size=140*140*3)
	#hardware2 = hw.Hardware(num_PE_lanes=1, num_PEs_per_lane=512, num_RFs_per_PE=11, size_RF=10, off_chip_bandwidth=bw, on_chip_bandwidth=bw, total_sram_size=140*140*3)
	
	decoder_layer = models.Layer(A_rows=198*64, A_cols_B_rows=num_heads // 2, B_cols=num_heads, sparsity=0.0, num_heads=1, store_immediate=True)
	decoder_params = ['rows', 'Output-Stationary', 1072, 134, 134, 10, 1, 10]
	
	score_layer = models.Layer(A_rows=198*num_heads, A_cols_B_rows=64, B_cols=198*num_heads, sparsity=sparsity, num_heads=num_heads, load_immediate=True)
	score_params = ['rows', 'Output-Stationary', 1072, 134, 134, 10, 1, 10]
	
	res_decoder = run_layer(hardware, decoder_params, decoder_layer, preload_cycles=decode_preload_cycles, pipeline_offloading=True)
	res_score = run_layer(hardware, score_params, score_layer, preload_cycles=0, pipeline_offloading=False)
	print("Result from decoder layer:", res_decoder)
	print("Result from score layer:", res_score)
	return res_decoder[0] * 2 + res_score[0]

def test_decoder_score_v2(pes_to_decode=1, decode_preload_cycles=0, bw=77):
	inst_const = 0.0001

	hardware = hw.Hardware(num_PE_lanes=8, num_PEs_per_lane=64 - pes_to_decode*2, num_RFs_per_PE=11, size_RF=10, off_chip_bandwidth=bw, on_chip_bandwidth=10, total_sram_size=140*140*3)
	hardware_decoder = hw.Hardware(num_PE_lanes=8, num_PEs_per_lane=pes_to_decode, num_RFs_per_PE=11, size_RF=10, off_chip_bandwidth=bw, on_chip_bandwidth=10, total_sram_size=140*140*3)
	
	decoder_layer = models.Layer(A_rows=198*64, A_cols_B_rows=6, B_cols=12, sparsity=0.0, num_heads=1, store_immediate=True)
	decoder_params = ['rows', 'Output-Stationary', 1072, 134, 134, 10, 1, 10]
	
	score_layer = models.Layer(A_rows=198*12, A_cols_B_rows=64, B_cols=198*12, sparsity=0.9, num_heads=12, load_immediate=True)
	score_params = ['rows', 'Output-Stationary', 1072, 134, 134, 10, 1, 10]
	
	res_decoder = run_layer(hardware_decoder, decoder_params, decoder_layer, preload_cycles=decode_preload_cycles, pipeline_offloading=True)
	res_score = run_layer(hardware, score_params, score_layer, preload_cycles=1000, pipeline_offloading=False)
	print("Result from decoder layer:", res_decoder)
	print("Result from score layer:", res_score)
	return max(res_decoder[0], res_score[0])

def run_roofline_score(bws, preload_cycles=0, pipeline_offloading=False):
	layer_score_dense = []
	layer_score_sparse = []
	layer_score_decode_q = []
	layer_score = []

	# add DeiT-Tiny
	model = models.get_DeiT_Tiny(1, 1.0, 1.0, 0.0)
	layer_set = models.model_to_layer_set(model)
	layer_score_dense.append(layer_set.unique_layers[2][0])
	#layer_set.print()
	
	model = models.get_DeiT_Tiny(1, 1.0, 1.0, 0.9)
	layer_set = models.model_to_layer_set(model)
	layer_score_sparse.append(layer_set.unique_layers[2][0])
	#layer_set.print()
	
	model = models.get_DeiT_Tiny(1, 0.5, 0.5, 0.9)
	layer_set = models.model_to_layer_set(model)
	#layer_set.print()
	layer_score_decode_q.append(layer_set.unique_layers[3][0])
	layer_score.append(layer_set.unique_layers[4][0])

	# add DeiT-Small
	model = models.get_DeiT_Small(1, 1.0, 1.0, 0.0)
	layer_set = models.model_to_layer_set(model)
	layer_score_dense.append(layer_set.unique_layers[2][0])
	#layer_set.print()
	
	model = models.get_DeiT_Small(1, 1.0, 1.0, 0.9)
	layer_set = models.model_to_layer_set(model)
	layer_score_sparse.append(layer_set.unique_layers[2][0])
	#layer_set.print()
	
	model = models.get_DeiT_Small(1, 0.5, 0.5, 0.9)
	layer_set = models.model_to_layer_set(model)
	#layer_set.print()
	layer_score_decode_q.append(layer_set.unique_layers[3][0])
	layer_score.append(layer_set.unique_layers[4][0])

	# add DeiT-Base
	model = models.get_DeiT_Base(1, 1.0, 1.0, 0.0)
	layer_set = models.model_to_layer_set(model)
	layer_score_dense.append(layer_set.unique_layers[2][0])
	#layer_set.print()
	
	model = models.get_DeiT_Base(1, 1.0, 1.0, 0.9)
	layer_set = models.model_to_layer_set(model)
	layer_score_sparse.append(layer_set.unique_layers[2][0])
	#layer_set.print()
	
	model = models.get_DeiT_Base(1, 0.5, 0.5, 0.9)
	layer_set = models.model_to_layer_set(model)
	#layer_set.print()
	layer_score_decode_q.append(layer_set.unique_layers[3][0])
	layer_score.append(layer_set.unique_layers[4][0])
	
	# add LeViT-128
	model = models.get_LeViT_128(1, 1.0, 1.0, 0.0)
	layer_set = models.model_to_layer_set(model)
	layer_score_dense.append(layer_set.unique_layers[5][0])
	#layer_set.print()
	
	model = models.get_LeViT_128(1, 1.0, 1.0, 0.9)
	layer_set = models.model_to_layer_set(model)
	layer_score_sparse.append(layer_set.unique_layers[5][0])
	#layer_set.print()
	
	model = models.get_LeViT_128(1, 0.5, 0.5, 0.9)
	layer_set = models.model_to_layer_set(model)
	#layer_set.print()
	layer_score_decode_q.append(layer_set.unique_layers[6][0])
	layer_score.append(layer_set.unique_layers[7][0])
	
	# add LeViT-192
	model = models.get_LeViT_192(1, 1.0, 1.0, 0.0)
	layer_set = models.model_to_layer_set(model)
	layer_score_dense.append(layer_set.unique_layers[5][0])
	#layer_set.print()
	
	model = models.get_LeViT_192(1, 1.0, 1.0, 0.9)
	layer_set = models.model_to_layer_set(model)
	layer_score_sparse.append(layer_set.unique_layers[5][0])
	#layer_set.print()
	
	model = models.get_LeViT_192(1, 0.5, 0.5, 0.9)
	layer_set = models.model_to_layer_set(model)
	#layer_set.print()
	layer_score_decode_q.append(layer_set.unique_layers[6][0])
	layer_score.append(layer_set.unique_layers[7][0])
	
	# add LeViT-256
	model = models.get_LeViT_256(1, 1.0, 1.0, 0.0)
	layer_set = models.model_to_layer_set(model)
	layer_score_dense.append(layer_set.unique_layers[5][0])
	#layer_set.print()
	
	model = models.get_LeViT_256(1, 1.0, 1.0, 0.9)
	layer_set = models.model_to_layer_set(model)
	layer_score_sparse.append(layer_set.unique_layers[5][0])
	#layer_set.print()
	
	model = models.get_LeViT_256(1, 0.5, 0.5, 0.9)
	layer_set = models.model_to_layer_set(model)
	#layer_set.print()
	layer_score_decode_q.append(layer_set.unique_layers[6][0])
	layer_score.append(layer_set.unique_layers[7][0])
	
	cycles_dense = []
	cycles_sparse = []
	cycles_q = []
	cycles_k = []
	cycles_score = []
	for dense, sparse, q, score in zip(layer_score_dense, layer_score_sparse, layer_score_decode_q, layer_score):
		for bw in bws:
			hardware = hw.Hardware(num_PE_lanes=8, num_PEs_per_lane=64, num_RFs_per_PE=10, size_RF=10, off_chip_bandwidth=bw, on_chip_bandwidth=10, total_sram_size=140*140*3)
			params = ['rows', 'Output-Stationary', 1072, 134, 134, 5, 5, 10]
			params_decode = ['rows', 'Output-Stationary', 1072*100, 12, 6, 4, 6, 10]
			cycles_dense.append(run_layer(hardware, params, dense, preload_cycles=preload_cycles // score.num_heads, pipeline_offloading=pipeline_offloading)[0] * score.num_heads)
			cycles_sparse.append(run_layer(hardware, params, sparse, preload_cycles=preload_cycles // score.num_heads, pipeline_offloading=pipeline_offloading)[0] * score.num_heads)
			cycles_q.append(run_layer(hardware, params_decode, q, preload_cycles=preload_cycles, pipeline_offloading=pipeline_offloading)[0])
			#cycles_k.append(run_layer(hardware, params, layer_score_decode_k, preload_cycles=preload_cycles, pipeline_offloading=pipeline_offloading)[0])
			cycles_score.append(run_layer(hardware, params, score, preload_cycles=preload_cycles, pipeline_offloading=pipeline_offloading)[0])
			#cycles_dense.append(run_attention_score_single_head(preload_cycles=0, bw=bw, num_heads=nh, sparsity=0.0, lanes=lanes, PEs_per_lane=PEs_per_lane))
			#cycles_sparse.append(run_attention_score_single_head(preload_cycles=0, bw=bw, num_heads=nh, sparsity=0.9, lanes=lanes, PEs_per_lane=PEs_per_lane))
		#cycles_tacos.append(test_decoder_score(decode_preload_cycles=0, bw=bw, num_heads=nh, sparsity=0.9, lanes=lanes, PEs_per_lane=PEs_per_lane))
	
	cycles_tacos = [i*2+j for i,j in zip(cycles_q, cycles_score)]
	
	print(bws)
	print(cycles_dense)
	print(cycles_sparse)
	print(cycles_tacos)

def run_test(bws=[77], preload_cycles=1000, sparsity=0.9, lanes=8, PEs_per_lane=[64]):
	#bws = [10] #10, 20, 30, 40, 50, 60, 70, 80]
	cycles_single = []
	cycles_overlapped = []
	cycles_decoder = []
	for bw in bws:
		for PE in PEs_per_lane:
			cycles_single.append(run_attention_score_single_head(preload_cycles=preload_cycles, bw=bw, sparsity=sparsity, lanes=lanes, PEs_per_lane=PE))
			#cycles_overlapped.append(run_attention_score(preload_cycles=preload_cycles, bw=bw, sparsity=sparsity, lanes=lanes, PEs_per_lane=PE))
			cycles_decoder.append(test_decoder_score(decode_preload_cycles=preload_cycles, bw=bw, sparsity=sparsity, lanes=lanes, PEs_per_lane=PE))
	
	print(bws)
	print(PEs_per_lane)
	print(cycles_single)
	#print(cycles_overlapped)
	print(cycles_decoder)
	
def find_ideal_num_PEs(layers, hardware, params, pes_per_lane_options, preload_cycles=0, pipeline_offloading=False):
	results = []
	for layer in layers:
		curr_res = []
		for pe in pes_per_lane_options:
			hardware.num_PEs_per_lane = pe
			curr_res.append(run_layer(hardware, params, layer, preload_cycles=preload_cycles, pipeline_offloading=pipeline_offloading))
		print("***" * 10)
		layer.print()
		print("Results:", curr_res)
		print("***" * 10)
		results.append(curr_res)
	print("***" * 15)
	print(results)
	print("PEs:", pes_per_lane_options)
	for idx, res in enumerate(results):
		print(idx)
		cycles = [i[0] for i in res]
		print(cycles)

def run_ideal_num_PEs():
	model = models.get_DeiT_Tiny(1, 0.5, 0.5, 0.9)
	layer_set = models.model_to_layer_set(model)
	layers = [i[0] for i in layer_set.unique_layers]
	#layers = []
	#layers.append(layer_set.unique_layers[5][0])
	#layers.append(layer_set.unique_layers[6][0])
	#layers.append(layer_set.unique_layers[7][0])
	hardware = hw.Hardware(num_PE_lanes=8, num_PEs_per_lane=64, num_RFs_per_PE=10, size_RF=10, off_chip_bandwidth=77, on_chip_bandwidth=10, total_sram_size=140*140*3)
	params = ['rows', 'Output-Stationary', 1072, 134, 134, 5, 5, 10]
	pes_per_lane_options = [(i + 1) * 10 for i in range(20)]
	find_ideal_num_PEs(layers, hardware, params, pes_per_lane_options)
	layer_set.print()

def test_single_layer():
	model = models.get_DeiT_Small(1, 0.5, 0.5, 0.9)
	layer_set = models.model_to_layer_set(model)
	layer_decode = layer_set.unique_layers[5][0]
	layer_score = layer_set.unique_layers[7][0]
	
	model = models.get_DeiT_Small(1, 1.0, 1.0, 0.9)
	layer_set = models.model_to_layer_set(model)
	layer_no_ed = layer_set.unique_layers[3][0]
	#layer_set.print()
	#print("Total FLOPs:", layer_set.get_total_flops_including_extras())
	#print("Ideal cycles:", layer_set.get_total_flops_including_extras() / 512)
	#for layer in layer_set.layers:
	#	layer.print()
	hardware = hw.Hardware(num_PE_lanes=8, num_PEs_per_lane=64, num_RFs_per_PE=10, size_RF=10, off_chip_bandwidth=10, on_chip_bandwidth=10, total_sram_size=140*140*3)
	#model = models.get_DeiT_Base(1, 1.0, 0.9)
	#layer_set = models.model_to_layer_set(model)
	#layer_set.unique_layers[1][0].print()
	#layer = models.Layer(A_rows=20, A_cols_B_rows=20, B_cols=20, sparsity=0.9, num_heads=1)
	#layer = layer_set.unique_layers[3][0] #models.Layer(A_rows=3, A_cols_B_rows=197*64, B_cols=2, sparsity=0.0, num_heads=1, A_transfer_scale=0.0001, B_transfer_scale=1) # encoding Q for DeiT Tiny like ViTCoD
	#layer = models.Layer(A_rows=100, A_cols_B_rows=100, B_cols=100, sparsity=0.9, num_heads=1) # attention score
	#layer = models.Layer(A_rows=198*64, A_cols_B_rows=12, B_cols=6, sparsity=0.0, num_heads=1) # encoder
	#layer = models.Layer(A_rows=6, A_cols_B_rows=198*64, B_cols=12, sparsity=0.0, num_heads=1) # decoder
	params = ['rows', 'Output-Stationary', 134*8*10*10, 13, 13, 5, 5, 10] # for encoder/decoder
	#params = ['rows', 'Output-Stationary', 10, 10, 10, 10, 1, 10] # for regular runs
	print("Result decode:", run_layer(hardware, params, layer_decode, preload_cycles=1000, pipeline_offloading=True))
	print("Result score:", run_layer(hardware, params, layer_score, preload_cycles=1000, pipeline_offloading=True))
	print("Result no enc/dec:", run_layer(hardware, params, layer_no_ed, preload_cycles=1000, pipeline_offloading=True))

def run_deit_tiny_heatvit(pipelining=True, comp_ratio=1.0, sparsity=0.0):
	hardware = hw.Hardware(num_PE_lanes=32, num_PEs_per_lane=64, num_RFs_per_PE=10, size_RF=10, off_chip_bandwidth=19.2/0.15, on_chip_bandwidth=10, total_sram_size=140*140*3)
	model = models.get_DeiT_Tiny(1, comp_ratio, comp_ratio, sparsity)
	layer_set = models.model_to_layer_set(model)
	# 12 MB mem total --> 354*32, 354, 354
	# Matching HeatViT w/ 8-bit precision
	# N*32*2N + N*N = 65N*N = 12MB - 100*32*64 == 11.7MB; N = 445
	params = [['rows', 'Output-Stationary', 445*32, 445, 445, 5, 5, 10] for i in range(len(layer_set.layers))]
	if pipelining:
		run_layer_set(hardware, params, layer_set)
	else:
		run_layer_set_no_pipelining(hardware, params, layer_set)
	print("finished deit_tiny, heatvit config:", pipelining, comp_ratio, sparsity, bw)

def run_deit_small_heatvit(pipelining=True, comp_ratio=1.0, sparsity=0.0):
	hardware = hw.Hardware(num_PE_lanes=32, num_PEs_per_lane=64, num_RFs_per_PE=10, size_RF=10, off_chip_bandwidth=19.2/0.15, on_chip_bandwidth=10, total_sram_size=140*140*3)
	model = models.get_DeiT_Small(1, comp_ratio, comp_ratio, sparsity)
	layer_set = models.model_to_layer_set(model)
	# 12 MB mem total --> 354*32, 354, 354
	# Matching HeatViT w/ 8-bit precision
	# N*32*2N + N*N = 65N*N = 12MB - 100*32*64 == 11.7MB; N = 445
	params = [['rows', 'Output-Stationary', 445*32, 445, 445, 5, 5, 10] for i in range(len(layer_set.layers))]
	if pipelining:
		run_layer_set(hardware, params, layer_set)
	else:
		run_layer_set_no_pipelining(hardware, params, layer_set)
	print("finished deit_small, heatvit config:", pipelining, comp_ratio, sparsity, bw)

def run_deit_base_heatvit(pipelining=True, comp_ratio=1.0, sparsity=0.0):
	hardware = hw.Hardware(num_PE_lanes=32, num_PEs_per_lane=64, num_RFs_per_PE=10, size_RF=10, off_chip_bandwidth=19.2/0.15, on_chip_bandwidth=10, total_sram_size=140*140*3)
	model = models.get_DeiT_Base(1, comp_ratio, comp_ratio, sparsity)
	layer_set = models.model_to_layer_set(model)
	# 12 MB mem total --> 354*32, 354, 354
	# Matching HeatViT w/ 8-bit precision
	# N*32*2N + N*N = 65N*N = 12MB - 100*32*64 == 11.7MB; N = 445
	params = [['rows', 'Output-Stationary', 445*32, 445, 445, 5, 5, 10] for i in range(len(layer_set.layers))]
	if pipelining:
		run_layer_set(hardware, params, layer_set)
	else:
		run_layer_set_no_pipelining(hardware, params, layer_set)
	print("finished deit_base, heatvit config:", pipelining, comp_ratio, sparsity, bw)

def test():
	hardware = hw.Hardware(num_PE_lanes=8, num_PEs_per_lane=64, num_RFs_per_PE=12, size_RF=11, off_chip_bandwidth=10, on_chip_bandwidth=100, total_sram_size=140*140*3)
	params = ['rows', 'B-Stationary', 360, 94, 52, 1, 11, 11]
	layer = models.Layer(A_rows=594, B_cols=594, A_cols_B_rows=64, sparsity=0.9, num_heads=3, load_immediate=False)
	print(run_layer(hardware, params, layer, preload_cycles=10000))

if __name__ == "__main__":
	#model = models.get_LeViT_128(1, 0.5, 0.5, 0.9)
	#layer_set = models.model_to_layer_set(model)
	#layer_set.print()
	#test_decoder_score(decode_preload_cycles=0, bw=5, sparsity=0.9, lanes=8, PEs_per_lane=8)
	#run_roofline_score([19.2*2], 1000, True)
	#run_test(bws=[10], preload_cycles=0, lanes=8, PEs_per_lane=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
	#run_test(bws=[10], preload_cycles=0, lanes=8, PEs_per_lane=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
	#test_decoder_score(bw=5)
	#test_decoder_score_v2(pes_to_decode=5, bw=10)
	#run_deit_tiny_heatvit(pipelining=True, comp_ratio=1.0, sparsity=0.9)
	#run_levit_256(pipelining=True, comp_ratio=1.0, sparsity=0.0, ViTCoD=True)
	run_levit_128(pipelining=False, comp_ratio=1.0, sparsity=0.9, ViTCoD=True, estimate=True)
	#run_nasvit_supernet(pipelining=False, sparsity=0.9)
	#run_levit_128()
	#run_attention_score_single_head(bw=77)
	#run_attention_score()
	#run_attention_score_decoder()
	#run_query_key(0.5)
	#run_query_key_encoder()
	#test_single_layer()
	#run_ideal_num_PEs()
	#test()
	
#	ops = 64 * 12
#	loading = 64 * 2 * 12
#	bandwdith = 76.8 * 1e9
#	frequency = 500 * 1e6
#	cycles_ops = ops / (64 * 8)
#	cycles_loading = loading / bandwdith * frequency
#	cycles = cycles_loading / cycles_ops
#	print(cycles)
