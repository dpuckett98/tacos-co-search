import math
import timeit

from fastarch import build_models_v2 as bm
from fastarch import build_hardware_v2 as bh
from fastarch import dataflow_wrapper as dw

def get_next_coords(dataflow, row, col, depth, tile_rows, tile_cols, tile_depth):
	next_row = row
	next_col = col
	next_depth = depth
	
	if dataflow == "Output-Stationary":
		next_depth += 1
		if next_depth == tile_depth:
			next_col += 1
			next_depth = 0
			if next_col == tile_cols:
				next_row += 1
				next_col = 0
				if next_row == tile_rows:
					return None
	elif dataflow == "A-Stationary":
		next_col += 1
		if next_col == tile_cols:
			next_col = 0
			next_depth += 1
			if next_depth == tile_depth:
				next_depth = 0
				next_row += 1
				if next_row == tile_rows:
					return None
	elif dataflow == "B-Stationary":
		next_row += 1
		if next_row == tile_rows:
			next_row = 0
			next_depth += 1
			if next_depth == tile_depth:
				next_depth = 0
				next_col += 1
				if next_col == tile_cols:
					return None
	return [next_row, next_col, next_depth]

def _calc_chunks(t_a, t_b, t_w, c_a, c_b, c_w, num_PEs, sparsity):
	chunk_startup_cost = 1
	if c_a > t_a:
		c_a = t_a
	if c_b > t_b:
		c_b = t_b
	if c_w > t_w:
		c_w = t_w
	
	# build chunks
	chunks = []
	for row in range(0, math.ceil(t_a / c_a)):
		ca_length = c_a if row != math.ceil(t_a / c_a) - 1 else t_a - row * c_a
		for col in range(0, math.ceil(t_b / c_b)):
			cb_length = c_b if col != math.ceil(t_b / c_b) - 1 else t_b - col * c_b
			for depth in range(0, math.ceil(t_w / c_w)):
				cw_length = c_w if depth != math.ceil(t_w / c_w) - 1 else t_w - depth * c_w
				chunks.append(ca_length * cb_length * cw_length * (1 - sparsity))
	
	cycles = 0
	
	# slightly inaccurate because it assumes that the PEs have synchronized start & stop, rather than asynchronous
	start = 0
	while len(chunks) > start:
		max_size = 0
		# each PE takes a chunk
		if start + num_PEs < len(chunks):
			max_size = max(chunks[start:start + num_PEs + 1])
			start += num_PEs
		else:
			max_size = max(chunks[start:])
			start = len(chunks)
		
		cycles += chunk_startup_cost + max_size
	
	return cycles
	

def fast_compute_bound_estimate(hardware, layer, params):
	# static variables
	tile_startup_cost = 0
	layer_startup_cost = 0
	
	total_cycles = 0

	# divide layers among PE lanes
	A_rows = math.ceil(layer.A_rows / hardware.num_PE_lanes) if params[0] == "rows" else layer.A_rows
	A_cols_B_rows = layer.A_cols_B_rows
	B_cols = math.ceil(layer.B_cols / hardware.num_PE_lanes) if params[0] == "cols" else layer.B_cols
	
	# divide tiles among PE lanes
	t_a = math.ceil(params[2] / hardware.num_PE_lanes) if params[0] == "rows" else params[2]
	t_w = params[4]
	t_b = math.ceil(params[3] / hardware.num_PE_lanes) if params[0] == "cols" else params[3]
	
	if t_a > layer.A_rows:
		t_a = layer.A_rows
	if t_w > layer.A_cols_B_rows:
		t_w = layer.A_cols_B_rows
	if t_b > layer.B_cols:
		t_b = layer.B_cols
	
	a_tiles = math.ceil(A_rows / t_a)
	b_tiles = math.ceil(B_cols / t_b)
	w_tiles = math.ceil(A_cols_B_rows / t_w)
	
	# do all the regular tiles
	num_reg = (a_tiles - 1) * (b_tiles - 1) * (w_tiles - 1)
	reg_cycles = num_reg * _calc_chunks(t_a, t_b, t_w, params[5], params[6], params[7], hardware.num_PEs_per_lane, layer.sparsity)
	
	# tiles where only a is short
	num_a_short = (b_tiles - 1) * (w_tiles - 1)
	short_t_a = A_rows - t_a * (a_tiles - 1)
	a_short_cycles = num_a_short * _calc_chunks(short_t_a, t_b, t_w, params[5], params[6], params[7], hardware.num_PEs_per_lane, layer.sparsity)
	
	# tiles where only b is short
	num_b_short = (a_tiles - 1) * (w_tiles - 1)
	short_t_b = B_cols - t_b * (b_tiles - 1)
	b_short_cycles = num_b_short * _calc_chunks(t_a, short_t_b, t_w, params[5], params[6], params[7], hardware.num_PEs_per_lane, layer.sparsity)
	
	# tiles where only w is short
	num_w_short = (b_tiles - 1) * (a_tiles - 1)
	short_t_w = A_cols_B_rows - t_w * (w_tiles - 1)
	w_short_cycles = num_w_short * _calc_chunks(t_a, t_b, short_t_w, params[5], params[6], params[7], hardware.num_PEs_per_lane, layer.sparsity)
	
	# tiles where a and b are short
	num_ab_short = (w_tiles - 1)
	ab_short_cycles = num_ab_short * _calc_chunks(short_t_a, short_t_b, t_w, params[5], params[6], params[7], hardware.num_PEs_per_lane, layer.sparsity)
	
	# tiles where a and w are short
	num_aw_short = (b_tiles - 1)
	aw_short_cycles = num_aw_short * _calc_chunks(short_t_a, t_b, short_t_w, params[5], params[6], params[7], hardware.num_PEs_per_lane, layer.sparsity)
	
	# tiles where w and b are short
	num_wb_short = (a_tiles - 1)
	wb_short_cycles = num_wb_short * _calc_chunks(t_a, short_t_b, short_t_w, params[5], params[6], params[7], hardware.num_PEs_per_lane, layer.sparsity)
	
	# tiles where a, b, and w are short
	num_abw_short = 1
	abw_short_cycles = num_abw_short * _calc_chunks(short_t_a, short_t_b, short_t_w, params[5], params[6], params[7], hardware.num_PEs_per_lane, layer.sparsity)
	
	return reg_cycles + a_short_cycles + b_short_cycles + w_short_cycles + ab_short_cycles + aw_short_cycles + wb_short_cycles + abw_short_cycles

def compute_bound_estimate(hardware, layer, params):
	# static variables
	chunk_startup_cost = 1
	tile_startup_cost = 0
	layer_startup_cost = 0
	
	total_cycles = 0

	# divide layers among PE lanes
	A_rows = math.ceil(layer.A_rows / hardware.num_PE_lanes) if params[0] == "rows" else layer.A_rows
	A_cols_B_rows = layer.A_cols_B_rows
	B_cols = math.ceil(layer.B_cols / hardware.num_PE_lanes) if params[0] == "cols" else layer.B_cols
	
	# divide tiles among PE lanes
	tt_a = math.ceil(params[2] / hardware.num_PE_lanes) if params[0] == "rows" else params[2]
	tt_w = params[4]
	tt_b = math.ceil(params[3] / hardware.num_PE_lanes) if params[0] == "cols" else params[3]
	
	num_tiles = 0
	tile_costs = []
	
	# for each tile
	for t_a in range(0, math.ceil(A_rows / tt_a)):
		a_length = tt_a if t_a != math.ceil(A_rows / tt_a) - 1 else A_rows - t_a * tt_a
		for t_b in range(0, math.ceil(B_cols / tt_b)):
			b_length = tt_b if t_b != math.ceil(B_cols / tt_b) - 1 else B_cols - t_b * tt_b
			for t_w in range(0, math.ceil(A_cols_B_rows / tt_w)):
				w_length = tt_w if t_w != math.ceil(A_cols_B_rows / tt_w) - 1 else A_cols_B_rows - t_w * tt_w
				init = 0
				init += tile_startup_cost
				num_tiles += 1
				
				# build chunks
				chunks = []
				for c_a in range(0, math.ceil(a_length / params[5])):
					ca_length = params[5] if c_a != math.ceil(a_length / params[5]) - 1 else a_length - c_a * params[5]
					for c_b in range(0, math.ceil(b_length / params[6])):
						cb_length = params[6] if c_b != math.ceil(b_length / params[6]) - 1 else b_length - c_b * params[6]
						for c_w in range(0, math.ceil(w_length / params[7])):
							cw_length = params[7] if c_w != math.ceil(w_length / params[7]) - 1 else w_length - c_w * params[7]
							chunks.append(ca_length * cb_length * cw_length * (1 - layer.sparsity))
				#print(len(chunks))
				# assign chunks to PEs
				#print("Starting new tile")
				
				# slightly inaccurate because it assumes that the PEs have synchronized start & stop, rather than asynchronous
				while len(chunks) > 0:
					max_size = 0
					# each PE takes a chunk
					for i in range(hardware.num_PEs_per_lane):
						if len(chunks) == 0:
							break
						size = chunks[0]
						del chunks[0]
						if size > max_size:
							max_size = size
					#print(max_size)
					init += chunk_startup_cost + max_size
				
				total_cycles += init
				tile_costs.append(init)
				#total_cycles += chunk_startup_cost
				#total_cycles += ca_length * cb_length * cw_length
							
	#print("Num tiles:", num_tiles)
	#print("Average tile cost:", sum(tile_costs) / len(tile_costs))
	#print("Predicted compute-bound time:", total_cycles) #num_tiles * predicted_time_per_tile)
	return total_cycles

def bandwidth_bound_estimate(hardware, layer, params, count_offload=True):
	# static variables
	tile_overhead = 0
	elements_per_cycle = hardware.off_chip_bandwidth # / hardware.num_PE_lanes
	
	total_cycles = 0
	total_accesses = 0
	
	A_rows = layer.A_rows
	B_cols = layer.B_cols
	A_cols_B_rows = layer.A_cols_B_rows
	tt_a = params[2]
	tt_b = params[3]
	tt_w = params[4]
	
	# for each tile
	prev_row = -1
	prev_col = -1
	prev_depth = -1
	
	t_a = 0
	t_b = 0
	t_w = 0
	while True:
		a_length = tt_a if t_a != math.ceil(A_rows / tt_a) - 1 else A_rows - t_a * tt_a
		b_length = tt_b if t_b != math.ceil(B_cols / tt_b) - 1 else B_cols - t_b * tt_b
		w_length = tt_w if t_w != math.ceil(A_cols_B_rows / tt_w) - 1 else A_cols_B_rows - t_w * tt_w
		
		tile_cycles = 0
		
		a_tile_size = a_length * w_length
		b_tile_size = b_length * w_length
		o_tile_size = a_length * b_length
		
		# check if A and B tiles need to be loaded
		if prev_row != -1:
			load_A = prev_row != t_a or prev_depth != t_w
			load_B = prev_col != t_b or prev_depth != t_w
			if t_w == 0:
				load_O = False
			else:
				load_O = prev_row != t_a or prev_col != t_b
		else:
			load_A = True
			load_B = True
			load_O = False
		
		# check if O tile needs to be saved
		res = get_next_coords(params[1], t_a, t_b, t_w, math.ceil(A_rows / tt_a), math.ceil(B_cols / tt_b), math.ceil(A_cols_B_rows / tt_w))
		if res != None:
			next_row, next_col, next_depth = res
			save_O = (t_a != next_row or t_b != next_col)
		else:
			save_O = True
		#if params[1] == 
		
		# load data
		if load_A:
			tile_cycles += a_tile_size / elements_per_cycle
			total_accesses += a_tile_size
		if load_B:
			tile_cycles += b_tile_size / elements_per_cycle
			total_accesses += b_tile_size
		if load_O:
			tile_cycles += o_tile_size / elements_per_cycle
			total_accesses += o_tile_size
		
		# store data
		if count_offload and save_O:
			tile_cycles += o_tile_size / elements_per_cycle
			total_accesses += o_tile_size
		#print(tile_cycles, a_tile_size, b_tile_size, o_tile_size, a_tile_size / elements_per_cycle, b_tile_size / elements_per_cycle, o_tile_size / elements_per_cycle)
		total_cycles += tile_cycles
		
		# prep for next tile
		if res == None:
			break
		prev_row = t_a
		prev_col = t_b
		prev_depth = t_w
		t_a = res[0]
		t_b = res[1]
		t_w = res[2]
	
	return [total_cycles, total_accesses]

def estimate_performance(hardware, layer, params):
	compute_bound_cycles = fast_compute_bound_estimate(hardware, layer, params)
	bandwidth_bound_cycles, accesses = bandwidth_bound_estimate(hardware, layer, params)
	#print("Compute cycles:", compute_bound_cycles)
	#print("Load/Store cycles:", bandwidth_bound_cycles)
	return [compute_bound_cycles, bandwidth_bound_cycles, accesses * hardware.unit_energy_DRAM + layer.get_flops_including_extras() * hardware.unit_energy_MAC]

def test():
	hardware = bh.Hardware(num_PE_lanes=8, num_PEs_per_lane=64, num_RFs_per_PE=11, size_RF=10, off_chip_bandwidth=20, on_chip_bandwidth=10, total_sram_size=320000//2 - 512*11*10)
	model = bm.get_LeViT_128(1, 1.0, 1.0, 0.0, True)
	layer_set = bm.model_to_layer_set(model)
	params = ['rows', 'B-Stationary', 1072, 134, 134, 1, 1, 10]
	
	layer = layer_set.unique_layers[0][0]
	#layer = bm.layer
	layer.print()
	#res = dw.run_layer(hardware, params, layer, preload_cycles=0, pipeline_offloading=False)
	#print(res)
	
	#res = estimate_performance(hardware, layer, params)
	setup = '''
from __main__ import fast_compute_bound_estimate, compute_bound_estimate, estimate_performance
import math
import build_models_v2 as bm
import build_hardware_v2 as bh
import dataflow_wrapper as dw

hardware = bh.Hardware(num_PE_lanes=8, num_PEs_per_lane=64, num_RFs_per_PE=11, size_RF=10, off_chip_bandwidth=20, on_chip_bandwidth=10, total_sram_size=320000//2 - 512*11*10)
model = bm.create_nasvit_supernet(1, 1.0, 1.0, 0.0)
layer_set = bm.model_to_layer_set(model)
params = ['rows', 'B-Stationary', 3845, 18, 10, 6, 4, 10]
layer = layer_set.unique_layers[0][0]
	'''
	
	res1 = timeit.timeit('fast_compute_bound_estimate(hardware, layer, params)', setup=setup, number=100)
	#print("Fast one done")
	res2 = timeit.timeit('estimate_performance(hardware, layer, params)', setup=setup, number=100)
	ideal_cycles = layer.get_flops_including_extras() / (hardware.num_PE_lanes * hardware.num_PEs_per_lane)
	#util = ideal_cycles / res
	print(res1, res2, ideal_cycles)

if __name__ == "__main__":
	test()
