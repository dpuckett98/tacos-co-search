import math

from fastarch import build_models_v2 as bm
from fastarch import build_hardware_v2 as bh
from fastarch import dataflow_wrapper as dw
from fastarch import dataflow_convolution as dc
import fastarch.conv_helper as ch

# returns 'None' on last one
# inputs are by tile; e.g. row=2 means this is the third tile
def get_next_coords(dataflow, row, col, c_in, c_out, num_tiles_rows, num_tiles_cols, num_tiles_ci, num_tiles_co):
	next_row = row
	next_col = col
	next_c_in = c_in
	next_c_out = c_out
	
	maxs = [num_tiles_rows, num_tiles_cols, num_tiles_ci, num_tiles_co]
	vals = [next_row, next_col, next_c_in, next_c_out]
	idxs = [0, 1, 2, 3]
	for i in range(4):
		if dataflow[i] == "y":
			idxs[i] = 0
		elif dataflow[i] == "x":
			idxs[i] = 1
		elif dataflow[i] == "ci":
			idxs[i] = 2
		elif dataflow[i] == "co":
			idxs[i] = 3
	
	vals[idxs[0]] += 1
	if vals[idxs[0]] >= maxs[idxs[0]]:
		vals[idxs[0]] = 0
		vals[idxs[1]] += 1
		if vals[idxs[1]] >= maxs[idxs[1]]:
			vals[idxs[1]] = 0
			vals[idxs[2]] += 1
			if vals[idxs[2]] >= maxs[idxs[2]]:
				vals[idxs[2]] = 0
				vals[idxs[3]] += 1
				if vals[idxs[3]] >= maxs[idxs[3]]:
					vals[idxs[3]] = 0
					# finished!
					return None
	
	return [vals[0], vals[1], vals[2], vals[3]]

# returns 'None' if no previous tile
# inputs are by tile; e.g. row=2 means this is the third tile
def get_prev_coords(dataflow, row, col, c_in, c_out, num_tiles_rows, num_tiles_cols, num_tiles_ci, num_tiles_co):
	next_row = row
	next_col = col
	next_c_in = c_in
	next_c_out = c_out
	
	maxs = [num_tiles_rows, num_tiles_cols, num_tiles_ci, num_tiles_co]
	vals = [next_row, next_col, next_c_in, next_c_out]
	idxs = [0, 1, 2, 3]
	for i in range(4):
		if dataflow[i] == "y":
			idxs[i] = 0
		elif dataflow[i] == "x":
			idxs[i] = 1
		elif dataflow[i] == "ci":
			idxs[i] = 2
		elif dataflow[i] == "co":
			idxs[i] = 3
	
	vals[idxs[0]] -= 1
	if vals[idxs[0]] < 0:
		vals[idxs[0]] = maxs[idxs[0]] - 1
		vals[idxs[1]] -= 1
		if vals[idxs[1]] < 0:
			vals[idxs[1]] = maxs[idxs[1]] - 1
			vals[idxs[2]] -= 1
			if vals[idxs[2]] < 0:
				vals[idxs[2]] = maxs[idxs[2]] - 1
				vals[idxs[3]] -= 1
				if vals[idxs[3]] < 0:
					vals[idxs[3]] = maxs[idxs[3]] - 1
					# finished!
					return None
	
	return [vals[0], vals[1], vals[2], vals[3]]

def find_pe_instrs(c_x=-1, c_y=-1, c_ci=-1, c_kx=-1, c_ky=-1, c_co=-1):
	
	# check if input chunk size is a vector or matrix
	num_greater = 0
	if c_x > 1:
		num_greater += 1
	if c_y > 1:
		num_greater += 1
	if c_ci > 1:
		num_greater += 1
	if num_greater > 1:
		input_vector = False
	else:
		input_vector = True
	#print(input_vector)
	if input_vector:
		# find a and w from the input
		a = max(c_x, c_y)
		w = c_ci #max(c_x, c_y, c_ci)
		
		# find b from the weight chunk params
		res = sorted([c_ci, c_kx, c_ky, c_co], reverse=True)
		if res[0] == w:
			b = res[1]
		else:
			b = res[0]
	else:
		# find b and w from the weights
		b = max(c_kx, c_ky, c_co)
		w = c_ci
		
		# find a from the input chunk params
		res = sorted([c_x, c_y, c_ci], reverse=True)
		
		if res[0] == w:
			a = res[1]
		else:
			a = res[0]
	
	return [a, b, w]

def _calc_chunks(hardware, dataflow, start_x, start_y, start_ci, start_kx, start_ky, start_co, t_x, t_y, t_ci, t_co, c_x, c_y, c_ci, c_kx, c_ky, c_co, num_PEs, layer, load_A, load_B, load_O, save_O):
	chunk_startup_cost = 1
	if c_x > t_x:
		c_x = t_x
	if c_y > t_y:
		c_y = t_y
	if c_ci > t_ci:
		c_ci = t_ci
	if c_co > t_co:
		c_co = t_co
	
	a_size = 0
	b_size = 0
	o_size = 0
	a_names = set()
	b_names = set()
	o_names = set()
	
	# build chunks
	chunks = []
	# generate ChunkSets in the order specified by the dataflow
	starts = [start_x, start_y, start_ci, start_kx, start_ky, start_co]
	ends = [start_x + t_x, start_y + t_y, start_ci + t_ci, start_kx + layer.filter_dim, start_ky + layer.filter_dim, start_co + t_co]
	steps = [c_x, c_y, c_ci, c_kx, c_ky, c_co]
	idxs = [0, 1, 2, 3, 4, 5]
	for i in range(6):
		if dataflow[i] == "x":
			idxs[i] = 0
		elif dataflow[i] == "y":
			idxs[i] = 1
		elif dataflow[i] == "kx":
			idxs[i] = 2
		elif dataflow[i] == "ky":
			idxs[i] = 3
		elif dataflow[i] == "ci":
			idxs[i] = 4
		elif dataflow[i] == "co":
			idxs[i] = 5
	
	#print(starts)
	#print(ends)
	#print(steps)
	
	for i0 in range(starts[idxs[0]], ends[idxs[0]], steps[idxs[0]]):
		for i1 in range(starts[idxs[1]], ends[idxs[1]], steps[idxs[1]]):
			for i2 in range(starts[idxs[2]], ends[idxs[2]], steps[idxs[2]]):
				for i3 in range(starts[idxs[3]], ends[idxs[3]], steps[idxs[3]]):
					for i4 in range(starts[idxs[4]], ends[idxs[4]], steps[idxs[4]]):
						for i5 in range(starts[idxs[5]], ends[idxs[5]], steps[idxs[5]]):
							x_index = idxs.index(0)
							if x_index == 0:
								x = i0
							elif x_index == 1:
								x = i1
							elif x_index == 2:
								x = i2
							elif x_index == 3:
								x = i3
							elif x_index == 4:
								x = i4
							elif x_index == 5:
								x = i5
							
							y_index = idxs.index(1)
							if y_index == 0:
								y = i0
							elif y_index == 1:
								y = i1
							elif y_index == 2:
								y = i2
							elif y_index == 3:
								y = i3
							elif y_index == 4:
								y = i4
							elif y_index == 5:
								y = i5
							
							ci_index = idxs.index(2)
							if ci_index == 0:
								ci = i0
							elif ci_index == 1:
								ci = i1
							elif ci_index == 2:
								ci = i2
							elif ci_index == 3:
								ci = i3
							elif ci_index == 4:
								ci = i4
							elif ci_index == 5:
								ci = i5
							
							kx_index = idxs.index(3)
							if kx_index == 0:
								kx = i0
							elif kx_index == 1:
								kx = i1
							elif kx_index == 2:
								kx = i2
							elif kx_index == 3:
								kx = i3
							elif kx_index == 4:
								kx = i4
							elif kx_index == 5:
								kx = i5
							
							ky_index = idxs.index(4)
							if ky_index == 0:
								ky = i0
							elif ky_index == 1:
								ky = i1
							elif ky_index == 2:
								ky = i2
							elif ky_index == 3:
								ky = i3
							elif ky_index == 4:
								ky = i4
							elif ky_index == 5:
								ky = i5
							
							co_index = idxs.index(5)
							if co_index == 0:
								co = i0
							elif co_index == 1:
								co = i1
							elif co_index == 2:
								co = i2
							elif co_index == 3:
								co = i3
							elif co_index == 4:
								co = i4
							elif co_index == 5:
								co = i5
							
							lx = x
							rx = x + c_x
							if rx > start_x + t_x:
								rx = start_x + t_x
							ly = y
							ry = y + c_y
							if ry > start_y + t_y:
								ry = start_y + t_y
							
							# check leftmost x
							if kx > lx:
								lx = kx
							# check rightmost x
							if rx > layer.pcols - layer.filter_dim + kx + 1:
								rx = layer.pcols - layer.filter_dim + kx + 1
							
							# check leftmost y
							if ky > ly:
								ly = ky
							# check rightmost y
							if ry > layer.prows - layer.filter_dim + ky + 1:
								ry = layer.prows - layer.filter_dim + ky + 1
							
							# check for kx and ky
							if c_kx + kx > start_kx + layer.filter_dim:
								c_kx = start_kx + layer.filter_dim - kx
							if c_ky + ky > start_ky + layer.filter_dim:
								c_ky = start_ky + layer.filter_dim - ky
							
							# check for ci and co
							if c_ci + ci > start_ci + t_ci:
								c_ci = start_ci + t_ci - ci
							if c_co + co > start_co + t_co:
								c_co = start_co + t_co - co
							
							# if this needs to be skipped, then skip it!
							if rx - lx <= 0 or ry - ly <= 0:
								#if start_y == 27:
								#	print(x, y, ry, ly)
								#	input()
								continue
							
							# generate a, b, and w
							a, b, w = find_pe_instrs(rx-lx, ry-ly, c_ci, c_kx, c_ky, c_co)
							
							chunks.append(a * b * w)
							
							# count accesses
							chunk_name_A = "A_" + str(x) + "_" + str(y) + "_" + str(ci)
							chunk_name_B = "B_" + str(ci) + "_" + str(kx) + "_" + str(ky) + "_" + str(co)
							chunk_name_O = "O_" + str(x) + "_" + str(y) + "_" + str(co)
							
							if not chunk_name_A in a_names:
								a_size += a*w
								a_names.add(chunk_name_A)
							if not chunk_name_B in b_names:
								b_size += b*w
								b_names.add(chunk_name_B)
							if not chunk_name_O in o_names:
								o_size += a*b
								o_names.add(chunk_name_O)
	#print("Num chunks:", len(chunks))
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
	
	# calculate accesses
	#print("A:", a_size, "B:", b_size, "O:", o_size)
	
	accesses = 0
	if load_A:
		accesses += a_size
	if load_B:
		accesses += b_size * hardware.num_PE_lanes
	if load_O:
		accesses += o_size * hardware.num_PE_lanes
	if save_O:
		accesses += o_size * hardware.num_PE_lanes
	
	return cycles, accesses

# params: [c_in, x/y/ci/co, x/y/ci/co, x/y/ci/co, x/y/ci/co, t_x, t_y, t_ci, t_co, x/y/ci/kx/ky/co, x/y/ci/kx/ky/co, x/y/ci/kx/ky/co, x/y/ci/kx/ky/co, x/y/ci/kx/ky/co, x/y/ci/kx/ky/co, c_x, c_y, c_ci, c_kx, c_ky, c_co]
# returns [estimated cycles, estimated power]
def estimate_performance(hardware, layer, params):
	# static variables
	tile_startup_cost = 0
	layer_startup_cost = 0
	
	output_tiles_created = set()
	
	total_cycles = layer_startup_cost
	total_accesses = 0
	
	row, col, c_in, c_out = [0, 0, 0, 0]
	num_rows = math.ceil(layer.prows / params[6])
	num_cols = math.ceil(layer.pcols / params[5])
	num_c_in = math.ceil(layer.c_in / params[7])
	num_c_out = math.ceil(math.ceil(layer.c_out / hardware.num_PE_lanes) / params[8])
	
	finished = False
	while not finished:
		#print("Starting tile")
		# process current tile
		total_cycles += tile_startup_cost
		
		# check bounds on t_X
		t_x = params[5]
		if col == num_cols - 1:
			t_x = layer.pcols - col * t_x
		t_y = params[6]
		if row == num_rows - 1:
			t_y = layer.prows - row * t_y
		t_ci = params[7]
		if c_in == num_c_in - 1:
			t_ci = layer.c_in - c_in * t_ci
		t_co = params[8]
		if c_out == num_c_out - 1:
			t_co = math.ceil(layer.c_out / hardware.num_PE_lanes) - c_out * t_co
		
		
		# generate dataflow params
		o_tile_name = str(row) + ", " + str(col) + ", " + str(c_out)
		load_A = True
		load_B = True
		load_O = True
		prev_coords = get_prev_coords(params[1:5], row, col, c_in, c_out, num_rows, num_cols, num_c_in, num_c_out)
		if prev_coords != None:
			load_A = row != prev_coords[0] or col != prev_coords[1] or c_in != prev_coords[2]
			load_B = c_in != prev_coords[2] or c_out != prev_coords[3]
			if o_tile_name in output_tiles_created: # check if O is being created from scratch
				load_O = not (row == prev_coords[0] and col == prev_coords[1] and c_out == prev_coords[3])
			else:
				load_O = False
				output_tiles_created.add(o_tile_name)
		
		save_O = True
		next_coords = get_next_coords(params[1:5], row, col, c_in, c_out, num_rows, num_cols, num_c_in, num_c_out)
		if next_coords != None:
			if row == next_coords[0] and col == next_coords[1] and c_out == next_coords[3]:
				save_O = False
		
		#print(t_co)
		cycles, accesses = _calc_chunks(hardware, params[9:15], col * params[5], row * params[6], c_in * params[7], 0, 0, c_out * params[8], t_x, t_y, t_ci, t_co, params[15], params[16], params[17], params[18], params[19], params[20], hardware.num_PEs_per_lane, layer, load_A, load_B, load_O, save_O)
		
		total_cycles += cycles
		total_accesses += accesses
		
		# get next tile
		res = get_next_coords(params[1:5], row, col, c_in, c_out, num_rows, num_cols, num_c_in, num_c_out)
		if res == None:
			finished = True
		else:
			row, col, c_in, c_out = res
	
	memory_cycles = total_accesses / hardware.off_chip_bandwidth
	#print(total_accesses, memory_cycles)
	final_cycles = max(total_cycles, memory_cycles)
	
	return [total_cycles, memory_cycles, total_accesses * hardware.unit_energy_DRAM + layer.get_flops() * hardware.unit_energy_MAC]

def bandwidth_bound_estimate(hardware, layer, params, count_offload=True):
	# static variables
	tile_overhead = 0
	elements_per_cycle = hardware.off_chip_bandwidth # / hardware.num_PE_lanes
	
	total_cycles = 0
	total_accesses = 0
	total_reads = 0
	total_writes = 0
	
	t_x = params[5]
	t_y = params[6]
	t_ci = params[7]
	t_co = math.ceil(params[8] / hardware.num_PE_lanes)
	
	num_rows = math.ceil(layer.prows / params[6])
	num_cols = math.ceil(layer.pcols / params[5])
	num_c_in = math.ceil(layer.c_in / params[7])
	num_c_out = math.ceil(math.ceil(layer.c_out / hardware.num_PE_lanes) / t_co)
	
	# for each tile
	prev_row = -1
	prev_col = -1
	prev_c_in = -1
	prev_c_out = -1
	
	curr_row, curr_col, curr_c_in, curr_c_out = [0, 0, 0, 0]
	
	o_tiles = set()
	
	while True:
		x_length = t_x if curr_col != math.ceil(layer.pcols / t_y) - 1 else layer.pcols - t_x * curr_col
		y_length = t_y if curr_row != math.ceil(layer.prows / t_y) - 1 else layer.prows - t_y * curr_row
		ci_length = t_ci if curr_c_in != math.ceil(layer.c_in / t_ci) - 1 else layer.c_in - t_ci * curr_c_in
		co_length = t_co if curr_c_out != math.ceil(math.ceil(layer.c_out / hardware.num_PE_lanes) / t_co) - 1 else math.ceil(layer.c_out / hardware.num_PE_lanes) - t_co * curr_c_out
		
		tile_cycles = 0
		
		a_tile_size = x_length * y_length * ci_length
		b_tile_size = ci_length * co_length * layer.filter_dim ** 2
		o_tile_size = x_length * y_length * co_length
		
		print("A:", a_tile_size, "B:", b_tile_size, "O:", o_tile_size)
		
		# check if A and B tiles need to be loaded
		o_tile_name = str(curr_row) + ", " + str(curr_col) + ", " + str(curr_c_out)
		if prev_row != -1:
			load_A = prev_row != curr_row or prev_col != curr_col or prev_c_in != curr_c_in
			load_B = prev_c_in != curr_c_in or prev_c_out != curr_c_out
			if o_tile_name in o_tiles: # check if O is being created from scratch
				load_O = prev_row != curr_row or prev_col != curr_col or prev_c_out != curr_c_out
			else:
				load_O = False
				o_tiles.add(o_tile_name)
		else:
			o_tiles.add(o_tile_name)
			load_A = True
			load_B = True
			load_O = False
		
		# check if O tile needs to be saved
		res = get_next_coords(params[1:5], curr_row, curr_col, curr_c_in, curr_c_out, num_rows, num_cols, num_c_in, num_c_out)
		if res != None:
			next_row, next_col, next_c_in, next_c_out = res
			save_O = (curr_row != next_row or curr_col != next_col or curr_c_out != next_c_out)
		else:
			save_O = True
		
		print(curr_col, curr_row, curr_c_in, curr_c_out)
		print("Load A:", load_A)
		print("Load B:", load_B)
		print("Load O:", load_O)
		print("Save O:", save_O)
		
		# load data
		if load_A:
			tile_cycles += a_tile_size / elements_per_cycle
			total_accesses += a_tile_size
			total_reads += a_tile_size
		if load_B:
			tile_cycles += b_tile_size / elements_per_cycle
			total_accesses += b_tile_size
			total_reads += b_tile_size
		if load_O:
			tile_cycles += o_tile_size / elements_per_cycle
			total_accesses += o_tile_size
			total_reads += o_tile_size
		
		# store data
		if count_offload and save_O:
			tile_cycles += o_tile_size / elements_per_cycle
			total_accesses += o_tile_size
			total_writes += o_tile_size
		#print(tile_cycles, a_tile_size, b_tile_size, o_tile_size, a_tile_size / elements_per_cycle, b_tile_size / elements_per_cycle, o_tile_size / elements_per_cycle)
		total_cycles += tile_cycles
		
		# prep for next tile
		if res == None:
			break
		prev_row = curr_row
		prev_col = curr_col
		prev_c_in = curr_c_in
		prev_c_out = curr_c_out
		curr_row = next_row
		curr_col = next_col
		curr_c_in = next_c_in
		curr_c_out = next_c_out
	
	print("Reads:", total_reads)
	print("Writes:", total_writes)
	
	return [total_cycles, total_accesses]

def test():
	hardware = bh.Hardware(num_PE_lanes=8, num_PEs_per_lane=128, num_RFs_per_PE=11, size_RF=10, off_chip_bandwidth=100, on_chip_bandwidth=10, total_sram_size=320000//2 - 512*11*10)
	model = bm.get_DeiT_Tiny(1, 1.0, 1.0, 0.0)
	layer_set = bm.model_to_layer_set(model)
	#params = ['c_out', 'ci', 'x', 'y', 'co', 72, 177, 2, 156*8, 'kx', 'ci', 'ky', 'x', 'y', 'co', 1, 4, 2, 2, 1, 1]
	#params = ['c_out', 'x', 'co', 'ci', 'y', 28, 15, 2, 191, 'ky', 'y', 'ci', 'x', 'kx', 'co', 1, 1, 2, 1, 5, 1]
	params = ['c_out', 'y', 'co', 'ci', 'x', 25, 9, 2, 191, 'kx', 'ci', 'x', 'y', 'ky', 'co', 1, 2, 2, 1, 6, 1]
	#params = ['c_out', 'x', 'y', 'ci', 'co', 5, 5, 1, 1, 'x', 'y', 'ci', 'kx', 'ky', 'co', 1, 1, 1, 1, 1, 1]
	
	layer = layer_set.unique_layers[0][0]
	layer.print()
	#layer = ch.ConvLayer(rows=10, cols=10, c_in=1, c_out=1, filter_dim=3, step_size=1)
	
	#print(layer.get_flops())
	#layer = bm.layer
	#layer.print()
	
	#dc.run_conv_dataflow(hardware, layer, params)
	print(estimate_performance(hardware, layer, params))
	#print(bandwidth_bound_estimate(hardware, layer, params))
	
	#res = dw.run_layer(hardware, params, layer, preload_cycles=0, pipeline_offloading=False)
	#print(res)

def timeit_test():
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
