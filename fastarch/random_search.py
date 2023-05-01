import random
import math

from fastarch.dataflow_wrapper import run_layer, run_layer_set
from fastarch.build_hardware_v2 import Hardware
from fastarch.build_models_v2 import LayerSet, model_to_layer_set, get_DeiT_Tiny, get_DeiT_Small, get_DeiT_Base, get_test, get_LeViT_128, get_LeViT_192, get_LeViT_256, create_nasvit_supernet, create_nasvit_smallest
import fastarch.conv_helper as ch

# search function
# Inputs:
# total_num_PEs = total number of PEs
# total_memory = total on-chip memory (in half words)
# clock_speed = clock speed in GHz
# bandwidth = bandwidth in GB/s
# layer_set = set of layers to be run
# iterations = number of iterations to search each layer
# cost_function = a function that scores a layer based on it's cycles & dram accesses
def run_search(total_num_PEs, total_memory, clock_speed, bandwidth, layer_set, hw_iterations, est_iterations, full_iterations, cost_function):

	hardware_configs = generate_hardware_configs(total_num_PEs, total_memory, clock_speed, bandwidth)
	
	hardware_configs = random.sample(hardware_configs, hw_iterations)
	
	best_hw = None
	best_params = None
	best_cycles = -1
	best_dram_accesses = -1
	best_score = float('-inf')
	
	for idx, hw in enumerate(hardware_configs):
		curr_params = []
		curr_cycles = 0
		curr_dram_accesses = 0
		
		print("***"*10)
		print("Starting hardware config", idx+1, "out of", len(hardware_configs))
		hw.print()
		print("***"*10)
		
		for i, (layer, count) in enumerate(layer_set.unique_layers):
			print("***"*7)
			print("Starting layer", i+1, "out of", len(layer_set.unique_layers))
			layer.print()
			print("***"*7)
		
			params, cycles, dram_accesses = search_single_layer(hw, layer, est_iterations, full_iterations, cost_function)
			curr_params.append(params)
			curr_cycles += cycles * count
			curr_dram_accesses += dram_accesses * count
		
		score = cost_function(curr_cycles, curr_dram_accesses)
		
		print("***"*10)
		print("Results:")
		print("Score:", score)
		print("Cycles:", curr_cycles)
		print("Dram Accesses:", curr_dram_accesses)
		print("***"*10)
		
		if score > best_score:
			print("New best score!!!")
			print("***"*10)
			best_hw = hw
			best_params = curr_params
			best_cycles = curr_cycles
			best_dram_accesses = curr_dram_accesses
			best_score = score
	
	return [best_hw, best_params, best_cycles, best_dram_accesses]

# searches "iterations" times for the best parameters for the given layer and hw config
def search_single_layer(hw, layer, est_iterations, full_iterations, cost_function):
	# first do est_iterations estimates
	print("***"*5)
	print("Starting", est_iterations, "estimates")
	print("***"*5)
	
	param_list = []
	compute_cycles_list = []
	memory_cycles_list = []
	dram_accesses_list = []
	score_list = []
	
	all_params = generate_random_param(hw, layer, est_iterations)
	
	for i in range(est_iterations):
		param = all_params[i]
		compute_cycles, memory_cycles, dram_accesses = run_layer(hw, param, layer, estimate=True)
		score = cost_function(max(compute_cycles, memory_cycles), dram_accesses)
		
		param_list.append(param)
		compute_cycles_list.append(compute_cycles)
		memory_cycles_list.append(memory_cycles)
		dram_accesses_list.append(dram_accesses)
		score_list.append(score)
	
	print("***"*5)
	print("Finished estimates")
	print("***"*5)
	
	# find the best estimates
	combined = sorted(zip(score_list, param_list, compute_cycles_list, memory_cycles_list, dram_accesses_list), reverse=True)
	#print(combined)
	# rerun the full_iterations best estimates & return the best one
	best_param = None
	best_cycles = -1
	best_dram_accesses = -1
	best_score = (float('-inf'))
	
	if full_iterations > 1:
		for i in range(full_iterations):
			param = combined[i][1]
			cycles, dram_accesses, _, _, _ = run_layer(hw, param, layer, estimate=False)
			score = cost_function(cycles, dram_accesses)
			
			print("***"*5)
			print("Finished Iteration", i+1, "out of", full_iterations)
			print("Score:", score)
			print("Cycles:", cycles)
			print("Dram Accesses:", dram_accesses)
			print("***"*5)
			
			if score > best_score:
				print("New best score!!!")
				print("***"*5)
				best_param = param
				best_cycles = cycles
				best_dram_accesses = dram_accesses
				best_score = score
	else:
		best_param = combined[0][1]
		best_compute_cycles = combined[0][2]
		best_memory_cycles = combined[0][3]
		best_dram_accesses = combined[0][4]
		best_score = combined[0][0]
	
	return [best_param, best_compute_cycles, best_memory_cycles, best_dram_accesses]

# generates all possible hardware configs; returns list of Hardware
def generate_hardware_configs(total_num_PEs, total_memory, clock_speed, bandwidth):
	res = []
	
	off_chip_bandwidth = bandwidth / 2 / clock_speed
	on_chip_bandwidth = 100

	min_PE_lanes = 2 # 2^2 = 4
	max_PE_lanes = 4 # 2^4 = 16
	min_sram_size = total_memory // 2
	min_RFs_per_PE = 2
	soft_max_RFs_per_PE = 20
	min_elems_per_RF = 1
	soft_max_elems_per_RF = 20
	max_RFs_per_PE = (total_memory - min_sram_size) // (total_num_PEs * min_elems_per_RF)
	
	for i in range(min_PE_lanes, max_PE_lanes+1):
		num_PE_lanes = 2 ** i
		num_PEs_per_lane = total_num_PEs // num_PE_lanes
		
		for num_RFs_per_PE in range(min_RFs_per_PE, min(soft_max_RFs_per_PE, max_RFs_per_PE)+1):
			max_elems_per_RF = (total_memory - min_sram_size) // (total_num_PEs * num_RFs_per_PE)
			
			for num_elems_per_RF in range(min_elems_per_RF, min(soft_max_elems_per_RF, max_elems_per_RF)+1):
				sram_size = total_memory - total_num_PEs * num_RFs_per_PE * num_elems_per_RF
				
				hw = Hardware(num_PE_lanes=num_PE_lanes, num_PEs_per_lane=num_PEs_per_lane, num_RFs_per_PE=num_RFs_per_PE, size_RF=num_elems_per_RF, off_chip_bandwidth=off_chip_bandwidth, on_chip_bandwidth=on_chip_bandwidth, total_sram_size=sram_size)
				
				res.append(hw)
	
	return res

# generates a random parameter set given a specific hw and layer
def generate_random_param(hw, layer, count):
	if isinstance(layer, ch.ConvLayer):
		return ch.generate_random_param(hw, layer, count)
	
	results = []
	
	tiling_choices_rows = []
	for t_a in range(hw.num_PE_lanes, layer.A_rows+1):
		for t_b in range(1, layer.B_cols+1):
			min_w = max(1, math.floor((0.9 * hw.total_sram_size - t_a * t_b) / (t_a + t_b)))
			max_w = min(layer.A_cols_B_rows, math.ceil((hw.total_sram_size - t_a * t_b) / (t_a + t_b)))
			for t_w in range(min_w, max_w + 1):
				#size = t_a * t_w + t_b * t_w + t_a * t_b
				# tiling must use at least 90% of the on-chip SRAM, but not more
				#if size <= hw.total_sram_size and size >= 0.9 * hw.total_sram_size:
				tiling_choices_rows.append([t_a, t_b, t_w])
	
	tiling_choices_cols = []
	for t_a in range(1, layer.A_rows+1):
		for t_b in range(hw.num_PE_lanes, layer.B_cols+1):
			#for t_w in range(1, layer.A_cols_B_rows+1):
			#	size = t_a * t_w + t_b * t_w + t_a * t_b
				# tiling must use at least 90% of the on-chip SRAM, but not more
			#	if size <= hw.total_sram_size and size >= 0.9 * hw.total_sram_size:
			#		tiling_choices_rows.append([t_a, t_b, t_w])
			min_w = max(1, math.floor((0.9 * hw.total_sram_size - t_a * t_b) / (t_a + t_b)))
			max_w = min(layer.A_cols_B_rows, math.ceil((hw.total_sram_size - t_a * t_b) / (t_a + t_b)))
			for t_w in range(min_w, max_w + 1):
				tiling_choices_cols.append([t_a, t_b, t_w])
	
	# if none of the above choices work, then the entire matrix mult fits on-chip, but doesn't take up 90% of the on-chip SRAM
	if len(tiling_choices_rows) == 0:
		tiling_choices_rows.append([layer.A_rows, layer.B_cols, layer.A_cols_B_rows])
	if len(tiling_choices_cols) == 0:
		tiling_choices_cols.append([layer.A_rows, layer.B_cols, layer.A_cols_B_rows])
	
	for i in range(count):
		params = []
		params.append(random.choice(["rows", "cols"]))
		
		A_rows = math.ceil(layer.A_rows / hw.num_PE_lanes) if params[0] == "rows" else layer.A_rows
		B_cols = math.ceil(layer.B_cols / hw.num_PE_lanes) if params[0] == "cols" else layer.B_cols
		
		params.append(random.choice(["Output-Stationary", "A-Stationary", "B-Stationary"]))
		
		if params[0] == "rows":
			t_a, t_b, t_w = random.choice(tiling_choices_rows)
		else:
			t_a, t_b, t_w = random.choice(tiling_choices_cols)
		
		params.append(t_a)
		params.append(t_b)
		params.append(t_w)
		
		c_a = random.randint(1, hw.num_RFs_per_PE - 1)
		c_b = hw.num_RFs_per_PE - c_a
		c_w = random.randint(1, hw.size_RF)
		
		params.append(c_a)
		params.append(c_b)
		params.append(c_w)
		
		results.append(params)
	
	return results
	

# params = one param for each unique layer in layer_set
def evaluate_results(hw, params, layer_set):
	for (layer, count), param in zip(layer_set.unique_layers, params):
		layer_set.update_layer_params(layer, param)
	
	new_params = [layer.params for layer in layer_set.layers]
	
	return run_layer_set(hw, new_params, layer_set)

def latency_cost(cycles, dram_accesses):
	return -cycles

def search_model(model, hw_iterations, est_iterations, full_iterations):
	layer_set = model_to_layer_set(model)
	hw, params, cycles, dram_accesses = run_search(512, 320000 // 2, 0.5, 77, layer_set, hw_iterations, est_iterations, full_iterations, latency_cost)
	print("***" * 10)
	
	#layer_set = evaluate_results(hw, params, layer_set)
	
	print("***" * 10)
	print("Finished! Results:")
	hw.print()
	print(params)
	print("Total Cycles:", cycles) #layer_set.get_total_cycles())
	print("Total DRAM Accesses:", dram_accesses) #layer_set.get_actual_memory_accesses())
	print("***" * 10)

def test():
	model = get_DeiT_Tiny(1, 1.0, 1.0, 0.0)
	layer_set = model_to_layer_set(model)
	
	res = generate_hardware_configs(512, 320000, 0.5, 10)
	for hw in res:
		hw.print()
	print(len(res))
	
	print(generate_random_param(res[-1], layer_set.layers[0]))

if __name__ == "__main__":
	#model = get_DeiT_Tiny(1, 1.0, 1.0, 0.0)
	model = get_LeViT_128(1, 1.0, 1.0, 0.0, ViTCoD=True)
	#model = get_test(1) #get_DeiT_Tiny(1, 0.5, 0.5, 0.9)
	search_model(model, 2, 5, 1)
