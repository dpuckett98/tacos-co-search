import sys
import build_models as models
import build_hardware as hardware
from dataflow_enc_dec import optimize_params, run_MM_dataflow

import random
from operator import attrgetter

class Combination:

	def __init__(self, hardware, model, params):
		self.hardware = hardware
		self.layers = models.model_to_layer_set(model)
		self.params = params
		self.accuracy = -1
		self.latency = -1
		self.dram_accesses = -1
		self.score = 0
	
	def update_score(self, latency_value, dram_accesses_value):
		self.score = latency_value / self.latency + dram_accesses_value / self.dram_accesses
	
	def evaluate_hardware(self, eval_hardware, latency_value, dram_accesses_value):
		self.latency, self.dram_accesses = eval_hardware(self.hardware, self.layers, self.params)
		self.update_score(latency_value, dram_accesses_value)
	
	def print(self):
		print("--- Combination ---")
		self.hardware.print()
		print(self.layers.get_string_stats(self.hardware.num_PEs, self.params), end='')
		print("Accuracy:", self.accuracy)
		print("Latency:", self.latency)
		print("Dram accesses:", self.dram_accesses)
		print("Score:", self.score)
	
	def get_string(self):
		res = "--- Combination ---\n"
		res += self.hardware.get_string()
		res += self.layers.get_string_stats(self.hardware.num_PEs, self.params)
		#file.write(layer_set.get_string_stats(best_hw.num_PEs, best_params))
		#file.write(best_hw.get_string())
		return res

# To Do: make a better way to generate params???
def make_random_combination(num_PEs, total_on_chip_memory, off_chip_bandwidth, on_chip_bandwidth, model):
	layers = models.model_to_layer_set(model)
	num_RFs, size_RF, buffer_size = hardware.get_on_chip_memory(num_PEs, total_on_chip_memory, layers)
	
	params = []
	
	for val in layers.unique_layers:
		layer = val[0]
		dataflow = hardware.get_dataflow(layer)
		t_a, t_b, t_w = hardware.get_tile_parameters(buffer_size, layer)
		c_a, c_b, c_w = hardware.get_chunk_parameters(t_a, t_b, t_w, num_RFs, size_RF, layer)
		params.append([dataflow, t_a, t_b, t_w, c_a, c_b, c_w])

	hw = hardware.Hardware(num_PEs, num_RFs, size_RF, off_chip_bandwidth, on_chip_bandwidth, buffer_size, None)

	return Combination(hw, model, params)

def mutate_combination(combination, num_PEs, total_on_chip_memory, model, mutate_chance):
	new_comb = Combination(combination.hardware, model, combination.params)
	layers = models.model_to_layer_set(model)
	
	# run through each configuration & its possibilities
	change_hw = random.choice([True, False])
	if change_hw:
		num_RFs, size_RF, buffer_size = hardware.get_on_chip_memory(num_PEs, total_on_chip_memory, layers)
		new_comb.hardware.num_RFs_per_PE = num_RFs
		new_comb.hardware.size_RF = size_RF
		new_comb.hardware.max_sram_size = buffer_size
	
	params = new_comb.params
	
	for idx, val in enumerate(layers.unique_layers):
		layer = val[0]
		
		change_dataflow = random.choice([True, False])
		if change_dataflow:
			dataflow = hardware.get_dataflow(layer)
			params[idx][0] = dataflow
		
		change_tiles = random.choice([True, False]) or change_hw
		if change_tiles:
			t_a, t_b, t_w = hardware.get_tile_parameters(new_comb.hardware.max_sram_size, layer)
			params[idx][1] = t_a
			params[idx][2] = t_b
			params[idx][3] = t_w
		
		change_chunks = random.choice([True, False]) or change_tiles
		if change_chunks:
			c_a, c_b, c_w = hardware.get_chunk_parameters(params[idx][1], params[idx][2], params[idx][3], new_comb.hardware.num_RFs_per_PE, new_comb.hardware.size_RF, layer)
			params[idx][4] = c_a
			params[idx][5] = c_b
			params[idx][6] = c_w
	
	new_comb.params = params
	
	return new_comb
	

def accelerator_evolutionary_search(num_PEs, total_on_chip_memory, off_chip_bandwidth, on_chip_bandwidth, model, eval_hardware, latency_value, dram_accesses_value, pool_size, iterations, rate, mutate_chance, file_name=None):

	# generate initial pool of combinations (sorted list?)
	pool = [make_random_combination(num_PEs, total_on_chip_memory, off_chip_bandwidth, on_chip_bandwidth, model) for i in range(pool_size)]
	
	# evaluate each combination using eval_hardware function
	for i in range(len(pool)):
		pool[i].evaluate_hardware(eval_hardware, latency_value, dram_accesses_value)
		'''
		going = True
		while going:
			going = False
			try:
				pool[i].evaluate_hardware(eval_hardware, latency_value, dram_accesses_value)
			except:
				pool[i] = make_random_combination(num_PEs, total_on_chip_memory, off_chip_bandwidth, on_chip_bandwidth, model)
				going = True
		#p.evaluate_hardware(eval_hardware, latency_value, dram_accesses_value)
		'''
	
	# for each iteration
	for i in range(iterations):
		print("\n\nIteration", i, "\n\n")
		# sort pool
		pool.sort(key=attrgetter('score'))
		
		# mutate top 'rate' combinations, evaluate their performance, and add them to the pool
		new_pool = []
		for i in range(pool_size - rate, pool_size):
			new_pool.append(mutate_combination(pool[i], num_PEs, total_on_chip_memory, model, mutate_chance))
		for i in range(len(pool)):
			pool[i].evaluate_hardware(eval_hardware, latency_value, dram_accesses_value)
		'''
		for i in range(len(new_pool)):
			going = True
			while going:
				going = False
				try:
					new_pool[i].evaluate_hardware(eval_hardware, latency_value, dram_accesses_value)
				except:
					new_pool[i] = make_random_combination(num_PEs, total_on_chip_memory, off_chip_bandwidth, on_chip_bandwidth, model)
					going = True
		'''
		pool = pool + new_pool
		
		# sort pool
		pool.sort(key=attrgetter('score'))
		
		# remove lowest 'rate' combinations
		del pool[:rate]
	
	for p in pool:
		p.print()
		print()
	
	# if file_name != None, save the pool of combinations to file_name
	if file_name != None:
		file = open(file_name, 'a')
		for c in pool:
			file.write(c.get_string())
			file.write('\n')
		file.close()
	
	# return the pool of combinations
	return pool

def random_eval(hardware, layers, params):
	return [random.random() * 10, random.random() * 10]

def run_layers(hardware, layers, params):
	for res, param in zip(layers.unique_layers, params):
		layer, count = res
		print("-------------------- Running layer -----------------------------")
		hardware.print()
		layer.print()
		print(param)
		cycles, dram_accesses = run_MM_dataflow(hardware.num_PEs, hardware.num_RFs_per_PE, hardware.size_RF, hardware.off_chip_bandwidth, hardware.on_chip_bandwidth, hardware.max_sram_size, layer.A_rows, layer.A_cols_B_rows, layer.B_cols, param[0], param[1], param[2], param[3], param[4], param[5], param[6], estimate=True, encode=layer.encode, decode=layer.decode, orig_heads=layer.orig_head_dim, comp_heads=layer.comp_head_dim, sparsity=layer.sparsity)
		
		layers.update_layer_latency(layer, cycles)
		layers.update_layer_dram_accesses(layer, dram_accesses)
	
	return [layers.get_total_cycles(), layers.get_actual_memory_accesses()]

# returns [number of cycles, ideal number of cycles]
def run_layer(hardware, layer):
	dataflow, t_i, t_w, t_d, c_i, c_w, c_d = hardware.get_tiling_parameters(layer)
	cycles = run_MM_dataflow(hardware.num_PEs, hardware.num_RFs_per_PE, hardware.size_RF, hardware.off_chip_bandwidth, hardware.on_chip_bandwidth, hardware.max_sram_size, layer.A_rows, layer.A_cols_B_rows, layer.B_cols, dataflow, t_i, t_w, t_d, c_i, c_w, c_d, estimate=True, encode=layer.encode, decode=layer.decode, orig_heads=layer.orig_head_dim, comp_heads=layer.comp_head_dim, sparsity=layer.sparsity)
	return [cycles, dataflow, t_i, t_w, t_d, c_i, c_w, c_d]
	#params = optimize_params(total_memory, num_banks_per_PE, size_banks_per_PE, A_rows, B_cols, A_cols_B_rows)
	#return [run_MM_dataflow(A_rows, B_cols, A_cols_B_rows, params, num_PEs), A_rows * A_cols_B_rows * B_cols / num_PEs]

def run_layer_set(hardware, layer_set):
	for layer in layer_set.unique_layers:
		cycles, dataflow, t_i, t_w, t_d, c_i, c_w, c_d = run_layer(hardware, layer[0])
		layer_set.update_layer_latency(layer[0], cycles)
	
	return [layer_set.get_total_cycles(), layer_set.get_total_flops() / hardware.num_PEs, dataflow, t_i, t_w, t_d, c_i, c_w, c_d]

def random_search(num_PEs, total_on_chip_memory, model, hw_iter=100, param_iter=10, file_name=None):
	layer_set = models.model_to_layer_set(model)
	num_unique = len(layer_set.unique_layers)
	
	best_cycles = 10000000000000000
	best_hw = None
	best_params = []
	
	for i in range(hw_iter):
		hw = hardware.generate_random_config(num_PEs, total_on_chip_memory, 100, 10, hardware.random_tiling_generator)
		
		layer_params = []
		for idx, layer in enumerate(layer_set.unique_layers):
			best = 10000000000000
			params = None
			for j in range(param_iter):
				cycles, dataflow, t_i, t_w, t_d, c_i, c_w, c_d = run_layer(hw, layer[0])
				if cycles < best:
					best = cycles
					params = [dataflow, t_i, t_w, t_d, c_i, c_w, c_d]
				print(i, "-", j, "-", idx, "out of", num_unique)
				#sys.stdout.flush()
			layer_set.update_layer_latency(layer[0], best)
			layer_params.append(params)
			#print("Best:", best, layer[0].get_utilization(hw.num_PEs))
		
		curr_cycles = layer_set.get_total_cycles()
		if curr_cycles < best_cycles:
			best_cycles = curr_cycles
			best_params = layer_params
			best_hw = hw
	
	#cycles, ideal_cycles = run_layer_set(hw, layer_set)
	
	#total_cycles = layer_set.get_total_cycles()
	
	ideal_cycles = layer_set.get_total_flops() / hw.num_PEs
	
	print("---------------------------------")
	layer_set.print_stats(best_hw.num_PEs)
	best_hw.print()
	print("Params:", best_params)
	print("---------------------------------")
	print("Total cycles:", best_cycles)
	print("Ideal cycles:", int(ideal_cycles))
	print("Utilization: {:.2f}%".format(ideal_cycles / best_cycles * 100))
	sys.stdout.flush()
	
	if file_name != None:
		file = open(file_name, 'a')
		file.write(layer_set.get_string_stats(best_hw.num_PEs, best_params))
		file.write(best_hw.get_string())
		file.write("--------------------------------\n")
		file.write("Total cycles: " + str(best_cycles) + "\n")
		file.write("Ideal cycles: " + str(ideal_cycles) + "\n")
		file.write("Utilization: {:.2f}%\n".format(ideal_cycles / best_cycles * 100))
		file.close()
	#layer_set_descrip = layer_set.get_string_stats(best_hw.num_PEs)
	#hw_descrip = best_hw.get_string()
	#print(layer_set_descrip)
	#print(hw_descrip)

if __name__ == "__main__":
	
	#hw_iter = 10
	#param_iter = 100
	#run = 2
	
	#accelerator_evolutionary_search(512, 320000 / 2, 100, 10, models.get_DeiT_Tiny(1, 1.0, 0.0), run_layers, 1, 0, 10, 10, 2, 0.5, file_name="deit_tiny_run_3.txt")
	#accelerator_evolutionary_search(512, 320000, 100, 10, models.get_DeiT_Small(1, 0.5, 0.5), run_layers, 1, 0, 10, 100, 2, 0.5, file_name="deit_small_run_2.txt")
	accelerator_evolutionary_search(512, 320000 / 2, 100, 10, models.get_DeiT_Base(1, 0.5, 0.5), run_layers, 1, 0, 10, 10, 2, 0.5, file_name="deit_base_run_3.txt")
	#accelerator_evolutionary_search(512, 320000, 100, 10, models.get_LeViT_128(1, 0.5, 0.5), run_layers, 1, 0, 10, 100, 2, 0.5, file_name="levit_128_run_2.txt")
	#accelerator_evolutionary_search(512, 320000 / 2, 100, 10, models.get_LeViT_192(1, 0.5, 0.5), run_layers, 1, 0, 10, 100, 2, 0.5, file_name="levit_192_run_2.txt")
	#accelerator_evolutionary_search(512, 320000, 100, 10, models.get_LeViT_256(1, 0.5, 0.5), run_layers, 1, 0, 10, 100, 2, 0.5, file_name="levit_256_run_2.txt")
	#accelerator_evolutionary_search(512, 320000, 100, 10, models.get_LeViT_384(1, 0.5, 0.5), run_layers, 1, 0, 10, 100, 2, 0.5, file_name="levit_384_run_2.txt")
	
	#random_search(512, 320000 / 16, models.get_DeiT_Tiny(1, 0.33, 0.0), hw_iter=hw_iter, param_iter=param_iter, file_name="deit_tiny_run_" + str(run))
	#random_search(512, 320000 / 16, models.get_DeiT_Small(1), hw_iter=hw_iter, param_iter=param_iter)
	#random_search(512, 320000 / 16, models.get_DeiT_Tiny(1), hw_iter=hw_iter, param_iter=param_iter)
	#random_search(512, 320000 / 16, models.get_LeViT_128(1), hw_iter=hw_iter, param_iter=param_iter)
	#random_search(512, 320000 / 16, models.get_LeViT_256(1), hw_iter=hw_iter, param_iter=param_iter)
	#random_search(512, 320000 / 16, models.get_LeViT_384(1), hw_iter=hw_iter, param_iter=param_iter)
	#run_model(models.get_AlexNet(1))