import sys
import build_models as models
import build_hardware as hardware
from dataflow_enc_dec import optimize_params, run_MM_dataflow

import random
from operator import attrgetter

from datetime import datetime

class Combination:

	def __init__(self, hardware, model, params):
		self.hardware = hardware
		self.layers = models.model_to_layer_set(model)
		self.params = params
		self.accuracy = -1
		self.latency = -1
		self.dram_accesses = -1
		self.score = 0
	
	def update_stats(self, latency, dram_accesses):
		self.latency = latency
		self.dram_accesses = dram_accesses
	
	def update_score(self, latency_value, dram_accesses_value):
		self.score = latency_value / self.latency + dram_accesses_value / self.dram_accesses
	
	#def evaluate_hardware(self, eval_hardware, latency_value, dram_accesses_value):
	#	self.latency, self.dram_accesses = eval_hardware(self.hardware, self.layers, self.params)
	#	self.update_score(latency_value, dram_accesses_value)
	
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

def eval_single_layer(hardware, layer, param, estimate=True):
	print("-------------------- Running layer -----------------------------")
	hardware.print()
	layer.print()
	print(param)
	cycles, dram_accesses = run_MM_dataflow(hardware.num_PEs, hardware.num_RFs_per_PE, hardware.size_RF, hardware.off_chip_bandwidth, hardware.on_chip_bandwidth, hardware.max_sram_size, layer.A_rows, layer.A_cols_B_rows, layer.B_cols, param[0], param[1], param[2], param[3], param[4], param[5], param[6], estimate=estimate, encode=layer.encode, decode=layer.decode, orig_heads=layer.orig_head_dim, comp_heads=layer.comp_head_dim, sparsity=layer.sparsity)
	
	return [cycles, dram_accesses]

def run_layers(hardware, layers, params, estimate=True):
	print("-------------------- Evaluating full model -----------------------------")
	for res, param in zip(layers.unique_layers, params):
		layer, count = res
		print("-------------------- Running layer -----------------------------")
		hardware.print()
		layer.print()
		print(param)
		cycles, dram_accesses = run_MM_dataflow(hardware.num_PEs, hardware.num_RFs_per_PE, hardware.size_RF, hardware.off_chip_bandwidth, hardware.on_chip_bandwidth, hardware.max_sram_size, layer.A_rows, layer.A_cols_B_rows, layer.B_cols, param[0], param[1], param[2], param[3], param[4], param[5], param[6], estimate=estimate, encode=layer.encode, decode=layer.decode, orig_heads=layer.orig_head_dim, comp_heads=layer.comp_head_dim, sparsity=layer.sparsity)
		
		layers.update_layer_latency(layer, cycles)
		layers.update_layer_dram_accesses(layer, dram_accesses)
	
	print("-------------------- Full model evaluation finished -----------------------------")
	print("Total utilization: {:.2f}%".format(((layers.get_total_flops() / hardware.num_PEs) / layers.get_total_cycles()) * 100))
	
	return [layers.get_total_cycles(), layers.get_actual_memory_accesses()]


def generate_all_hardware_configs(num_PEs, total_on_chip_memory, off_chip_bandwidth, on_chip_bandwidth, model):
	layers = models.model_to_layer_set(model)
	configs = hardware.get_potential_hardware_configs(512, 320000/2, layers)
	
	results = []
	
	for c in configs:
		num_RFs = c[0]
		size_RF = c[1]
		buffer_size = int(total_on_chip_memory - num_RFs * size_RF * num_PEs)
		hw = hardware.Hardware(num_PEs, num_RFs, size_RF, off_chip_bandwidth, on_chip_bandwidth, buffer_size, None)
		results.append(hw)
	
	return results

def accelerator_deep_search(num_PEs, total_on_chip_memory, off_chip_bandwidth, on_chip_bandwidth, model, eval_hardware, latency_value, dram_accesses_value, iterations, param_iters, rate, mutate_chance, file_name=None):

	# generate pool of hardware settings
	hardware_configs = generate_all_hardware_configs(num_PEs, total_on_chip_memory, off_chip_bandwidth, on_chip_bandwidth, model)
	layers = models.model_to_layer_set(model)
	pool = [Combination(hw, model, [[] for i in range(len(layers.unique_layers))]) for hw in hardware_configs]
	#pool = [make_random_combination(num_PEs, total_on_chip_memory, off_chip_bandwidth, on_chip_bandwidth, model) for i in range(pool_size)]
	
	print("-------------------- Evaluating initial pool -----------------------------\n\n")
	
	# for each element in the pool:
	for p in pool:
		# for each layer in the model:
		for idx, l_c in enumerate(p.layers.unique_layers):
			l = l_c[0]
			best_score = 0
			best_params = []
			best_cycles = -1
			best_accesses = -1
			# for n iterations:
			for i in range(param_iters):
				# generate a random set of parameters & evaluate its performance; save the best one
				params = get_random_params(p, idx)
				cycles, dram_accesses = eval_hardware(p.hardware, l, params)
				score = latency_value / cycles + dram_accesses_value / dram_accesses
				if score > best_score:
					best_score = score
					best_params = params
					best_cycles = cycles
					best_accesses = dram_accesses
			p.params[idx] = best_params
			p.layers.update_layer_latency(l, best_cycles)
			p.layers.update_layer_dram_accesses(l, best_accesses)
		#cycles, accesses = run_layers(p.hardware, p.layers, p.params)
		p.update_stats(p.layers.get_total_cycles(), p.layers.get_actual_memory_accesses())
		p.update_score(latency_value, dram_accesses_value)
	
	# rank the combinations
	pool.sort(key=attrgetter('score'))
	
	print("Writing to memory...")
	# if file_name != None, save the pool of combinations to file_name
	if file_name != None:
		file = open(file_name, 'a')
		file.write("-------------------- Initial Pool -----------------------------\n")
		file.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "\n")
		for c in pool:
			file.write(c.get_string())
			file.write('\n')
		file.close()
	print("Finished writing to memory")
	
	# for each searching iteration:
	itr = 0
	while len(pool) > iterations:
		
		# print "starting iteration x"
		print("\n\n-------------------- Starting iteration", itr, "-----------------------------\n\n")
		
		# drop the lowest 'rate' combinations
		del pool[:min(rate, len(pool) - iterations)]
		
		# search again on the rest of the results
		for p in pool:
			# for each layer in the model:
			for idx, l_c in enumerate(p.layers.unique_layers):
				l = l_c[0]
				best_cycles = l.get_actual_cycles()
				best_accesses = l.get_actual_memory_accesses()
				best_score = latency_value / best_cycles + dram_accesses_value / best_accesses
				best_params = p.params[idx]
				# for n iterations:
				for i in range(param_iters):
					# generate a random set of parameters & evaluate its performance; save the best one
					params = get_random_params(p, idx)
					cycles, dram_accesses = eval_hardware(p.hardware, l, params)
					score = latency_value / cycles + dram_accesses_value / dram_accesses
					if score > best_score:
						best_score = score
						best_params = params
						best_cycles = cycles
						best_accesses = dram_accesses
				p.params[idx] = best_params
				p.layers.update_layer_latency(l, best_cycles)
				p.layers.update_layer_dram_accesses(l, best_accesses)
			#cycles, accesses = run_layers(p.hardware, p.layers, p.params)
			p.update_stats(p.layers.get_total_cycles(), p.layers.get_actual_memory_accesses())
			p.update_score(latency_value, dram_accesses_value)
		
		# rank the combinations
		pool.sort(key=attrgetter('score'))
		
		# print "finished iteration x; best utilization is y"
		print("\n\n-------------------- Finished iteration", itr, "-----------------------------")
		print("Best utilization: {:.2f}%".format(((pool[len(pool)-1].layers.get_total_flops() / pool[len(pool)-1].hardware.num_PEs) / pool[len(pool)-1].layers.get_total_cycles()) * 100))
		#print("Best utilization is", (pool[len(pool)-1].layers.get_total_flops() / pool[len(pool)-1].hardware.num_PEs) / pool[len(pool)-1].layers.get_total_cycles(), "\n\n")
		
		print("Writing to memory...")
		# if file_name != None, save the pool of combinations to file_name
		if file_name != None:
			file = open(file_name, 'a')
			file.write("-------------------- Iteration " + str(itr) + " -----------------------------\n")
			file.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "\n")
			for c in pool:
				file.write(c.get_string())
				file.write('\n')
			file.close()
		print("Finished writing to memory")
		
		itr += 1
		
	# print "finished searching; best overall utilization is y"
	print("\n\n-------------------- Finished searching -----------------------------\n\n")
	
	# print each result
	for p in pool:
		p.print()
		print()
	
	# print best utilization
	print("\nBest utilization: {:.2f}%".format(((pool[len(pool)-1].layers.get_total_flops() / pool[len(pool)-1].hardware.num_PEs) / pool[len(pool)-1].layers.get_total_cycles()) * 100))
	#print("\n\nBest utilization is", (pool[-1].layers.get_total_flops() / pool[-1].hardware.num_PEs) / pool[-1].layers.get_total_cycles())

	# if file_name != None, save the pool of combinations to file_name
	if file_name != None:
		file = open(file_name, 'a')
		file.write("-------------------- Finished searching -----------------------------\n")
		file.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "\n")
		for c in pool:
			file.write(c.get_string())
			file.write('\n')
		file.close()
	
	# return the pool of combinations
	return pool

def eval_single_layer_no_estimate(hardware, layer, param):
	print("-------------------- Running layer -----------------------------")
	hardware.print()
	layer.print()
	print(param)
	cycles, dram_accesses = run_MM_dataflow(hardware.num_PEs, hardware.num_RFs_per_PE, hardware.size_RF, hardware.off_chip_bandwidth, hardware.on_chip_bandwidth, hardware.max_sram_size, layer.A_rows, layer.A_cols_B_rows, layer.B_cols, param[0], param[1], param[2], param[3], param[4], param[5], param[6], estimate=False, encode=layer.encode, decode=layer.decode, orig_heads=layer.orig_head_dim, comp_heads=layer.comp_head_dim, sparsity=layer.sparsity)
	
	return [cycles, dram_accesses]

def evaluate_pool(pool, file_name, latency_value, dram_accesses_value):
	for p in pool:
		# for each layer in the model:
		for idx, (param, l_c) in enumerate(zip(p.params, p.layers.unique_layers)):
			l = l_c[0]
			cycles, dram_accesses = eval_single_layer_no_estimate(p.hardware, l, param)
			p.layers.update_layer_latency(l, cycles)
			p.layers.update_layer_dram_accesses(l, dram_accesses)
		p.update_stats(p.layers.get_total_cycles(), p.layers.get_actual_memory_accesses())
		p.update_score(latency_value, dram_accesses_value)
	
	pool.sort(key=attrgetter('score'))
	
	# if file_name != None, save the pool of combinations to file_name
	if file_name != None:
		file = open(file_name, 'a')
		file.write("-------------------- Finished Full Evaluation -----------------------------\n")
		file.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "\n")
		for c in pool:
			file.write(c.get_string())
			file.write('\n')
		file.close()
	
	return pool
	
if __name__ == "__main__":
	
	#layers = models.model_to_layer_set(models.get_DeiT_Tiny_Attention(1, 0.33, 0.0))
	layers = models.model_to_layer_set(models.get_DeiT_Tiny(1, 0.5, 0.9))
	possibilities = hardware.get_potential_hardware_configs(512, 320000/2, layers)
	num = 0
	for p in possibilities:
		
	print(len(possibilities))
	#layers.print()
	#hw = hardware.Hardware(10, 10, 10, 100, 10, 1000, None)
	#params = [
	#	["Output-Stationary", 20, 39, 20, 5, 5, 10],
	#	["Output-Stationary", 20, 20, 20, 5, 5, 10],
	#	["Output-Stationary", 20, 20, 7, 5, 5, 7],
	#	["Output-Stationary", 20, 7, 20, 5, 4, 10]
	#	]
	
	#cycles, dram_accesses = run_layers(hw, layers, params, estimate=False)
	
	#print(layers.get_string_stats(hw.num_PEs, params), end='')
	#layers = models.model_to_layer_set(models.get_DeiT_Small(1, 1.0, 0.0))
	#layers.print()
	
	#layer = layers.unique_layers[1][0]
	#layer.print()
	
	#layer = layers.unique_layers[1][0]
	#layer.print()
	
	#possibilities = hardware.get_potential_hardware_configs(512, 320000/2, layers)
	#print(possibilities)
	
	#num_RFs, size_RF = [5, 5]#possibilities[0]
	#print(num_RFs, size_RF)
	#t_a, t_b, t_w = [4, 363, 6]
	#print(t_a, t_b, t_w)
	#print(hardware._get_chunk_parameters(t_a, t_b, t_w, num_RFs, size_RF, layer))
	
	#hw_iter = 10
	#param_iter = 100
	#run = 2
	
	#layers = models.model_to_layer_set(models.get_LeViT_384(1, 0.5, 0.0))
	#results = hardware.get_potential_hardware_configs(512, 320000/2, layers)
	#print(len(results))
	
	# ---------------actual stuff here-------------
	#model = models.get_test() #DeiT_Tiny(1, 1.0, 0.0)
	#file_name = "test_run_1.txt"
	
	#pool = accelerator_deep_search(512, 320000 / 2, 100, 10, model, eval_single_layer, 1, 0, iterations=10, param_iters=10, rate=50, mutate_chance=0.5, file_name=file_name)
	#evaluate_pool(pool, file_name, 1, 0)
	