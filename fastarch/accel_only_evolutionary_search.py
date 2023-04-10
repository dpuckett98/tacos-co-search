import random
import math

import generic_evolutionary_search as ges
import random_search as rs
import build_models_v2 as bm
import build_hardware_v2 as bh
import dataflow_wrapper as dw

# global settings
num_PEs = 512
on_chip_memory = 320000 // 2
clock_speed = 0.5
bandwidth = 77
model = bm.get_LeViT_128(1, 1.0, 1.0, 0.0, ViTCoD=True)
layer_set = bm.model_to_layer_set(model)

# search parameters
param_search_iters = 50
PE_lane_options = [4, 8, 16]
num_RFs_options = [i for i in range(2, 21)]
size_RFs_options = [i for i in range(1, 21)]
min_sram_size = on_chip_memory // 2

# returns [[cycles, power], [hw_config, param_list]]
def sample_and_eval():
	# generate random hw config
	hw_config = random.choice(rs.generate_hardware_configs(num_PEs, on_chip_memory, clock_speed, bandwidth))
	
	# generate params
	param_list = []
	total_cycles = 0
	total_dram_accesses = 0
	for i, (layer, count) in enumerate(layer_set.unique_layers):
		print("***"*7)
		print("Starting layer", i+1, "out of", len(layer_set.unique_layers))
		layer.print()
		print("***"*7)
	
		params, cycles, dram_accesses = rs.search_single_layer(hw_config, layer, param_search_iters, 1, rs.latency_cost)
		param_list.append(params)
		total_cycles += cycles * count
		total_dram_accesses += dram_accesses * count
	
	return [[total_cycles, total_dram_accesses], [hw_config, param_list]]

# returns [[cycles, power], [hw_config, param_list]]
def mutate_and_eval(entity, mutate_rate):
	# duplicate hw config
	hw_config = bh.Hardware(entity[1][0].num_PE_lanes, entity[1][0].num_PEs_per_lane, entity[1][0].num_RFs_per_PE, entity[1][0].size_RF, entity[1][0].off_chip_bandwidth, entity[1][0].on_chip_bandwidth, entity[1][0].total_sram_size)
	
	# mutate hw config
	
	# change num PE lanes
	if random.uniform(0, 1) < mutate_rate:
		hw_config.num_PE_lanes = random.choice(PE_lane_options)
		hw_config.num_PEs_per_lane = num_PEs // hw_config.num_PE_lanes
	
	# change num RFs per PE
	if random.uniform(0, 1) < mutate_rate:
		max_option = min(max(num_RFs_options), math.floor((on_chip_memory - min_sram_size) / num_PEs / hw_config.size_RF))
		hw_config.num_RFs_per_PE = random.choice(list(range(2, max_option)))
	
	# change size RFs
	if random.uniform(0, 1) < mutate_rate:
		max_option = min(max(size_RFs_options), math.floor((on_chip_memory - min_sram_size) / num_PEs / hw_config.num_RFs_per_PE))
		hw_config.size_RF = random.choice(list(range(1, max_option)))
	
	# update sram size
	sram_size = on_chip_memory - num_PEs * hw_config.num_RFs_per_PE * hw_config.size_RF
	
	# regenerate params
	param_list = []
	total_cycles = 0
	total_dram_accesses = 0
	for i, (layer, count) in enumerate(layer_set.unique_layers):
		print("***"*7)
		print("Starting layer", i+1, "out of", len(layer_set.unique_layers))
		layer.print()
		print("***"*7)
	
		params, cycles, dram_accesses = rs.search_single_layer(hw_config, layer, param_search_iters, 1, rs.latency_cost)
		param_list.append(params)
		total_cycles += cycles * count
		total_dram_accesses += dram_accesses * count
	
	return [[total_cycles, total_dram_accesses], [hw_config, param_list]]

# returns fitness of the entity (higher is better)
def eval_fitness(entity):
	return entity[0][0]

def test():
	pool = ges.run_evolutionary_search(pool_size=5, num_generations=5, growth_rate=0.2, mutate_rate=0.5, sample_and_eval=sample_and_eval, mutate_and_eval=mutate_and_eval, eval_fitness=eval_fitness)
	print(pool)
	for p in pool:
		print(p[0])

if __name__ == "__main__":
	test()