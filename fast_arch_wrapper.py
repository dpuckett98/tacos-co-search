import fastarch.evolutionary_search_v3 as es
import fastarch.build_models_v2 as bm
import fastarch.dataflow_wrapper as dw

import random

class FastArchWrapper:
	# does whatever init the accelerator needs
	def init(self, total_num_PEs, total_memory, clock_speed, bandwidth):
		print("FastArchWrapper init")
		self.total_num_PEs = total_num_PEs
		self.total_memory = total_memory
		self.clock_speed = clock_speed
		self.bandwidth = bandwidth
	
	# takes the model configuration, and returns [hw_config, params, latency, power] of a randomly-generated accelerator running that model
	def generate_random_accel(self, model_config):
		model = bm.create_nasvit_from_config(model_config)
		layer_set = bm.model_to_layer_set(model)
		
		hw = random.choice(es.generate_hardware_configs(self.total_num_PEs, self.total_memory, self.clock_speed, self.bandwidth))
		params = [es.generate_random_param(hw, layer) for (layer, count) in layer_set.unique_layers]
		
		cycles, dram_accesses = dw.run_layer_set_no_pipelining(hw, params, layer_set)
		
		return [hw, params, cycles / self.clock_speed / 1000000000, 0]
	
	# does a full evaluation of the given model run on the given accelerator (if this is the same as the latency & power generated earlier, can just pass the inputs directly to the outputs)
	def full_eval(self, model_config, hw_config, param_config, prev_latency, prev_power):
		return [prev_latency, prev_power]