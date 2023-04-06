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
		
		self.hw_iters = 2
		self.param_iters = 2
	
	# takes the model configuration, and returns [hw_config, params, latency, power] of a randomly-generated accelerator running that model
	def generate_random_accel(self, model_config):
		model = bm.create_nasvit_from_config(model_config)
		layer_set = bm.model_to_layer_set(model)
		
		best_hw = None
		best_params = None
		best_cycles_overall = -1
		best_dram_accesses_overall = -1
		
		for h in range(hw_iters):
			hw = random.choice(es.generate_hardware_configs(self.total_num_PEs, self.total_memory, self.clock_speed, self.bandwidth))
			params = []
			total_cycles = 0
			total_dram_accesses = 0
			
			for idx, (layer, count) in layer_set.unique_layers:
				
				best_param = None
				best_cycles = -1
				best_dram_accesses = -1
				
				for p in range(param_iters):
					param = es.generate_random_param(hw, layer)
					cycles, dram_accesses, _, _, _ = dw.run_layer(hw, param, layer, estimate=True)
					
					if best_param == None or cycles < best_cycles:
						best_param = param
						best_cycles = cycles
						best_dram_accesses = dram_accesses
				
				params.append(best_param)
				total_cycles += best_cycles
				total_dram_accesses += best_dram_accesses
			
			if best_hw == None or total_cycles < best_cycles_overall:
				best_hw = hw
				best_params = params
				best_cycles_overall = total_cycles
				best_dram_accesses_overall = total_dram_accesses
		
		# Old way (no estimate, just one random guess)
		#hw = random.choice(es.generate_hardware_configs(self.total_num_PEs, self.total_memory, self.clock_speed, self.bandwidth))
		#params = [es.generate_random_param(hw, layer) for (layer, count) in layer_set.unique_layers]
		
		#cycles, dram_accesses = dw.run_layer_set_no_pipelining(hw, params, layer_set)
		
		return [best_hw, best_params, best_cycles_overall / self.clock_speed / 1000000000, best_dram_accesses_overall]
	
	# does a full evaluation of the given model run on the given accelerator (if this is the same as the latency & power generated earlier, can just pass the inputs directly to the outputs)
	def full_eval(self, model_config, hw_config, param_config, prev_latency, prev_power):
		model = bm.create_nasvit_from_config(model_config)
		layer_set = bm.model_to_layer_set(model)
		layer_set_final = es.evaluate_results(hw_config, param_config, layer_set)
		return [layer_set_final.get_total_cycles() / self.clock_speed / 1000000000, layer_set_final.get_actual_memory_accesses] # TODO: actually run the full eval