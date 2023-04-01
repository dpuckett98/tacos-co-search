from mpi4py import MPI

import sys
import build_models as models
import build_hardware as hardware
from dataflow4 import optimize_params, run_MM_dataflow

# returns [number of cycles, ideal number of cycles]
def run_layer(hardware, layer):
	dataflow, t_i, t_w, t_d, c_i, c_w, c_d = hardware.get_tiling_parameters(layer)
	cycles = run_MM_dataflow(hardware.num_PEs, hardware.num_RFs_per_PE, hardware.size_RF, hardware.off_chip_bandwidth, hardware.on_chip_bandwidth, hardware.max_sram_size, layer.A_rows, layer.A_cols_B_rows, layer.B_cols, dataflow, t_i, t_w, t_d, c_i, c_w, c_d, estimate=True)
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
	
	if rank == 0:
		layer_set.print()
		sys.stdout.flush()
	
	best_cycles = 10000000000000000
	best_hw = None
	best_params = []
	
	for i in range(hw_iter // world_size):
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
				print(rank, ":", i, "-", j, "-", idx, "out of", num_unique)
				sys.stdout.flush()
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
	
	result_cycles = comm.gather(best_cycles, root=0)
	result_hw = comm.gather(best_hw, root=0)
	result_params = comm.gather(best_params, root=0)
	
	if rank == 0:
		ideal_cycles = layer_set.get_total_flops() / hw.num_PEs
		
		print(result_cycles)
		print(result_hw)
		
		best_cycles = 100000000000000000000
		best_hw = None
		best_params = None
		
		for cycles, params, hw in zip(result_cycles, result_params, result_hw):
			if cycles < best_cycles:
				best_cycles = cycles
				best_params = params
				best_hw = hw
	
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
	
	comm = MPI.COMM_WORLD
	world_size = comm.Get_size()
	rank = comm.Get_rank()
	
	hw_iter = 100
	param_iter = 20
	run = 2
	
	random_search(512, 320000 / 16, models.get_DeiT_Base(1), hw_iter=hw_iter, param_iter=param_iter, file_name="../data/deit_base_run_" + str(run) + ".txt")
	random_search(512, 320000 / 16, models.get_DeiT_Small(1), hw_iter=hw_iter, param_iter=param_iter, file_name="../data/deit_small_run_" + str(run) + ".txt")
	random_search(512, 320000 / 16, models.get_DeiT_Tiny(1), hw_iter=hw_iter, param_iter=param_iter, file_name="../data/deit_tiny_run_" + str(run) + ".txt")
	random_search(512, 320000 / 16, models.get_LeViT_128(1), hw_iter=hw_iter, param_iter=param_iter, file_name="../data/levit_128_run_" + str(run) + ".txt")
	random_search(512, 320000 / 16, models.get_LeViT_256(1), hw_iter=hw_iter, param_iter=param_iter, file_name="../data/levit_256_run_" + str(run) + ".txt")
	random_search(512, 320000 / 16, models.get_LeViT_384(1), hw_iter=hw_iter, param_iter=param_iter, file_name="../data/levit_384_run_" + str(run) + ".txt")
	#run_model(models.get_AlexNet(1))