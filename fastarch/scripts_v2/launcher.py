from mpi4py import MPI

import dataflow_wrapper as dw
import evolutionary_search_v3 as es
import build_models_v2 as bm
import build_hardware_v2 as bh

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# settings
search_iterations = 10

exps = [
	[bm.get_test(1), "base", 10],
	[bm.get_test(1), "search", 10],
	]

assert len(exps) == comm.Get_size()

model, run_type, bw = exps[rank]

layer_set = bm.model_to_layer_set(model)

if run_type == "base":
	hardware = bh.Hardware(8, 64, 10, 10, bw, 100, 320000//2)
	params = [['rows', 'Output-Stationary', 134*8, 134, 134, 5, 5, 10] for i in range(len(layer_set.layers))]
	layer_set_res = dw.run_layer_set(hardware, params, layer_set)
	results = [layer_set_res, params, hardware]
elif run_type == "search":
	best_hw, best_params, best_cycles, best_dram_accesses = es.run_search(512, 320000//2, 0.5, bw, layer_set, search_iterations, es.latency_cost)
	layer_set_res = es.evaluate_results(best_hw, best_params, layer_set)
	results = [layer_set_res, best_params, best_hw]

total_results = comm.gather(results, root=0)

if rank == 0:
	for (model, run_type, bw), (layer_set, params, hw) in zip(exps, total_results):
		print("***"*15)
		print(model.name, run_type, bw)
		hw.print()
		print(layer_set.get_string_stats(512, bw, params))