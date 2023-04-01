import random
import pickle

import dataflow_wrapper as dw
import evolutionary_search_v3 as es
import build_models_v2 as bm
import build_hardware_v2 as bh

def run_test(iterations, save_file, min_size=100, max_size=10000):
	
	dim_choices = list(range(min_size, max_size+1))
	
	layer_list = []
	hw_list = []
	param_list = []
	act_cycles_list = []
	est_cycles_list = []
	
	# run test
	for i in range(iterations):
		layer = bm.Layer(random.choice(dim_choices), random.choice(dim_choices), random.choice(dim_choices))
	
		hw = random.choice(es.generate_hardware_configs(512, 1000000, 0.5, 10))
		
		param = es.generate_random_param(hw, layer)
		
		est_cycles, _, _, _ = dw.run_layer(hw, param, layer, estimate=True)
		act_cycles, _, _, _ = dw.run_layer(hw, param, layer, estimate=False)
		
		layer_list.append(layer)
		hw_list.append(hw)
		param_list.append(param)
		act_cycles_list.append(est_cycles)
		est_cycles_list.append(act_cycles)
	
	# print average miss amt
	diff = [abs(i - j) / i for i, j in zip(act_cycles_list, est_cycles_list)] # difference between the two values, scaled by the actual compute time
	print("Average difference:", sum(diff) / len(diff))
	
	# save the results
	with open(save_file, 'wb') as out_file:
		pickle.dump(layer_list, out_file, pickle.HIGHEST_PROTOCOL)
		pickle.dump(hw_list, out_file, pickle.HIGHEST_PROTOCOL)
		pickle.dump(param_list, out_file, pickle.HIGHEST_PROTOCOL)
		pickle.dump(act_cycles_list, out_file, pickle.HIGHEST_PROTOCOL)
		pickle.dump(est_cycles_list, out_file, pickle.HIGHEST_PROTOCOL)

def evaluate_results(save_file):
	with open(save_file, 'rb') as in_file:
		hw_list = pickle.load(in_file)
		param_list = pickle.load(in_file)
		cycles_list = pickle.load(in_file)
		accesses_list = pickle.load(in_file)
	
	print(min(cycles_list))
	
	x_values = list(range(len(param_list))) #[p[1] for p in param_list]
	x_name = "Index"
	y_values = cycles_list
	y_name = "Cycles"
	
	plt.plot(x_values, y_values, '.')
	plt.xlabel(x_name)
	plt.ylabel(y_name)
	plt.show()

if __name__ == "__main__":
	run_test(3, "estimate_accuracy_test.txt", max_size=200)
	#evaluate_results("explore_search_space_res.txt")