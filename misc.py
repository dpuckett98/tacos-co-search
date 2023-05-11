import math
import matplotlib.pyplot as plt

import fastarch.build_models_v2 as bm
import fastarch.build_hardware_v2 as bh
import fastarch.dataflow_wrapper as dw

def examine_nasvit_results():
	supernet_sparsity_params = [['rows', 'Output-Stationary', 4024, 6, 21, 6, 4, 10], ['rows', 'Output-Stationary', 6287, 1, 15, 8, 2, 10], ['rows', 'A-Stationary', 6346, 4, 13, 1, 9, 10], ['cols', 'Output-Stationary', 3145, 20, 12, 5, 5, 10], ['rows', 'A-Stationary', 4129, 1, 25, 3, 7, 10], ['cols', 'Output-Stationary', 1066, 19, 79, 8, 2, 10], ['cols', 'Output-Stationary', 1232, 71, 12, 9, 1, 10], ['cols', 'A-Stationary', 1263, 16, 64, 4, 6, 10], ['rows', 'A-Stationary', 1296, 1, 9, 9, 1, 10], ['cols', 'B-Stationary', 1, 48, 192, 6, 4, 10], ['cols', 'B-Stationary', 1, 192, 48, 9, 1, 10], ['cols', 'B-Stationary', 788, 22, 100, 2, 8, 10], ['cols', 'Output-Stationary', 542, 159, 32, 5, 5, 10], ['cols', 'A-Stationary', 1, 64, 240, 9, 1, 10], ['cols', 'Output-Stationary', 1, 240, 64, 7, 3, 10], ['cols', 'A-Stationary', 362, 27, 234, 9, 1, 10], ['cols', 'A-Stationary', 324, 1, 9, 4, 6, 10], ['cols', 'Output-Stationary', 309, 69, 212, 2, 8, 10], ['cols', 'Output-Stationary', 324, 72, 72, 8, 2, 10], ['cols', 'A-Stationary', 322, 289, 8, 4, 6, 10], ['cols', 'Output-Stationary', 299, 218, 67, 9, 1, 10], ['cols', 'Output-Stationary', 8307, 8, 5, 2, 8, 10], ['cols', 'A-Stationary', 266, 29, 322, 8, 2, 10], ['rows', 'Output-Stationary', 271, 60, 257, 1, 9, 10], ['rows', 'B-Stationary', 225, 365, 38, 2, 8, 10], ['cols', 'B-Stationary', 81, 1, 9, 2, 8, 10], ['cols', 'A-Stationary', 1, 112, 432, 6, 4, 10], ['cols', 'Output-Stationary', 1, 432, 112, 7, 3, 10], ['cols', 'A-Stationary', 81, 125, 432, 7, 3, 10], ['cols', 'A-Stationary', 81, 128, 128, 9, 1, 10], ['rows', 'A-Stationary', 81, 81, 8, 9, 1, 10], ['cols', 'A-Stationary', 71, 473, 128, 3, 7, 10], ['cols', 'A-Stationary', 3822, 11, 15, 8, 2, 10], ['cols', 'B-Stationary', 81, 32, 81, 2, 8, 10], ['cols', 'B-Stationary', 79, 126, 448, 7, 3, 10], ['cols', 'Output-Stationary', 62, 588, 96, 7, 3, 10], ['cols', 'Output-Stationary', 1, 183, 539, 1, 9, 10], ['cols', 'Output-Stationary', 1, 753, 138, 5, 5, 10], ['cols', 'A-Stationary', 17, 183, 491, 6, 4, 10], ['cols', 'B-Stationary', 81, 184, 184, 7, 3, 10], ['cols', 'B-Stationary', 28, 559, 148, 8, 2, 10], ['rows', 'Output-Stationary', 4337, 18, 7, 8, 2, 10], ['cols', 'B-Stationary', 58, 139, 493, 7, 3, 10], ['cols', 'B-Stationary', 50, 519, 141, 7, 3, 10], ['rows', 'B-Stationary', 25, 1, 9, 3, 7, 10], ['cols', 'Output-Stationary', 1, 154, 675, 8, 2, 10], ['cols', 'Output-Stationary', 1, 623, 167, 6, 4, 10], ['cols', 'B-Stationary', 23, 118, 730, 4, 6, 10], ['cols', 'A-Stationary', 25, 224, 224, 5, 5, 10], ['rows', 'B-Stationary', 25, 25, 8, 8, 2, 10], ['cols', 'A-Stationary', 22, 698, 118, 3, 7, 10], ['rows', 'Output-Stationary', 625, 28, 28, 2, 8, 10], ['rows', 'B-Stationary', 25, 32, 25, 9, 1, 10], ['rows', 'A-Stationary', 13, 175, 559, 1, 9, 10], ['rows', 'Output-Stationary', 16, 697, 135, 3, 7, 10], ['cols', 'B-Stationary', 1, 396, 247, 4, 6, 10]]
	supernet_enc_dec_params = [['rows', 'Output-Stationary', 4821, 11, 11, 1, 9, 10], ['rows', 'A-Stationary', 4665, 1, 21, 8, 2, 10], ['rows', 'Output-Stationary', 3874, 14, 14, 1, 9, 10], ['rows', 'Output-Stationary', 901, 105, 6, 1, 9, 10], ['rows', 'Output-Stationary', 4478, 1, 21, 9, 1, 10], ['cols', 'Output-Stationary', 984, 30, 76, 8, 2, 10], ['rows', 'Output-Stationary', 669, 113, 30, 5, 5, 10], ['cols', 'B-Stationary', 811, 21, 108, 3, 7, 10], ['rows', 'B-Stationary', 1296, 1, 9, 4, 6, 10], ['cols', 'B-Stationary', 1, 48, 192, 8, 2, 10], ['cols', 'A-Stationary', 1, 192, 48, 9, 1, 10], ['rows', 'B-Stationary', 449, 21, 192, 9, 1, 10], ['cols', 'Output-Stationary', 455, 165, 40, 2, 8, 10], ['cols', 'A-Stationary', 1, 64, 240, 8, 2, 10], ['cols', 'B-Stationary', 1, 240, 64, 8, 2, 10], ['cols', 'B-Stationary', 527, 17, 164, 3, 7, 10], ['rows', 'B-Stationary', 324, 1, 9, 2, 8, 10], ['cols', 'B-Stationary', 291, 58, 239, 8, 2, 10], ['rows', 'B-Stationary', 324, 72, 72, 8, 2, 10], ['rows', 'Output-Stationary', 2592, 4, 9, 1, 9, 10], ['rows', 'Output-Stationary', 2592, 9, 4, 8, 2, 10], ['cols', 'Output-Stationary', 75, 1327, 6, 8, 2, 10], ['cols', 'B-Stationary', 288, 273, 40, 7, 3, 10], ['rows', 'Output-Stationary', 9795, 9, 1, 4, 6, 10], ['cols', 'A-Stationary', 296, 17, 305, 9, 1, 10], ['cols', 'B-Stationary', 269, 72, 239, 2, 8, 10], ['cols', 'Output-Stationary', 259, 313, 36, 4, 6, 10], ['cols', 'A-Stationary', 81, 1, 9, 1, 9, 10], ['cols', 'Output-Stationary', 1, 112, 432, 8, 2, 10], ['cols', 'Output-Stationary', 1, 432, 112, 8, 2, 10], ['cols', 'Output-Stationary', 81, 126, 430, 2, 8, 10], ['cols', 'A-Stationary', 81, 128, 128, 2, 8, 10], ['cols', 'A-Stationary', 648, 8, 16, 5, 5, 10], ['cols', 'B-Stationary', 648, 16, 8, 3, 7, 10], ['rows', 'Output-Stationary', 272, 357, 2, 2, 8, 10], ['cols', 'Output-Stationary', 69, 506, 117, 2, 8, 10], ['cols', 'A-Stationary', 4083, 8, 16, 4, 6, 10], ['cols', 'A-Stationary', 81, 32, 81, 9, 1, 10], ['cols', 'A-Stationary', 71, 119, 512, 5, 5, 10], ['rows', 'Output-Stationary', 58, 690, 90, 1, 9, 10], ['cols', 'A-Stationary', 1, 170, 593, 4, 6, 10], ['cols', 'A-Stationary', 1, 581, 172, 9, 1, 10], ['cols', 'B-Stationary', 33, 180, 454, 8, 2, 10], ['cols', 'A-Stationary', 81, 184, 184, 8, 2, 10], ['rows', 'Output-Stationary', 648, 11, 23, 4, 6, 10], ['cols', 'B-Stationary', 648, 23, 11, 1, 9, 10], ['cols', 'A-Stationary', 320, 299, 8, 9, 1, 10], ['cols', 'B-Stationary', 69, 380, 179, 6, 4, 10], ['cols', 'Output-Stationary', 6286, 8, 8, 9, 1, 10], ['rows', 'Output-Stationary', 64, 151, 443, 8, 2, 10], ['rows', 'Output-Stationary', 69, 710, 63, 2, 8, 10], ['rows', 'Output-Stationary', 25, 1, 9, 2, 8, 10], ['cols', 'B-Stationary', 1, 142, 727, 8, 2, 10], ['cols', 'B-Stationary', 1, 441, 221, 4, 6, 10], ['cols', 'B-Stationary', 25, 148, 590, 4, 6, 10], ['cols', 'A-Stationary', 25, 224, 224, 5, 5, 10], ['cols', 'A-Stationary', 200, 14, 28, 2, 8, 10], ['rows', 'B-Stationary', 200, 28, 14, 8, 2, 10], ['rows', 'Output-Stationary', 138, 677, 6, 5, 5, 10], ['cols', 'A-Stationary', 14, 454, 217, 8, 2, 10], ['rows', 'A-Stationary', 625, 28, 28, 9, 1, 10], ['cols', 'B-Stationary', 25, 32, 25, 2, 8, 10], ['cols', 'A-Stationary', 11, 153, 633, 4, 6, 10], ['cols', 'B-Stationary', 9, 1056, 89, 4, 6, 10], ['cols', 'B-Stationary', 1, 1119, 91, 9, 1, 10]]
	
	supernet_enc_dec_model = bm.create_nasvit_supernet(1, 0.5, 0.5, 0.9)
	supernet_enc_dec_layer_set = bm.model_to_layer_set(supernet_enc_dec_model)
	supernet_sparsity_model = bm.create_nasvit_supernet(1, 1.0, 1.0, 0.9)
	supernet_sparsity_layer_set = bm.model_to_layer_set(supernet_sparsity_model)
	
	# running each score layer
	hardware = bh.Hardware(num_PE_lanes=8, num_PEs_per_lane=64, num_RFs_per_PE=11, size_RF=10, off_chip_bandwidth=77, on_chip_bandwidth=100, total_sram_size=97946)
	
	score_layers = []
	for i in supernet_enc_dec_layer_set.layers:
		if i.flags["type"] == "attention" and i.flags["part"] == "score" and not i in score_layers:
			score_layers.append(i)
	
	score_layers_2 = []
	for i in supernet_sparsity_layer_set.layers:
		if i.flags["type"] == "attention" and i.flags["part"] == "score" and not i in score_layers:
			score_layers_2.append(i)
	
	enc_dec_results = []
	for idx, (layer, count) in enumerate(supernet_enc_dec_layer_set.unique_layers):
		if layer in score_layers:
			print(idx)
			#layer.print()
			#print(supernet_enc_dec_params[idx])
			res = dw.run_layer(hardware, supernet_enc_dec_params[idx], layer, 100000, pipeline_offloading=True)
			enc_dec_results.append(res)
	print("***"*20)
	sparsity_results = []
	for idx, (layer, count) in enumerate(supernet_sparsity_layer_set.unique_layers):
		if layer in score_layers_2: #layer.flags["type"] == "attention": # and layer.flags["part"] == "score":
			print(idx)
			#layer.print()
			#print(supernet_sparsity_params[idx])
			res = dw.run_layer(hardware, supernet_sparsity_params[idx], layer, 100000, pipeline_offloading=True)
			sparsity_results.append(res)
	
	# print results
	print("\n\n")
	i = 0
	for idx, (layer, count) in enumerate(supernet_enc_dec_layer_set.unique_layers):
		if layer in score_layers:
			layer.load_immediate = False
			print(idx)
			layer.print()
			print(supernet_enc_dec_params[idx])
			print(enc_dec_results[i])
			print((layer.get_flops_including_extras() / 512) / enc_dec_results[i][0])
			i += 1
	print("***"*20)
	i = 0
	for idx, (layer, count) in enumerate(supernet_sparsity_layer_set.unique_layers):
		if layer in score_layers_2: #layer.flags["type"] == "attention": # and layer.flags["part"] == "score":
			print(idx)
			layer.print()
			print(supernet_sparsity_params[idx])
			#res = dw.run_layer(hardware, supernet_sparsity_params[idx], layer, 100000, pipeline_offloading=True)
			#sparsity_results.append(res)
			print(sparsity_results[i])
			print((layer.get_flops_including_extras() / 512) / sparsity_results[i][0])
			i += 1

def test():
	hardware = bh.Hardware(num_PE_lanes=8, num_PEs_per_lane=64, num_RFs_per_PE=11, size_RF=10, off_chip_bandwidth=77, on_chip_bandwidth=100, total_sram_size=140*140*3)
	supernet_enc_dec_model = bm.create_nasvit_supernet(1, 0.5, 0.5, 0.9)
	supernet_enc_dec_layer_set = bm.model_to_layer_set(supernet_enc_dec_model)
	supernet_sparsity_model = bm.create_nasvit_supernet(1, 1.0, 1.0, 0.9)
	supernet_sparsity_layer_set = bm.model_to_layer_set(supernet_sparsity_model)
	
	layer = supernet_sparsity_layer_set.unique_layers[19][0]
	#params = ['rows', 'B-Stationary', 25, 25, 8, 1, 7, 1]
	params = ['cols', 'A-Stationary', 322, 289, 8, 3, 5, 2]
	
	res = dw.run_layer(hardware, params, layer, 100000, pipeline_offloading=True)
	print(res)
	
def gen_choices():
	hardware = bh.Hardware(num_PE_lanes=8, num_PEs_per_lane=64, num_RFs_per_PE=11, size_RF=10, off_chip_bandwidth=77, on_chip_bandwidth=100, total_sram_size=97946)
	supernet_enc_dec_model = bm.create_nasvit_supernet(1, 0.5, 0.5, 0.9)
	supernet_enc_dec_layer_set = bm.model_to_layer_set(supernet_enc_dec_model)
	supernet_sparsity_model = bm.create_nasvit_supernet(1, 1.0, 1.0, 0.9)
	supernet_sparsity_layer_set = bm.model_to_layer_set(supernet_sparsity_model)
	params = ['cols', 'A-Stationary', 322, 289, 8, 3, 5, 7]
	layer = supernet_sparsity_layer_set.unique_layers[19][0]
	
	t_a = 322 #math.ceil(25 / 8)
	t_b = math.ceil(289 / 8)
	t_w = 8
	
	print(t_a, t_b, t_w)
	
	# score each choice based on the closest to the multiple of # of PEs ... (not quite right way to say it)
	
	best_score = 0
	best_vals = None
	
	possibilities = []
	
	for c_a in range(1, hardware.num_RFs_per_PE):
		for c_b in range(1, hardware.num_RFs_per_PE + 1 - c_a):
			for c_w in range(1, hardware.size_RF):
				num_chunks = math.ceil(t_a / c_a) * math.ceil(t_b / c_b) * math.ceil(t_w / c_w)
				num_off = num_chunks % hardware.num_PEs_per_lane
				if num_off == 0:
					num_off = hardware.num_PEs_per_lane
					possibilities.append([c_a, c_b, c_w])
					#print(c_a, c_b, c_w)
				#if num_off == best_score:
				#	if c_a * c_b * c_w < best_vals[0] * best_vals[1] * best_vals[2]:
				#		best_score = num_off
				#		best_vals = [c_a, c_b, c_w]
				#elif num_off > best_score:
				#	best_score = num_off
				#	best_vals = [c_a, c_b, c_w]
				#chunk_choices.append([c_a, c_b, c_w, num_chunks, num_off])
	
	#print(chunk_choices)
	#print(best_score)
	#print(best_vals)
	
	final_res = []
	for c_a, c_b, c_w in possibilities:
		params[5] = c_a
		params[6] = c_b
		params[7] = c_w
		res = dw.run_layer(hardware, params, layer, 100000, pipeline_offloading=True)
		final_res.append(res)
	
	print(possibilities)
	print(final_res)

def test_2():
	hardware = bh.Hardware(num_PE_lanes=1, num_PEs_per_lane=64, num_RFs_per_PE=20, size_RF=10, off_chip_bandwidth=77, on_chip_bandwidth=100, total_sram_size=97946)
	layer = bm.Layer(50, 50, 50, sparsity=0.0)
	params = ['rows', 'Output-Stationary', 134, 134, 134, 5, 5, 10]
	
	res = dw.run_layer(hardware, params, layer, preload_cycles=1000, pipeline_offloading=True)
	print(res)
	
	num_tiles = math.ceil(layer.A_rows / params[2]) * math.ceil(layer.A_cols_B_rows / params[3]) * math.ceil(layer.B_cols / params[4])
	num_chunks = math.ceil(params[2] / params[5]) * math.ceil(params[3] / params[6]) * math.ceil(params[4] / params[7])
	chunk_compute_time = params[5] * params[6] * params[7]
	chunk_startup_cost = 1
	tile_startup_cost = 0
	layer_startup_cost = 0
	predicted_time_per_tile = math.ceil(num_chunks / hardware.num_PEs_per_lane) * (chunk_startup_cost + chunk_compute_time)
	
	total_cycles = 0
	
	total_cycles += layer_startup_cost
	
	# for each tile
	for t_a in range(0, math.ceil(layer.A_rows / params[2])):
		a_length = params[2] if t_a != math.ceil(layer.A_rows / params[2]) - 1 else layer.A_rows - t_a * params[2]
		for t_b in range(0, math.ceil(layer.B_cols / params[3])):
			b_length = params[3] if t_b != math.ceil(layer.B_cols / params[3]) - 1 else layer.B_cols - t_b * params[3]
			for t_w in range(0, math.ceil(layer.A_cols_B_rows / params[4])):
				w_length = params[4] if t_w != math.ceil(layer.A_cols_B_rows / params[4]) - 1 else layer.A_cols_B_rows - t_w * params[4]
				total_cycles += tile_startup_cost
				
				# build chunks
				chunks = []
				for c_a in range(0, math.ceil(a_length / params[5])):
					ca_length = params[5] if c_a != math.ceil(a_length / params[5]) - 1 else a_length - c_a * params[5]
					for c_b in range(0, math.ceil(b_length / params[6])):
						cb_length = params[6] if c_b != math.ceil(b_length / params[6]) - 1 else b_length - c_b * params[6]
						for c_w in range(0, math.ceil(w_length / params[7])):
							cw_length = params[7] if c_w != math.ceil(w_length / params[7]) - 1 else w_length - c_w * params[7]
							chunks.append(ca_length * cb_length * cw_length * (1 - layer.sparsity))
				#print(len(chunks))
				# assign chunks to PEs
				#print("Starting new tile")
				while len(chunks) > 0:
					max_size = 0
					# each PE takes a chunk
					for i in range(hardware.num_PEs_per_lane):
						if len(chunks) == 0:
							break
						size = chunks[0]
						del chunks[0]
						if size > max_size:
							max_size = size
					#print(max_size)
					total_cycles += chunk_startup_cost + max_size
				
				#total_cycles += chunk_startup_cost
				#total_cycles += ca_length * cb_length * cw_length
							
	
	print("Predicted compute-bound time:", total_cycles) #num_tiles * predicted_time_per_tile)

def test_3():
	supernet_model = bm.create_nasvit_supernet(1, 1.0, 1.0, 0.0)
	supernet_layer_set = bm.model_to_layer_set(supernet_model)
	supernet_sparsity_model = bm.create_nasvit_supernet(1, 1.0, 1.0, 0.9)
	supernet_sparsity_layer_set = bm.model_to_layer_set(supernet_sparsity_model)
	
	flops_att_score = 0
	for layer in supernet_layer_set.layers:
		if layer.flags["type"] == "attention" and layer.flags["part"] == "score":
			flops_att_score += layer.get_flops_including_extras()
	print(flops_att_score)
	print(supernet_layer_set.get_total_flops_including_extras())
	print(flops_att_score / supernet_layer_set.get_total_flops_including_extras())
	
	flops_att_score_sparse = 0
	for layer in supernet_sparsity_layer_set.layers:
		if layer.flags["type"] == "attention" and layer.flags["part"] == "score":
			flops_att_score_sparse += layer.get_flops_including_extras()
	print(flops_att_score_sparse)
	print(supernet_layer_set.get_total_flops_including_extras())
	print(flops_att_score_sparse / supernet_layer_set.get_total_flops_including_extras())

def display_res():
	flops = [1163.618304, 942.4614399999999, 856.9099520000001, 283.69464400000004, 895.6357100000001, 543.350656, 608.5595099999999, 584.70399, 468.75897599999996, 796.897152, 436.20190600000006, 663.8513820000001, 477.547192, 1231.6000219999999, 518.3666760000001, 378.831056, 677.626534, 422.548248, 862.4221439999999, 450.50328]
	acc = [81.08399963378906, 78.41999816894531, 81.25999450683594, 75.78599548339844, 77.51200103759766, 77.6240005493164, 80.25599670410156, 75.88800048828125, 78.69200134277344, 80.06999969482422, 79.11799621582031, 78.30799865722656, 78.0999984741211, 79.10599517822266, 78.83399963378906, 76.03199768066406, 79.48400115966797, 78.8219985961914, 80.66199493408203, 78.71599578857422]
	
	plt.plot(flops, acc, '.')
	plt.show()

display_res()
#test_3()
#test()
#gen_choices()
#examine_nasvit_results()