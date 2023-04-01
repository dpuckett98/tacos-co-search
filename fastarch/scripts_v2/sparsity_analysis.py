import build_models_v2 as models
import matplotlib.pyplot as plt
import numpy as np

# bandwidth is in GB/s; clockspeed is in MHz
def bottlenecks(layer, num_PEs, bandwidth, clockspeed, utilization=1.0):
	elemsPerCycle = bandwidth * 1000000000 / 2 / (clockspeed * 1000000)
	
	# compute
	compute_cycles = layer.get_flops_including_extras() / (num_PEs * utilization)
	# load
	load_cycles = (layer.A_rows * layer.A_cols_B_rows + layer.A_cols_B_rows * layer.B_cols) / elemsPerCycle
	# store
	store_cycles = layer.A_rows * layer.B_cols / elemsPerCycle
	
	print((layer.A_rows * layer.A_cols_B_rows + layer.A_cols_B_rows * layer.B_cols + layer.A_rows * layer.B_cols) * 2)
	
	#print("Compute cycles:", compute_cycles)
	#print("Load cycles:", load_cycles)
	#print("Store cycles:", store_cycles)
	
	#return max(compute_cycles, load_cycles, store_cycles)
	return [compute_cycles, load_cycles, store_cycles]

def show_bottleneck_by_sparsity(num_PEs, bandwidth, clockspeed):
	
	sparsity_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	util_list = [1.0, 0.8, 0.6, 0.4, 0.2]
	compute_cycles = [[] for i in util_list]
	load_cycles = [[] for i in util_list]
	store_cycles = [[] for i in util_list]

	for sparsity in sparsity_list:
		model = models.get_DeiT_Base(1, 1.0, sparsity)
		layers = models.model_to_layer_set(model)
		
		att_score = layers.unique_layers[3][0]
		#print(layers.unique_layers[3][0].A_rows, layers.unique_layers[3][0].A_cols_B_rows, layers.unique_layers[3][0].B_cols)
		
		for idx, util in enumerate(util_list):
			c, l, s = bottlenecks(att_score, num_PEs, bandwidth, clockspeed, utilization=util)
			compute_cycles[idx].append(c)
			load_cycles[idx].append(l+s)
			#store_cycles[idx].append(s)
		#c, l, s = bottlenecks(att_score, num_PEs, bandwidth, clockspeed)
		#c, l, s = bottlenecks(att_score, num_PEs, bandwidth, clockspeed, utilization = 0.8)
		#c, l, s = bottlenecks(att_score, num_PEs, bandwidth, clockspeed, utilization = 0.6)
		#c, l, s = bottlenecks(att_score, num_PEs, bandwidth, clockspeed)
		#c, l, s = bottlenecks(att_score, num_PEs, bandwidth, clockspeed)
		
		#compute_cycles.append(c)
		#load_cycles.append(l+s)
		#store_cycles.append(s)
	
	for idx, util in enumerate(util_list):
		plt.plot(sparsity_list, compute_cycles[idx], '.-', label='Compute Cycles, ' + str(util * 100) + '% Utilization')
	plt.plot(sparsity_list, load_cycles[0], '.-', label='Load/Store Cycles')
	#plt.plot(sparsity_list, store_cycles, '.-', label='Store Cycles')
	plt.legend()
	plt.xlabel("Sparsity")
	plt.ylabel("Cycles")
	plt.title("DeiT Base, Cycles vs Sparsity on Ideal Accelerator for Q*K^T")
	plt.show()

def graph_speedups():
	sparsity = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	act_cycles = [2817504, 2817504, 2789100, 2761128, 2732940, 2606328, 2575548, 2547144, 2518848, 2491920]
	layers = []
	flops = []
	flops_plain = []
	
	for i in sparsity:
		model = models.get_DeiT_Tiny(1, 1.0, i)
		layers.append(models.model_to_layer_set(model))
		flops.append(layers[-1].get_total_flops_including_extras())
		flops_plain.append(layers[-1].get_total_flops())
	
	ideal_cycles = [i / 512 for i in flops]
	ideal_cycles_plain = [i / 512 for i in flops_plain]
	
	speedup = [ideal_cycles[i] / act_cycles[i] for i in range(len(sparsity))]
	
	#plt.bar(sparsity, speedup, width=0.05)
	
	plt.plot(sparsity, ideal_cycles, '-', label='Ideal')
	plt.plot(sparsity, act_cycles, '-', label='Actual with sparsity and enc/dec')
	plt.plot(sparsity, ideal_cycles_plain, '-', label='Actual without sparsity and enc/dec')
	plt.xlabel("Sparsity")
	plt.ylabel("Ideal / Actual Cycles")
	plt.title("DeiT Tiny on TACoS Accelerator")
	plt.legend()
	plt.show()
	
	#for ls in layers:
	#	print(ls.get_total_flops_including_extras())

def graph_cycles_by_sparsity():
	
	# w/ 0 global tokens, 76.8 GB/s, and 512 PEs
	#cycles_layer_3 = [6261, 5801, 5303, 4845, 4337, 3849, 3395, 2909, 2467, 2011]
	#pe = [6158, 5698, 5200, 4742, 4234, 3746, 3292, 2806, 2364, 1908]
	#mem = [1763, 1763, 1763, 1763, 1763, 1763, 1763, 1763, 1763, 1763]
	
	# w/ 0 global tokens, 40 GB/s, and 512 PEs
	#cycles_layer_3 = [7421, 6940, 6458, 6035, 5535, 5061, 4621, 4144, 3761, 3400]
	#pe = [7233, 6752, 6270, 5847, 5347, 4873, 4433, 3956, 3573, 3210]
	#mem = [3351, 3351, 3351, 3351, 3351, 3351, 3351, 3351, 3351, 3351]
	
	# w/ 0 global tokens, 10 GB/s, and 512 PEs
	cycles_layer_3 = [15328, 14996, 14596, 14258, 13899, 13572, 13547, 13507, 13491, 13458]
	pe = [14603, 14271, 13871, 13533, 13174, 12847, 12818, 12778, 12762, 12729]
	mem = [13413, 13413, 13413, 13413, 13413, 13413, 13413, 13413, 13413, 13413]

	# w/ 0 global tokens, 76.8 GB/s, and 1024 PEs
	#cycles_layer_3 = [3741, 3497, 3233, 3001, 2729, 2466, 2215, 1972, 1855, 1814]
	#pe = [3638, 3394, 3130, 2898, 2626, 2363, 2112, 1869, 1748, 1707]
	#mem = [1763, 1763, 1763, 1763, 1763, 1763, 1763, 1763, 1763, 1763]
	
	# w/ 0 global tokens, 40 GB/s, and 1024 PEs
	#[4734, 4481, 4229, 3987, 3754, 3550, 3500, 3460, 3440, 3400]
	#[4546, 4293, 4041, 3799, 3566, 3362, 3310, 3270, 3248, 3208]
	#[3351, 3351, 3351, 3351, 3351, 3351, 3351, 3351, 3351, 3351]
	
	# w/ 0 global tokens, 76.8 GB/s, and 512 PEs w/ double-buffering and 1563 cycles of preloading
	cycles_layer_3 = [5219, 4742, 4234, 3726, 3218, 2710, 2236, 1807, 1567, 1326]
	pe = [5121, 4644, 4136, 3628, 3120, 2611, 2134, 1626, 1166, 763]
	mem = [1029, 1029, 1029, 1029, 1029, 1029, 1030, 1029, 1030, 1030]
	
	# w/ 0 global tokens, 76.8 GB/s, and 512 PEs w/ double-buffering and 1563 cycles of preloading, and pipelining offloading outputs
	cycles_layer_3 = [5129, 4652, 4144, 3636, 3128, 2620, 2142, 1634, 1174, 780]
	pe = [5121, 4644, 4136, 3628, 3120, 2611, 2134, 1626, 1166, 763]
	mem = [1029, 1029, 1029, 1029, 1029, 1029, 1030, 1029, 1030, 1030]
	
	# w/ 0 global tokens, 76.8 GB/s, and 512 PEs w/ double-buffering and 1563 cycles of preloading, and pipelining offloading outputs and spending 70% of bandwidth on loading inputs
	cycles_layer_3 = [5129, 4652, 4144, 3636, 3128, 2620, 2142, 1634, 1126, 694]
	pe = [5121, 4644, 4136, 3628, 3120, 2611, 2134, 1626, 1118, 682]
	mem = [1704, 1704, 1704, 1704, 1704, 1704, 1704, 1704, 1704, 1704]
	
	# w/ 0 global tokens, 76.8 GB/s, and 512 PEs w/ double-buffering and 1563 cycles of preloading, and pipelining offloading outputs and fully utilizing DRAM bandwidth
	cycles_layer_3 = [5129, 4652, 4144, 3636, 3128, 2620, 2142, 1634, 1126, 630]
	pe = [5121, 4644, 4136, 3628, 3120, 2611, 2134, 1626, 1118, 622]
	mem = [727, 727, 727, 727, 727, 727, 727, 727, 727, 729]
	
	# w/ 0 global tokens, 5 GB/s, and 512 PEs w/ double-buffering and 1563 cycles of preloading, and pipelining offloading outputs and fully utilizing DRAM bandwidth
	cycles_layer_3 = [7792, 7440, 7071, 6752, 6468, 6225, 6082, 6072, 6062, 6052]
	pe = [7505, 7053, 6584, 6165, 5781, 5438, 5195, 5083, 4925, 4175]
	mem = [13879, 13879, 13879, 13879, 13879, 13879, 13879, 13879, 13879, 13879]
	
	# w/ 0 global tokens, 5 GB/s, and 256 PEs w/ double-buffering and 1563 cycles of preloading, and pipelining offloading outputs and fully utilizing DRAM bandwidth
	cycles_layer_3 = [11417, 10761, 10057, 9337, 8626, 7909, 7261, 6641, 6141, 6052]
	pe = [11102, 10260, 9378, 8518, 7667, 6810, 6022, 5260, 4602, 3870]
	mem = [13879, 13879, 13879, 13879, 13879, 13879, 13879, 13879, 13879, 13879]
	
	sparsity = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	
	#z = np.polyfit(sparsity, cycles_layer_3, 1)
	#p = np.poly1d(z)
	
	plt.plot(sparsity, cycles_layer_3, '.-', label="Total cycles")
	plt.plot(sparsity, pe, '.-', label="Compute cycles")
	plt.plot(sparsity, mem, '.-', label="Load/Store cycles")
	#plt.plot(sparsity, p(sparsity), '.-')
	#plt.plot(sparsity, cycles_layer_4, '.-', label="Attention Value")
	plt.xlabel("Sparsity")
	plt.ylabel("Cycles")
	plt.title("DeiT Base, Cycles vs Sparsity on TACoS Accelerator for Q*K^T")
	plt.legend()
	plt.show()

def show_enc_dec_space():
	num_PEs = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512]
	total_cycles = [9661, 4937, 3696, 3097, 2760, 2558, 2435, 2343, 2270, 2255, 2251, 2251, 2251, 2251, 2251, 2251]
	compute_cycles = [9537, 4786, 3473, 2875, 2567, 2352, 2219, 2117, 2141, 2127, 2123, 2123, 2113, 2180, 2180, 2175]
	mem_cycles = [6156, 6158, 6158, 6158, 6158, 6158, 6158, 6158, 6158, 6158, 6158, 6158, 6158, 6158, 6158, 6158]
	#num_PEs = [4, 8, 16, 32, 64, 128, 256, 512]
	#total_cycles = [77231, 38616, 19314, 9661, 4937, 3097, 2343, 2251]
	#compute_cycles = [75238, 38124, 19068, 9537, 4786, 2875, 2117, 2175]
	#mem_cycles = [6156, 6156, 6156, 6156, 6158, 6158, 6158, 6158]
	
	base_cycles = [6052 for i in num_PEs]
	
	plt.plot(num_PEs, total_cycles, '.-', label="Cycles with 10 GB/s and various PEs")
	plt.plot(num_PEs, base_cycles, '.-', label="Cycles with 5 GB/s and 512 PEs")
	plt.yscale("log")
	plt.xlabel("Number of PEs")
	plt.ylabel("Cycles")
	plt.title("DeiT Base, Tradeoff Space for Encoder/Decoder on TACoS Accelerator for Q*K^T")
	plt.legend()
	plt.show()

def build_roofline_model():
	# GOPS = GFLOPs / second
	# Comp. Intensity = Ops/Byte
	
	sparsity = np.linspace(0, 1, 11)
	
	comp_intensity = []
	
	for s in sparsity:
		layer = models.Layer(A_rows=198, A_cols_B_rows=64, B_cols=198, sparsity=s)
		print(s, layer.get_flops_including_extras())
	

# varying number of PEs
	#[6751, 6751, 6751, 6751, 6751, 6751, 6751]
	#[76048.0, 76048.0, 76048.0, 76048.0, 76048.0, 76048.0, 76048.0]
	#[6380, 6380, 6380, 6380, 6380, 6380, 6380]
	#[6705, 6705, 6705, 6705, 6705, 6705, 6705]

#graph_speedups()
#graph_cycles_by_sparsity()
#show_enc_dec_space()
build_roofline_model()

#show_bottleneck_by_sparsity(512, 76.8, 500)

#model = models.get_DeiT_Base(1, 1.0, 0.9)
#model.print_out_sizes()
#layers = models.model_to_layer_set(model)
#print("FLOPs:", layers.get_total_flops())

#model = models.get_DeiT_Base(1, 1.0, 0.9)
#layers = models.model_to_layer_set(model)
#layers.print()

#print(layers.get_total_flops() / 512)
#print(layers.get_total_flops_including_extras() / 512)

#att_score = layers.unique_layers[3][0]
#qk_v = layers.unique_layers[4][0]

#bottlenecks(att_score, 512, 100)
#bottlenecks(qk_v, 512, 100)