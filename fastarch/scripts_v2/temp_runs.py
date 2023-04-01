from dataflow_enc_dec import run_MM_dataflow
import build_models_v2 as models
import build_hardware as hardware

# taken from run_3

def run_LeViT_128_layer_1():
	cycles_1, dram_1 = run_MM_dataflow(512, 6, 17, 100, 10, 107776, 12544, 32, 27, "B-Stationary", 2574, 32, 9, 4, 2, 9, estimate=False)

	# results after fixing bandwidth
	# cycles: 32347
	# dram: 3349952
	# pe util (overall): 65.4%
	# bandwidth util (overall): 52.6%

def run_LeViT_128_layer_2():
	num_PEs = 512
	num_RFs_per_PE = 6
	size_RF = 17
	off_chip_bandwidth = 100
	on_chip_bandwidth = 10
	max_sram_size = 107776
	A_rows = 3136 #layer.A_rows
	A_cols_B_rows = 288 #layer.A_cols_B_rows
	B_cols = 64 #layer.B_cols
	dataflow = "Output-Stationary"
	t_a = 526
	t_b = 64
	t_w = 12
	c_a = 1
	c_b = 5
	c_w = 12
	print(A_rows, t_a, A_cols_B_rows, t_w, B_cols, t_b)
	cycles, dram_accesses = run_MM_dataflow(num_PEs, num_RFs_per_PE, size_RF, off_chip_bandwidth, on_chip_bandwidth, max_sram_size, A_rows, A_cols_B_rows, B_cols, dataflow, t_a, t_b, t_w, c_a, c_b, c_w, estimate=False)

	# results after fixing bandwidth
	# cycles: 128568
	# dram: 1226388
	# pe util (overall): 87.8%
	# bandwidth util (overall): 7.9%

def run_LeViT_128_layer_3():
	num_PEs = 512
	num_RFs_per_PE = 6
	size_RF = 17
	off_chip_bandwidth = 100
	on_chip_bandwidth = 10
	max_sram_size = 107776
	A_rows = 784 #layer.A_rows
	A_cols_B_rows = 576 #layer.A_cols_B_rows
	B_cols = 128 #layer.B_cols
	dataflow = "Output-Stationary"
	t_a = 784
	t_b = 32
	t_w = 64
	c_a = 2
	c_b = 4
	c_w = 17
	print(A_rows, t_a, A_cols_B_rows, t_w, B_cols, t_b)
	cycles, dram_accesses = run_MM_dataflow(num_PEs, num_RFs_per_PE, size_RF, off_chip_bandwidth, on_chip_bandwidth, max_sram_size, A_rows, A_cols_B_rows, B_cols, dataflow, t_a, t_b, t_w, c_a, c_b, c_w, estimate=False)

	# results after fixing bandwidth
	# cycles: 127635
	# dram: 2097920
	# pe util (overall): 88.5%
	# bandwidth util (overall): 15.7%

def run_DeiT_Tiny_layer_1():
	num_PEs = 512
	num_RFs_per_PE = 11
	size_RF = 11
	off_chip_bandwidth = 100
	on_chip_bandwidth = 10
	max_sram_size = 98048
	A_rows = 196
	A_cols_B_rows = 384
	B_cols = 192
	params = ['A-Stationary', 196, 96, 106, 7, 4, 11]
	#cycles, dram_accesses = run_MM_dataflow(num_PEs, num_RFs_per_PE, size_RF, off_chip_bandwidth, on_chip_bandwidth, max_sram_size, A_rows, A_cols_B_rows, B_cols, params[0], params[1], params[2], params[3], params[4], params[5], params[6], estimate=False)
	#return [cycles, dram_accesses]
	return [31628, 412416]

# taken from run_2

def run_DeiT_Tiny_layer_1():
	num_PEs = 512
	num_RFs_per_PE = 11
	size_RF = 11
	off_chip_bandwidth = 100
	on_chip_bandwidth = 10
	max_sram_size = 98048
	A_rows = 196
	A_cols_B_rows = 384
	B_cols = 192
	params = ['A-Stationary', 196, 96, 106, 7, 4, 11]
	#cycles, dram_accesses = run_MM_dataflow(num_PEs, num_RFs_per_PE, size_RF, off_chip_bandwidth, on_chip_bandwidth, max_sram_size, A_rows, A_cols_B_rows, B_cols, params[0], params[1], params[2], params[3], params[4], params[5], params[6], estimate=False)
	#return [cycles, dram_accesses]
	return [31628, 412416]

def run_DeiT_Tiny_layer_2():
	num_PEs = 512
	num_RFs_per_PE = 11
	size_RF = 11
	off_chip_bandwidth = 100
	on_chip_bandwidth = 10
	max_sram_size = 98048
	A_rows = 196
	A_cols_B_rows = 192
	B_cols = 192
	params = ['B-Stationary', 107, 96, 97, 3, 8, 11]
	#cycles, dram_accesses = run_MM_dataflow(num_PEs, num_RFs_per_PE, size_RF, off_chip_bandwidth, on_chip_bandwidth, max_sram_size, A_rows, A_cols_B_rows, B_cols, params[0], params[1], params[2], params[3], params[4], params[5], params[6], estimate=False)
	#return [cycles, dram_accesses]
	return [16504, 225024]

# this is computing the attention scores
def run_DeiT_Tiny_layer_3(sparsity=0.0, off_chip_bandwidth=100, num_global_tokens=0, num_PEs=512):
	#num_PEs = 512
	num_RFs_per_PE = 11
	size_RF = 11
	#off_chip_bandwidth = 200
	#on_chip_bandwidth = 10
	max_sram_size = 98048
	A_rows = 198
	A_cols_B_rows = 64
	B_cols = 198
	
	#num_PEs = 1024
	num_RFs_per_PE = 11
	size_RF = 10
	#off_chip_bandwidth = 100
	on_chip_bandwidth = 10
	max_sram_size = 57600
	
	params = ['Output-Stationary', 140, 140, 140, 10, 1, 10] #['B-Stationary', 98, 100, 64, 2, 9, 11]
	
	cycles = 0
	dram_accesses = 0
	
	if num_global_tokens != 0:
		c, d = run_MM_dataflow(num_PEs, num_RFs_per_PE, size_RF, off_chip_bandwidth, on_chip_bandwidth, max_sram_size, num_global_tokens, A_cols_B_rows, B_cols, params[0], params[1], params[2], params[3], params[4], params[5], params[6], estimate=False, sparsity=0.0)
		cycles += c
		dram_accesses += d
	
	c, d, pe, mem, a, b = run_MM_dataflow(num_PEs, num_RFs_per_PE, size_RF, off_chip_bandwidth, on_chip_bandwidth, max_sram_size, A_rows - num_global_tokens, A_cols_B_rows, B_cols, params[0], params[1], params[2], params[3], params[4], params[5], params[6], estimate=False, sparsity=sparsity, preload_cycles=1563, pipeline_offloading=True)
	cycles += c
	dram_accesses += d
	return [cycles, dram_accesses, pe, mem, a, b]
	#return [6345, 138768] # no sparsity results

# this is computing attention score*V
def run_DeiT_Tiny_layer_4(sparsity=0.0, off_chip_bandwidth=100):
	num_PEs = 512
	num_RFs_per_PE = 11
	size_RF = 11
	#off_chip_bandwidth = 100
	#on_chip_bandwidth = 10
	max_sram_size = 98048
	A_rows = 196
	A_cols_B_rows = 64
	B_cols = 196
	
	num_PEs = 512
	num_RFs_per_PE = 11
	size_RF = 10
	#off_chip_bandwidth = 100
	on_chip_bandwidth = 10
	max_sram_size = 57600
	
	params = ['Output-Stationary', 140, 140, 140, 10, 1, 10] #['Output-Stationary', 101, 64, 52, 9, 2, 11]
	cycles, dram_accesses = run_MM_dataflow(num_PEs, num_RFs_per_PE, size_RF, off_chip_bandwidth, on_chip_bandwidth, max_sram_size, A_rows, A_cols_B_rows, B_cols, params[0], params[1], params[2], params[3], params[4], params[5], params[6], estimate=False, sparsity=sparsity, orig_heads=3, comp_heads=1, decode=False)
	return [cycles, dram_accesses]
	#return [7732, 113680] # no sparsity results

def run_DeiT_Tiny_layer_5():
	num_PEs = 512
	num_RFs_per_PE = 11
	size_RF = 11
	off_chip_bandwidth = 100
	on_chip_bandwidth = 10
	max_sram_size = 98048
	A_rows = 196
	A_cols_B_rows = 768
	B_cols = 192
	params = ['Output-Stationary', 196, 265, 32, 6, 5, 11]
	#cycles, dram_accesses = run_MM_dataflow(num_PEs, num_RFs_per_PE, size_RF, off_chip_bandwidth, on_chip_bandwidth, max_sram_size, A_rows, A_cols_B_rows, B_cols, params[0], params[1], params[2], params[3], params[4], params[5], params[6], estimate=False)
	#return [cycles, dram_accesses]
	return [64403, 335616]

def run_DeiT_Tiny_layer_6():
	num_PEs = 512
	num_RFs_per_PE = 11
	size_RF = 11
	off_chip_bandwidth = 100
	on_chip_bandwidth = 10
	max_sram_size = 98048
	A_rows = 196
	A_cols_B_rows = 192
	B_cols = 768
	params = ['B-Stationary', 103, 97, 387, 4, 7, 11]
	#cycles, dram_accesses = run_MM_dataflow(num_PEs, num_RFs_per_PE, size_RF, off_chip_bandwidth, on_chip_bandwidth, max_sram_size, A_rows, A_cols_B_rows, B_cols, params[0], params[1], params[2], params[3], params[4], params[5], params[6], estimate=False)
	#return [cycles, dram_accesses]
	return [63522, 599040]

def time_DeiT_Tiny(sparsity=0.0):
	deit = models.get_DeiT_Tiny(1, 1.0, 0.0)
	layers = models.model_to_layer_set(deit)
	layers.print()
	
	c, a = run_DeiT_Tiny_layer_1()
	layers.update_layer_latency(layers.unique_layers[0][0], c)
	layers.update_layer_dram_accesses(layers.unique_layers[0][0], a)
	
	c, a = run_DeiT_Tiny_layer_2()
	layers.update_layer_latency(layers.unique_layers[1][0], c)
	layers.update_layer_dram_accesses(layers.unique_layers[1][0], a)
	
	c, a = run_DeiT_Tiny_layer_3(sparsity)
	layers.update_layer_latency(layers.unique_layers[2][0], c)
	layers.update_layer_dram_accesses(layers.unique_layers[2][0], a)
	
	c, a = run_DeiT_Tiny_layer_4(sparsity)
	layers.update_layer_latency(layers.unique_layers[3][0], c)
	layers.update_layer_dram_accesses(layers.unique_layers[3][0], a)
	
	c, a = run_DeiT_Tiny_layer_5()
	layers.update_layer_latency(layers.unique_layers[4][0], c)
	layers.update_layer_dram_accesses(layers.unique_layers[4][0], a)
	
	c, a = run_DeiT_Tiny_layer_6()
	layers.update_layer_latency(layers.unique_layers[5][0], c)
	layers.update_layer_dram_accesses(layers.unique_layers[5][0], a)
	
	layers.print_stats(512)

# this is the attention score
def run_DeiT_Base_layer_4(sparsity=0.0):
	num_PEs = 512
	num_RFs_per_PE = 11
	size_RF = 13
	off_chip_bandwidth = 100
	on_chip_bandwidth = 10
	max_sram_size = 86784
	A_rows = 196
	A_cols_B_rows = 64
	B_cols = 196
	params = ['A-Stationary', 196, 38, 75, 10, 1, 13]
	cycles, dram_accesses = run_MM_dataflow(num_PEs, num_RFs_per_PE, size_RF, off_chip_bandwidth, on_chip_bandwidth, max_sram_size, A_rows, A_cols_B_rows, B_cols, params[0], params[1], params[2], params[3], params[4], params[5], params[6], estimate=False, sparsity=sparsity, orig_heads=3, comp_heads=1, decode=False)
	return [cycles, dram_accesses]
	#return [7732, 113680] # no sparsity results

def run_model_with_params(model, params):
	layers = models.model_to_layer_set(model)
	num_PEs = 512
	num_RFs_per_PE = 10
	size_RF = 10
	off_chip_bandwidth = 100
	on_chip_bandwidth = 10
	max_sram_size = 57600
	for layer, count in layers.unique_layers:
		cycles, dram_accesses = run_MM_dataflow(num_PEs, num_RFs_per_PE, size_RF, off_chip_bandwidth, on_chip_bandwidth, max_sram_size, layer.A_rows, layer.A_cols_B_rows, layer.B_cols, params[0], params[1], params[2], params[3], params[4], params[5], params[6], estimate=False, sparsity=layer.sparsity, orig_heads=3, comp_heads=1, decode=False)
		layers.update_layer_latency(layer, cycles)
		layers.update_layer_dram_accesses(layer, dram_accesses)
	return layers

def get_cycles_by_sparsity():
	#sparsity = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	sparsity = [0.9]
	#sparsity = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
	
	bandwidth = 5 #int(76.8)
	clockspeed = 500
	elemsPerCycle = bandwidth * 1000000000 / 2 / (clockspeed * 1000000)
	#print(elemsPerCycle)
	layer_3_cycles = []
	layer_3_pe_active_cycles = []
	layer_3_mem_active_cycles = []
	layer_4_cycles = []
	
	for s in sparsity:
		c, d, pe, mem, _, _ = run_DeiT_Tiny_layer_3(s, elemsPerCycle, num_PEs=512)
		layer_3_cycles.append(c)
		layer_3_pe_active_cycles.append(pe)
		layer_3_mem_active_cycles.append(mem)
		#c, d = run_DeiT_Tiny_layer_4(s, elemsPerCycle)
		#layer_4_cycles.append(c)
	
	print(layer_3_cycles)
	print(layer_3_pe_active_cycles)
	print(layer_3_mem_active_cycles)
	print(layer_4_cycles)

def get_cycles_2():
	sparsity = 0.9
	bandwidth = 10 # doubled bandwidth from enc/dec
	clockspeed = 500
	elemsPerCycle = bandwidth * 1000000000 / 2 / (clockspeed * 1000000)
	
	num_pes = list(range(32, 512+1, 32))
	#num_pes = [32, 64, 128, 256, 512] #, 420, 440, 460, 480, 500, 520]
	
	cl = []
	dl = []
	pel = []
	meml = []
	
	for p in num_pes:
		c, d, pe, mem, _, _ = run_DeiT_Tiny_layer_3(sparsity, elemsPerCycle, num_PEs=p)
		cl.append(c)
		dl.append(d)
		pel.append(pe)
		meml.append(mem)
	
	print(num_pes)
	print(cl)
	#print(dl)
	print(pel)
	print(meml)

#layers = run_model_with_params(models.get_DeiT_Base(1, 1.0, 0.0), ['Output-Stationary', 140, 140, 140, 5, 5, 10])
#layers.print_stats(512)

get_cycles_2()
#get_cycles_by_sparsity()

#print(run_LeViT_128_layer_1())
#c1, d1 = run_DeiT_Tiny_layer_3(0.9, 400)
#c2, d2 = run_DeiT_Tiny_layer_4(0.9, 400)
#print(c1 + c2)

# just Q*K and (Q*K)*V
# 90% sparsity, 1x bandwidth = 5033
# 90% sparsity, 2x bandwidth = 4489 (1.12x speedup) --> this is what TACoS would get with the benefits of the enc/dec and without its costs
# 90% sparsity, 4x bandwidth = 4230 (1.19x speedup)

#time_DeiT_Tiny(0.9)
#levit = models.get_LeViT_128(1, 1.0, 0.0)
#layers = models.model_to_layer_set(levit)
#layers.print()

#for layer, count in layers.unique_layers:
#	print((layer.A_rows % 22) / 22, (layer.B_cols % 22) / 22)
#	layer.print()

# Results:
# Sparsity	Cycles
# 0%		2817504
# 10%		2817504
# 20%		2789100
# 30%		2761128
# 40%		2732940
# 50%		2606328
# 60%		2575548
# 70%		2547144
# 80%		2518848
# 90%		2491920

# [2817504, 2817504, 2789100, 2761128, 2732940, 2606328, 2575548, 2547144, 2518848, 2491920]