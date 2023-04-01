from dataflow_enc_dec import run_MM_dataflow
import build_models_v2 as models
import build_hardware as hardware

def simple_test():
	num_PEs = 512
	num_RFs_per_PE = 11
	size_RF = 11
	max_sram_size = 98048
	off_chip_bandwidth = 77
	on_chip_bandwidth = 10
	max_sram_size = 57600
	
	A_rows = 300
	A_cols_B_rows = 300
	B_cols = 300
	
	params = ['A-Stationary', 100, 100, 100, 5, 5, 10]
	sparsity = 0.0
	
	c, d, pe, mem, a, b = run_MM_dataflow(num_PEs, num_RFs_per_PE, size_RF, off_chip_bandwidth, on_chip_bandwidth, max_sram_size, A_rows, A_cols_B_rows, B_cols, params[0], params[1], params[2], params[3], params[4], params[5], params[6], estimate=False, sparsity=sparsity, preload_cycles=1500, pipeline_offloading=True)
	
	print(c, d, pe, mem, a, b)

simple_test()