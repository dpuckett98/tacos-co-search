import numpy as np
import math

from fastarch.src.types import Node, connect, StateMachine, Port
from fastarch.src.buffer_node import BufferNode
from fastarch.src.dram_node import DRAMNode
from fastarch.src.complex_PE_node import ComplexPENode
from fastarch.src.memory_controller_node_v2 import MemoryControllerNode
from fastarch.src.controller import run_system

# TODO
# (optimization): Start loading data for the next tile once the current tile is finished (remove "idle" cycles)
# (optimization): If the tiling parameters don't exactly fit the input matrices, then reduce the size of the final tiles/chunks to exactly fit the input matrices
#	- then, parameter sweep may no longer have to optimize for an exact fit, and can instead optimize for inter-PE utilization and intra-PE utilization (e.g. making sure the % of PEs active is high, and making sure the % of time a PE spends computing vs loading or flushing is high)
# (optimization???): Vary the # of multipliers inside a PE

# Does matrix-matrix multiplication using multiple PEs
# PE can consume 1 chunk of A and 1 chunk of B to produce 1 chunk of O
# Chip can consume 1 tile of A and 1 tile of B to produce 1 partial tile of O

# outer dataflow (e.g. how tiles are loaded/offloaded -- A-Stationary, B-Stationary, or Output-Stationary)
# inner dataflow (e.g. how chunks are loaded/offloaded in a single tile -- A-Stationary, B-Stationary, Output-Stationary)
# PE dataflow (e.g. how chunks are computed inside a PE -- A-Stationary, B-Stationary, or Output-Stationary) --> this one probably is unnecessary

# preload_cycles and pipeline_offloading assume that you're running lots of matrix multiplications back to back and you can preload inputs at the end of the previous matrix multiplication and offload outputs at the beginning of the next matrix multiplication
class MMDataflow(StateMachine):

	def __init__(self, num_PEs, num_banks_per_PE, size_banks_per_PE, off_chip_bandwidth, on_chip_bandwidth, A_rows, A_cols_B_rows, B_cols, dataflow, t_a, t_b, t_w, c_a, c_b, c_w, estimate, encode=False, decode=False, orig_heads=-1, comp_heads=-1, sparsity = 0.0, preload_cycles=0, pipeline_offloading=False, share_load="None", num_PE_lanes=1, A_transfer_scale=1, B_transfer_scale=1, O_transfer_scale=1, num_heads=1, load_immediate=False, store_immediate=False, sparse_map=None, memory_target_size=-1, memory_initial_size=0):
		# configuration and initialization
		self.num_PEs = num_PEs
		self.num_banks_per_PE = num_banks_per_PE
		self.size_banks_per_PE = size_banks_per_PE
		self.off_chip_bandwidth = off_chip_bandwidth
		self.on_chip_bandwidth = on_chip_bandwidth
		self.A_rows = A_rows
		self.A_cols_B_rows = A_cols_B_rows
		self.B_cols = B_cols
		self.dataflow = dataflow
		self.inner_dataflow = "None"
		self.estimate = estimate
		self.encode = encode
		self.decode = decode
		self.orig_heads = orig_heads
		self.comp_heads = comp_heads
		self.sparsity = sparsity
		self.preload_cycles = preload_cycles
		self.pipeline_offloading = pipeline_offloading
		self.sparse_map = sparse_map
		
		self.start_tile_estimate = 0
		self.estimate_start_cycle = 0
		self.tiles_in_estimate = 1
		self.last_tile_estimate = self.start_tile_estimate + self.tiles_in_estimate
		
		self.A_transfer_scale = A_transfer_scale
		self.B_transfer_scale = B_transfer_scale
		self.O_transfer_scale = O_transfer_scale
		self.num_heads = num_heads
		
		assert not (self.encode and self.decode)
		if self.encode or self.decode:
			assert self.orig_heads != -1 and self.comp_heads != -1
		if self.encode:
			assert self.dataflow == 'Output-Stationary'
			assert t_b % self.orig_heads == 0
		
		# setting up nodes
		
		# DRAM
		self.dram = DRAMNode("DRAM", self.off_chip_bandwidth, initialization_time=0)
		
		# PE
		self.pes = [ComplexPENode("PE_" + str(i), self.num_banks_per_PE, self.size_banks_per_PE, self.on_chip_bandwidth) for i in range(self.num_PEs)]
		
		# Memory Controller
		memory_input_ports = ["in_psums_" + str(i) for i in range(self.num_PEs)]
		memory_output_ports = [item for i in range(self.num_PEs) for item in ["out_A_" + str(i), "out_B_" + str(i), "out_psums_" + str(i)]]
		self.memory = MemoryControllerNode("MemoryController", self.off_chip_bandwidth, self.on_chip_bandwidth, memory_input_ports, memory_output_ports, share_load=share_load, num_PE_lanes=num_PE_lanes, load_immediate=load_immediate, store_immediate=store_immediate, target_size=memory_target_size)
		
		# Connections
		connect(self.dram, "out", self.memory, "DRAM_in")
		connect(self.dram, "in", self.memory, "DRAM_out")
		for i in range(self.num_PEs):
			connect(self.memory, "in_psums_" + str(i), self.pes[i], "out")
			connect(self.memory, "out_A_" + str(i), self.pes[i], "in_A")
			connect(self.memory, "out_B_" + str(i), self.pes[i], "in_B")
			connect(self.memory, "out_psums_" + str(i), self.pes[i], "in_psums")
		
		# initialize tiling parameters
		self.x = c_a
		self.y = c_w
		self.z = c_b
		self.t_a = t_a
		self.t_b = t_b
		self.t_w = t_w
		
		self.tile_rows = math.ceil(A_rows / t_a)
		self.tile_cols = math.ceil(B_cols / t_b)
		self.tile_depth = math.ceil(A_cols_B_rows / t_w)
		self.chunk_rows = math.ceil(t_a / c_a)
		self.chunk_cols = math.ceil(t_b / c_b)
		self.chunk_depth = math.ceil(t_w / c_w)
		
		# per-tile statistics
		self.ideal_cycles_per_tile = self.A_rows / self.tile_rows * self.A_cols_B_rows / self.tile_depth * self.B_cols / self.tile_cols / self.num_PEs
		self.last_tile_start = self.preload_cycles
		self.last_tile_ideal_input_bandwidth_used = 0
		self.last_tile_actual_input_bandwidth_used = 0
		self.last_tile_ideal_output_bandwidth_used = 0
		self.last_tile_actual_output_bandwidth_used = 0
		
		# initialize tiles
		
		self.current_A_dec_tile = None
		self.current_B_dec_tile = None
		self.tiles_assigned = 0
		self.total_dec_tiles = 0
		self.mode = "processing"
		
		self.finished_dec = False
		self.total_dec_tiles = 0
		self.dec_row = 0
		self.dec_col = 0
		self.dec_tiles_completed = 0
		
		self.ready_to_decode = False
		
		if self.decode:
			self.total_dec_tiles = self.tile_rows * self.tile_cols * self.tile_depth
			self.mode = "decoding"
		
		self.current_tile_row = 0
		self.current_tile_col = 0
		self.current_tile_depth = 0
		self.current_tile = self.generate_tile(0, 0, 0) #self.tiles[0][0][0]
		self.current_tile.setup_chunk_sets(self.memory)
		self.load_next_tile()
		
		self.finished_tiles = False
		
		self.finished_enc = True
		self.total_enc_tiles = 0
		self.enc_row = 0
		self.enc_col = 0
		self.enc_tiles_completed = 0
		self.current_enc_tile = None #self.generate_enc_tile(0, 0)
		self.ready_to_encode = False
		
		if self.encode:
			self.finished_enc = False
			self.total_enc_tiles = self.tile_rows * self.tile_cols * self.tile_depth
			self.enc_row = 0
			self.enc_col = 0
			self.enc_tiles_completed = 0
			self.ready_to_encode = False
			#self.generate_enc_tile(0, 0)
			#self.mode = "encoding"
		
		# initialize PE
		self.current_pe_assignments = {} # PE to tile_set
		if preload_cycles == 0:
			for idx, pe in enumerate(self.pes):
				if self.mode == "processing":
					self.assign_chunk(pe, 0)
				else:
					if idx < self.num_PEs // 2:
						self.assign_A_dec_chunk(pe, 0)
					else:
						self.assign_B_dec_chunk(pe, 0)
		
		self.total_cycles = -1
		self.total_dram_accesses = -1
		
		# PE statistics
		self.idle_cycles = 0 # idle because not assigned a task
		self.waiting_cycles = 0 # idle because waiting for data
		self.loading_cycles = 0
		self.flushing_cycles = 0
		self.computing_cycles = 0
		self.last_pe_active_cycle = 0 # marks the last cycle a PE is active
		self.pe_idle_at_end_cycles = 0
		
		# mem statistics
		self.mem_idle_at_end_cycles = 0 # counts the number of consecutive cycles that the input buffer isn't being used at the end of the matrix multiplication (used for preloading data for the next matrix multiplication)
		self.mem_idle_cycles = 0 # counts the number of cycles the memory is idle
		
		# overall statistics
		self.pe_active_cycles = 0
		self.memory_active_cycles = 0
		
		# debug
		self.prev_percent_complete = 0
	
	def assign_chunk(self, pe, current_cycle):
		if not self.finished_tiles:
			# give the PE an assignment
			self.current_tile.get_assignment(self.memory, pe)
			
			# check if the current tile is finished
			if self.current_tile.finished(self.memory):
				#print("starting new tile! old tile:", self.current_tile_row, self.current_tile_col, self.current_tile_depth)
				#print((current_cycle - self.last_tile_start), self.chunk_depth * self.x * self.chunk_cols * self.z * self.chunk_rows * self.y / self.num_PEs)
				print("Finished tile -- row:", self.current_tile_row, "col:", self.current_tile_col, "depth:", self.current_tile_depth)
				print("Cycles:", current_cycle - self.last_tile_start)
				ideal_cycles_curr_tile = self.current_tile.act_cols * self.current_tile.act_rows * self.current_tile.act_depth / self.num_PEs * (1 - self.sparsity)
				#print(ideal_cycles_curr_tile, self.ideal_cycles_per_tile)
				#print("PE Util:", ideal_cycles_curr_tile / (current_cycle - self.last_tile_start) * 100)
				#print("Input DRAM Bandwidth Util:", (self.last_tile_actual_input_bandwidth_used - self.memory.actual_input_bandwidth_used) / (self.last_tile_ideal_input_bandwidth_used - self.memory.ideal_input_bandwidth_used) * 100)
				#print("Output DRAM Bandwidth Util:", (self.last_tile_actual_output_bandwidth_used - self.memory.actual_output_bandwidth_used) / (self.last_tile_ideal_output_bandwidth_used - self.memory.ideal_output_bandwidth_used) * 100)
				self.last_tile_start = current_cycle
				self.last_tile_ideal_input_bandwidth_used = self.memory.ideal_input_bandwidth_used
				self.last_tile_actual_input_bandwidth_used = self.memory.actual_input_bandwidth_used
				self.last_tile_ideal_output_bandwidth_used = self.memory.ideal_output_bandwidth_used
				self.last_tile_actual_output_bandwidth_used = self.memory.actual_output_bandwidth_used
				
				#print("Idle cycles:", self.idle_cycles / self.num_PEs)
				#print(">", end='', flush=True)
				
				# if it is finished, then the PE should still be idle
				assert pe.get_state() == "idle" or pe.next_state == "idle"
				# assign the next tile
				if self.dataflow == "Output-Stationary":
					# clear old A and B data
					#self.current_tile.erase_chunks(self.memory, True, True, False)
					
					self.current_tile_depth += 1
					if self.current_tile_depth == self.tile_depth:
						if self.encode:
							self.ready_to_encode = True
							self.mode = "encoding"
							#print("Flipped ready_to_encode")
							#print("Reg. tile:", self.current_tile_row, self.current_tile_col)
						self.current_tile_col += 1
						self.current_tile_depth = 0
						if self.current_tile_col == self.tile_cols:
							self.current_tile_row += 1
							self.current_tile_col = 0
							if self.current_tile_row == self.tile_rows:
								self.current_tile_col = self.tile_cols
								self.current_tile_depth = self.tile_depth
								# if all tiles have been completed, so flip "done" and return
								self.finished_tiles = True
								return
				elif self.dataflow == "A-Stationary":
					# clear old A and B data
					#self.current_tile.erase_chunks(self.memory, False, True, False)
					
					self.current_tile_col += 1
					if self.current_tile_col == self.tile_cols:
						self.current_tile_col = 0
						self.current_tile_depth += 1
						if self.current_tile_depth == self.tile_depth:
							#self.current_tile.erase_chunks(self.memory, True, False, False)
							self.current_tile_depth = 0
							self.current_tile_row += 1
							if self.current_tile_row == self.tile_rows:
								self.current_tile_col = self.tile_cols
								self.current_tile_depth = self.tile_depth
								self.finished_tiles = True
								return
				elif self.dataflow == "B-Stationary":
					# clear old A and B data
					#self.current_tile.erase_chunks(self.memory, False, True, False)
					
					self.current_tile_row += 1
					if self.current_tile_row == self.tile_rows:
						self.current_tile_row = 0
						self.current_tile_depth += 1
						if self.current_tile_depth == self.tile_depth:
							#self.current_tile.erase_chunks(self.memory, True, False, False)
							self.current_tile_depth = 0
							self.current_tile_col += 1
							if self.current_tile_col == self.tile_cols:
								self.current_tile_row = self.tile_rows
								self.current_tile_depth = self.tile_depth
								self.finished_tiles = True
								return
				else:
					raise Exception(self.dataflow + " dataflow is not supported (outer dataflow)")
				
				# if estimating, then stop here
				if self.estimate: # and self.tiles_assigned + 1 == self.last_tile_estimate:
					#self.current_tile.erase_chunks(self.memory, True, True, False)
					#print("finished tiles")
					self.finished_tiles = True
					#print(self.tiles_assigned)
					return
				
				#if self.estimate and self.tiles_assigned + 1 == self.start_tile_estimate:
				#	self.estimate_start_cycle = current_cycle
				
				# setup the new tile
				self.current_tile = self.generate_tile(self.current_tile_row, self.current_tile_col, self.current_tile_depth) #self.tiles[self.current_tile_row][self.current_tile_col][self.current_tile_depth]
				self.current_tile.setup_chunk_sets(self.memory)
				self.tiles_assigned += 1
				
				if not self.finished_tiles:
					self.load_next_tile()
				
				if self.decode:
					self.mode = "decoding"
				else:
					# go ahead and assign a new chunk set to the PE
					self.assign_chunk(pe, current_cycle)
	
	def load_next_tile(self):
		# get coords of next tile & preload the data
		res = self.get_next_coords()
		if res != None:
			next_row, next_col, next_depth = res
			next_tile = self.generate_tile(next_row, next_col, next_depth)
			next_tile.setup_chunk_sets(self.memory, preload=True)
	
	def get_next_coords(self, row=-1, col=-1, depth=-1):
		next_row = self.current_tile_row if row == -1 else row
		next_col = self.current_tile_col if col == -1 else col
		next_depth = self.current_tile_depth if depth == -1 else depth
		
		if self.dataflow == "Output-Stationary":
			next_depth += 1
			if next_depth == self.tile_depth:
				next_col += 1
				next_depth = 0
				if next_col == self.tile_cols:
					next_row += 1
					next_col = 0
					if next_row == self.tile_rows:
						return None
		elif self.dataflow == "A-Stationary":
			next_col += 1
			if next_col == self.tile_cols:
				next_col = 0
				next_depth += 1
				if next_depth == self.tile_depth:
					#self.current_tile.erase_chunks(self.memory, True, False, False)
					next_depth = 0
					next_row += 1
					if next_row == self.tile_rows:
						return None
		elif self.dataflow == "B-Stationary":
			next_row += 1
			if next_row == self.tile_rows:
				next_row = 0
				next_depth += 1
				if next_depth == self.tile_depth:
					#self.current_tile.erase_chunks(self.memory, True, False, False)
					next_depth = 0
					next_col += 1
					if next_col == self.tile_cols:
						return None
		return [next_row, next_col, next_depth]
	
	def get_prev_coords(self, row=-1, col=-1, depth=-1):
		next_row = self.current_tile_row if row == -1 else row
		next_col = self.current_tile_col if col == -1 else col
		next_depth = self.current_tile_depth if depth == -1 else depth
		
		if self.dataflow == "Output-Stationary":
			next_depth -= 1
			if next_depth < 0:
				next_col -= 1
				next_depth = self.tile_depth - 1
				if next_col < 0:
					next_row -= 1
					next_col = self.tile_cols - 1
					if next_row < 0:
						return None
		elif self.dataflow == "A-Stationary":
			next_col -= 1
			if next_col < 0:
				next_col = self.tile_cols - 1
				next_depth -= 1
				if next_depth < 0:
					#self.current_tile.erase_chunks(self.memory, True, False, False)
					next_depth = self.tile_depth - 1
					next_row -= 1
					if next_row < 0:
						return None
		elif self.dataflow == "B-Stationary":
			next_row -= 1
			if next_row < 0:
				next_row = self.tile_rows - 1
				next_depth -= 1
				if next_depth < 0:
					#self.current_tile.erase_chunks(self.memory, True, False, False)
					next_depth = self.tile_depth - 1
					next_col -= 1
					if next_col < 0:
						return None
		return [next_row, next_col, next_depth]
	
	def generate_tile(self, i, j, k):
		
		curr_rows = self.t_a
		if i == self.tile_rows - 1: # if this is the last row, then shorten the curr_rows to fit the original matrix dims exactly
			curr_rows = self.A_rows - (i) * self.t_a
		curr_cols = self.t_b
		if j == self.tile_cols - 1: # if this is the last col, then shorten the curr_cols to fit the original matrix dims exactly
			curr_cols = self.B_cols - (j) * self.t_b
		curr_depth = self.t_w
		if k == self.tile_depth - 1: # if this is the last depth, then shorten the curr_depth to fit the original matrix dims exactly
			curr_depth = self.A_cols_B_rows - (k) * self.t_w
		
		#print(self.t_a, self.t_b, self.t_w)
		#print(curr_rows, curr_cols, curr_depth)
		
		res = self.get_prev_coords(i, j, k)
		if res != None:
			prev_i, prev_j, prev_k = res
			#print(res)
			A_load_from_DRAM = (prev_i != i or prev_k != k)
			B_load_from_DRAM = (prev_j != j or prev_k != k)
			if k == 0:
				O_load_option = 0 # create new
			elif prev_i != i or prev_j != j:
				O_load_option = 1 # load from DRAM
			else:
				O_load_option = 2 # present in buffer
		else:
			A_load_from_DRAM = True
			B_load_from_DRAM = True
			O_load_option = 0
			#print("here")
		#print(A_load_from_DRAM, B_load_from_DRAM, O_load_option)
		
		A_load_from_DRAM = True
		B_load_from_DRAM = True
		
		res = self.get_next_coords(i, j, k)
		if res != None:
			next_i, next_j, next_k = res
			A_clear_at_end = (i != next_i or k != next_k)
			B_clear_at_end = (j != next_j or k != next_k)
			O_save_to_dram = (i != next_i or j != next_j)
			#print(i, j, k, next_i, next_j, next_k)
		else:
			A_clear_at_end = True
			B_clear_at_end = True
			O_save_to_dram = True
		#print(A_clear_at_end, B_clear_at_end, O_save_to_dram)
		
		out_tile = None
		if self.dataflow == "Output-Stationary":
			#O_load_option = 2
			#if k == 0:
			#	O_load_option = 0
			#if self.decode:
			#	O_load_option = 2
			#O_save_to_dram = False
			#if k == self.tile_depth - 1 and not self.encode:
			#	O_save_to_dram = True
			#res = self.get_next_coords()
			#A_clear_at_end = True
			#B_clear_at_end = True
			#if res != None:
			#	next_row, next_col, next_depth = res
			#	A_clear_at_end = not (next_row == i and next_depth == k)
			#	B_clear_at_end = not (next_col == j and next_depth == k)
				#print(A_clear_at_end, B_clear_at_end)
			out_tile = Tile(self.memory, i * self.t_a, j * self.t_b, k * self.t_w, curr_rows, curr_cols, curr_depth, self.x, self.z, self.y, A_clear_at_end=A_clear_at_end, B_clear_at_end=B_clear_at_end, O_load_option=O_load_option, O_save_to_dram=O_save_to_dram, dataflow=self.inner_dataflow, A_load_from_DRAM=A_load_from_DRAM, B_load_from_DRAM=B_load_from_DRAM, sparsity=self.sparsity, A_transfer_scale=self.A_transfer_scale, B_transfer_scale=self.B_transfer_scale, O_transfer_scale=self.O_transfer_scale, num_heads=self.num_heads, sparse_map=self.sparse_map)
			#out_tile = Tile(self.memory, i * self.t_a, j * self.t_b, k * self.t_w, curr_rows, curr_cols, curr_depth, self.x, self.z, self.y, A_clear_at_end=A_clear_at_end, B_clear_at_end=B_clear_at_end, O_load_option=O_load_option, O_save_to_dram=O_save_to_dram, dataflow=self.inner_dataflow, A_load_from_DRAM=not self.decode, B_load_from_DRAM=not self.decode, sparsity=self.sparsity, A_transfer_scale=self.A_transfer_scale, B_transfer_scale=self.B_transfer_scale, O_transfer_scale=self.O_transfer_scale, num_heads=self.num_heads, sparse_map=self.sparse_map)
			
			# memory, index_A, index_B, length, width, num_chunks_to_decode, single_PE_length, orig_heads, comp_heads, chunk_is_A, clear_weights=False
			if self.decode:
				#print("here")
				val = self.num_banks_per_PE - self.orig_heads
				num_chunks_to_decode_A = math.ceil((self.chunk_rows * self.x) * (self.chunk_depth * self.y // self.orig_heads) / val) #(self.chunk_rows * self.x) * (self.chunk_cols * self.z // self.orig_heads)
				num_chunks_to_decode_B = math.ceil((self.chunk_cols * self.z) * (self.chunk_depth * self.y // self.orig_heads) / val) #(self.chunk_cols * self.z) * (self.chunk_rows * self.x // self.orig_heads)
				#print("A", num_chunks_to_decode_A, self.chunk_rows, self.x, self.chunk_depth, self.y, self.chunk_cols, self.z, self.orig_heads)
				#print("B", num_chunks_to_decode_B, self.chunk_rows, self.x, self.chunk_depth, self.y, self.chunk_cols, self.z, self.orig_heads)
				#out_size = self.x * self.y #min(
				clear_weights = i == self.tile_rows - 1 and j == self.tile_cols - 1 and k == self.tile_depth - 1 #(self.tiles_assigned == self.total_dec_tiles-2)
				self.current_A_dec_tile = Dec_Tile(self.memory, i * self.chunk_rows, k * self.chunk_depth, self.chunk_rows, self.chunk_depth, num_chunks_to_decode_A, val, self.orig_heads, self.comp_heads, True, self.x * self.y, clear_weights)
				self.current_B_dec_tile = Dec_Tile(self.memory, k * self.chunk_depth, j * self.chunk_cols, self.chunk_depth, self.chunk_cols, num_chunks_to_decode_B, val, self.orig_heads, self.comp_heads, False, self.z * self.y, clear_weights)
		elif self.dataflow == "A-Stationary":
			#A_clear_at_end = False
			#if j == self.tile_cols - 1:
			#	A_clear_at_end = True
			#B_clear_at_end = True
			#if self.tile_cols == 1:
			#	B_clear_at_end = False
			#O_load_option = 1
			#if k == 0:
			#	O_load_option = 0
			#O_save_to_dram = True
			out_tile = Tile(self.memory, i * self.t_a, j * self.t_b, k * self.t_w, curr_rows, curr_cols, curr_depth, self.x, self.z, self.y, A_clear_at_end=A_clear_at_end, B_clear_at_end=B_clear_at_end, O_load_option=O_load_option, O_save_to_dram=O_save_to_dram, dataflow=self.inner_dataflow, A_load_from_DRAM=A_load_from_DRAM, B_load_from_DRAM=B_load_from_DRAM, sparsity=self.sparsity, out_tag = str(k * self.chunk_depth), A_transfer_scale=self.A_transfer_scale, B_transfer_scale=self.B_transfer_scale, O_transfer_scale=self.O_transfer_scale, num_heads=self.num_heads, sparse_map=self.sparse_map)
			#out_tile = Tile(self.memory, i * self.t_a, j * self.t_b, k * self.t_w, curr_rows, curr_cols, curr_depth, self.x, self.z, self.y, A_clear_at_end=A_clear_at_end, B_clear_at_end=B_clear_at_end, O_load_option=O_load_option, O_save_to_dram=O_save_to_dram, dataflow=self.inner_dataflow, A_load_from_DRAM=not self.decode, B_load_from_DRAM=not self.decode, sparsity=self.sparsity, out_tag = str(k * self.chunk_depth), A_transfer_scale=self.A_transfer_scale, B_transfer_scale=self.B_transfer_scale, O_transfer_scale=self.O_transfer_scale, num_heads=self.num_heads, sparse_map=self.sparse_map)
			
			if self.decode:
				val = self.num_banks_per_PE - self.orig_heads
				if j == 0:
					clear_weights = (i == self.tile_rows - 1 and k == self.tile_depth - 1)
					self.current_A_dec_tile = Dec_Tile(self.memory, i * self.chunk_rows, k * self.chunk_depth, self.chunk_rows, self.chunk_depth, math.ceil((self.chunk_rows * self.x) * (self.chunk_depth * self.y // self.orig_heads / val)), val, self.orig_heads, self.comp_heads, True, self.x * self.y, clear_weights)
				else:
					self.current_A_dec_tile = None
				clear_weights = (i == self.tile_rows - 1 and j == self.tile_cols - 1 and k == self.tile_depth - 1)
				self.current_B_dec_tile = Dec_Tile(self.memory, k * self.chunk_depth, j * self.chunk_cols, self.chunk_depth, self.chunk_cols, math.ceil((self.chunk_cols * self.z) * (self.chunk_depth * self.y // self.orig_heads / val)), val, self.orig_heads, self.comp_heads, False, self.z * self.y, clear_weights)
		elif self.dataflow == "B-Stationary":
			#A_clear_at_end = True
			#if i == self.tile_rows - 1:
				#A_clear_at_end = True
			#B_clear_at_end = False
			#if i == self.tile_rows - 1:
			#	B_clear_at_end = True
			#if self.tile_cols == 1:
			#	B_clear_at_end = False
			#O_load_option = 1
			#if k == 0:
			#	O_load_option = 0
			#O_save_to_dram = True
			out_tile = Tile(self.memory, i * self.t_a, j * self.t_b, k * self.t_w, curr_rows, curr_cols, curr_depth, self.x, self.z, self.y, A_clear_at_end=A_clear_at_end, B_clear_at_end=B_clear_at_end, O_load_option=O_load_option, O_save_to_dram=O_save_to_dram, dataflow=self.inner_dataflow, A_load_from_DRAM=A_load_from_DRAM, B_load_from_DRAM=B_load_from_DRAM, sparsity=self.sparsity, out_tag = str(k * self.chunk_depth), A_transfer_scale=self.A_transfer_scale, B_transfer_scale=self.B_transfer_scale, O_transfer_scale=self.O_transfer_scale, num_heads=self.num_heads, sparse_map=self.sparse_map)
			#out_tile = Tile(self.memory, i * self.t_a, j * self.t_b, k * self.t_w, curr_rows, curr_cols, curr_depth, self.x, self.z, self.y, A_clear_at_end=A_clear_at_end, B_clear_at_end=B_clear_at_end, O_load_option=O_load_option, O_save_to_dram=O_save_to_dram, dataflow=self.inner_dataflow, A_load_from_DRAM=not self.decode, B_load_from_DRAM=not self.decode, sparsity=self.sparsity, out_tag = str(k * self.chunk_depth), A_transfer_scale=self.A_transfer_scale, B_transfer_scale=self.B_transfer_scale, O_transfer_scale=self.O_transfer_scale, num_heads=self.num_heads, sparse_map=self.sparse_map)
			
			if self.decode:
				val = self.num_banks_per_PE - self.orig_heads
				#print(self.tiles_assigned, self.total_dec_tiles)
				#memory, index_A, index_B, length, width, num_chunks_to_decode, single_PE_length, orig_heads, comp_heads, chunk_is_A, out_size, clear_weights=False
				clear_weights = (i == self.tile_rows - 1 and j == self.tile_cols - 1 and k == self.tile_depth - 1)
				num_chunks_to_decode = math.ceil((self.chunk_rows * self.x) * (self.chunk_depth * self.y // self.orig_heads) / val) #(self.chunk_rows * self.x) * (self.chunk_cols * self.z // self.orig_heads)
				out_size = self.x * self.y #min(
				self.current_A_dec_tile = Dec_Tile(self.memory, i * self.chunk_rows, k * self.chunk_depth, self.chunk_rows, self.chunk_depth, num_chunks_to_decode, val, self.orig_heads, self.comp_heads, True, out_size, clear_weights)
				if i == 0:
					clear_weights = (j == self.tile_cols - 1 and k == self.tile_depth - 1)
					#print(j, k, self.tile_cols, self.tile_depth)
					self.current_B_dec_tile = Dec_Tile(self.memory, k * self.chunk_depth, j * self.chunk_cols, self.chunk_depth, self.chunk_cols, math.ceil((self.chunk_cols * self.z) * (self.chunk_depth * self.y // self.orig_heads) / val), val, self.orig_heads, self.comp_heads, False, self.z * self.y, clear_weights)
				else:
					self.current_B_dec_tile = None
		else:
			raise Exception(self.dataflow + " dataflow is not supported")
		
		return out_tile
	
	def assign_enc_chunk(self, pe, current_cycle):
		if not self.finished_enc:
			
			if self.current_enc_tile != None:
				self.current_enc_tile.get_assignment(self.memory, pe)
			
			if self.enc_row == self.tile_rows and self.enc_col == self.tile_cols and self.current_enc_tile != None and self.current_enc_tile.finished(self.memory):
				#print("finished enc set to true")
				self.finished_enc = True
				#self.memory.print_status()
				return
			
			# only assign a new enc tile if the old one is finished and we're about done with these output buffers
			if (self.current_enc_tile == None or self.current_enc_tile.finished(self.memory)) and self.ready_to_encode:
			
				#print("Assigning encoder chunk")
				self.ready_to_encode = False
				
				# generate out tag
				out_tag = ""
				if self.dataflow == "A-Stationary" or self.dataflow == "B-Stationary":
					out_tag = str(k * self.chunk_depth)
				
				self.generate_enc_tile(self.enc_row, self.enc_col, out_tag)
				
				self.mode = "processing"
				#print("finished encoder processing")
				
				self.enc_col += 1
				if self.enc_col == self.tile_cols:
					self.enc_col = 0
					self.enc_row += 1
					if self.enc_row == self.tile_rows:
						self.enc_col = self.tile_cols
						self.mode = "encoding" # stay in encoding mode to trigger exit
				#print("New enc tile:", self.enc_col, self.enc_row, "out of:", self.tile_cols, self.tile_rows)
				self.current_enc_tile.get_assignment(self.memory, pe)
	
	def generate_enc_tile(self, index_A, index_B, out_tag=""):
		#print("Enc tile:", index_A, index_B, (index_A == self.tile_rows - 1 and index_B == self.tile_cols - 1))
	
		# memory, index_A, index_B, length, width, num_chunks_to_encode, single_PE_length, orig_heads, comp_heads
		
		enc_chunk_width = (self.chunk_cols * self.z) // self.orig_heads
		
		single_PE_length = 1
		#for i in factors(enc_chunk_width):
		#	if i <= self.num_banks_per_PE - self.orig_heads:
		#		single_PE_length = i
		#print(single_PE_length)
		enc_chunk_height = (self.chunk_rows * self.x) // single_PE_length #math.ceil(self.chunk_rows / self.x)
		
		num_chunks_to_encode = enc_chunk_width * enc_chunk_height
		#print(enc_chunk_width, enc_chunk_height, num_chunks_to_encode)
		self.current_enc_tile = Enc_Tile(self.memory, index_A * self.chunk_rows, index_B * self.chunk_cols, self.chunk_rows, self.chunk_cols, num_chunks_to_encode, single_PE_length, self.orig_heads, self.comp_heads, (index_A == self.tile_rows - 1 and index_B == self.tile_cols - 1), out_tag=out_tag)
	
	def assign_A_dec_chunk(self, pe, current_cycle):
		if not self.finished_dec:
			if self.current_A_dec_tile != None:
				self.current_A_dec_tile.get_assignment(self.memory, pe)
	
	def assign_B_dec_chunk(self, pe, current_cycle):
		if not self.finished_dec:
			
			if self.current_B_dec_tile != None:
				self.current_B_dec_tile.get_assignment(self.memory, pe)
			
	
			'''
			# check finish condition?
			
			# only assign a new dec tile if the old one is finished and we're about done with these output buffers
			if (self.current_dec_tile == None or self.current_dec_tile.finished(self.memory)):
			
				#print("Assigning encoder chunk")
				#self.ready_to_encode = False
				
				self.generate_dec_tile(self.dec_row, self.dec_col)
				
				self.dec_col += 1
				if self.dec_col == self.tile_cols:
					self.dec_col = 0
					self.dec_row += 1
					if self.dec_row == self.tile_rows:
						self.dec_col = self.tile_cols
				#print("New enc tile:", self.enc_col, self.enc_row, "out of:", self.tile_cols, self.tile_rows)
				self.current_dec_tile.get_assignment(self.memory, pe)
			'''
	
	#def generate_dec_tile(self, index_A, index_B):
		
	
	def get_nodes(self):
		return [self.dram, self.memory] + [pe for pe in self.pes]

	def finished(self):
		if self.estimate:
			pes_idle = True
			for pe in self.pes:
				if pe.get_state() != "idle":
					pes_idle = False
					break
			if self.finished_tiles:
				for pe in self.pes:
					if pe.get_state() != "idle":
						#pe.print_status()
						break
			return self.finished_tiles and pes_idle
			#return self.get_percent_complete() >= self.finished_tiles and pes_idle
		#if self.finished_tiles:
		#	self.memory.print_status()
		#	print(self.memory.get_flag("idle"))
		#	print(len(self.memory.chunks), len(self.memory.save_chunk_requests))
		#	print("Finished encoder:", self.finished_enc)
		#	num_requests = 0
		#	for port in self.memory.ports.values():
		#		if port.is_current_request():
		#			num_requests += 1
		#			print(port.name)
		#	print("Number of ports with open requests:", num_requests)
		#	num_non_empty = 0
		#	for name, chunk in self.memory.chunks.items():
		#		if not chunk.empty:
		#			num_non_empty += 1
		#	print("Number of non-empty chunks:", num_non_empty)
		#if self.finished_tiles and self.decode:
			#self.memory.erase_chunk("A_dec_weights")
			#self.memory.erase_chunk("B_dec_weights")
			#self.memory.erase_chunk("A_dec_0")
			#self.memory.erase_chunk("B_dec_-")
			#self.memory.print_status()
			#raise Exception()
		return self.finished_tiles and self.memory.get_flag("idle") == "True" and self.finished_enc

	def update(self, current_cycle):
		if current_cycle >= self.preload_cycles:
			if current_cycle == self.preload_cycles:
				self.memory.finish_preload()
			# transition from decoding to processing
			if self.decode and self.mode == "decoding":
				if (self.current_A_dec_tile == None or self.current_A_dec_tile.finished(self.memory)) and (self.current_B_dec_tile == None or self.current_B_dec_tile.finished(self.memory)):
					#print("transitioned to processing")
					#print(self.current_A_dec_tile.finished(self.memory), self.current_B_dec_tile.finished(self.memory))
					#if self.current_A_dec_tile.finished(self.memory) and self.current_B_dec_tile.finished(self.memory):
					self.mode = "processing"
						#print("switched to processing")
			if not self.finished_tiles:
				for idx, pe in enumerate(self.pes):
					if pe.next_state == "idle":
						if self.mode == "processing":
							self.assign_chunk(pe, current_cycle)
							#if pe.name == "PE_0":
							#	print(pe.name, self.current_tile.pe_mapping[pe.name].chunk_name_Out)
						elif self.mode == "encoding":
							self.assign_enc_chunk(pe, current_cycle)
						elif self.mode == "decoding":
							if idx < self.num_PEs // 2:
								if self.current_A_dec_tile != None:
									self.assign_A_dec_chunk(pe, current_cycle)
									#print("assigning A chunk")
								else:
									self.assign_B_dec_chunk(pe, current_cycle)
							else:
								if self.current_B_dec_tile != None:
									self.assign_B_dec_chunk(pe, current_cycle)
									#print("assigning B chunk")
								else:
									self.assign_A_dec_chunk(pe, current_cycle)
					'''
					if self.encode:
						if idx < self.process_PEs:
							self.assign_chunk(pe, current_cycle)
						elif idx < self.process_PEs + self.en_de_PEs:
							self.assign_enc_chunk(pe, current_cycle)
					elif self.decode:
						if idx < self.process_PEs:
							self.assign_chunk(pe, current_cycle)
						elif idx < self.process_PEs + self.en_de_PEs / 2:
							self.assign_A_dec_chunk(pe, current_cycle)
						else:
							self.assign_B_dec_chunk(pe, current_cycle)
					else:
						self.assign_chunk(pe, current_cycle)
					'''
					
			#print(self.mode)
			#self.print_data(current_cycle)
			#print(list(self.memory.chunks.keys())[0])
			#print(len(self.memory.chunks))
			
			#if self.get_percent_complete() == self.prev_percent_complete:
			#	for pe in self.pes:
			#		if not pe.get_state() == "idle":
			#			pe.print_status()
			#			self.current_tile.print_pe_data(pe)
			#			#print(self.current_tile.pe_mapping[pe.name].chunk_name_A)
			#			chunk = self.memory.chunks[self.current_tile.pe_mapping[pe.name].chunk_name_A]
			#			print(chunk.name, chunk.current_size, chunk.max_size, chunk.source_is_dram, chunk.remove_on_read)
			#			print(chunk in self.memory.load_chunk_queue)
			#			break
			#self.prev_percent_complete = self.get_percent_complete()
			
			#if self.get_percent_complete() > 0.085:
				#print(len(self.memory.chunks))
				#self.memory.print_status()
			
			#if self.get_percent_complete() > 0.999:
				#self.pes[200].print_status()
			#if self.mode == "processing":
			#	self.memory.print_status()
			#	raise Exception()
			#	self.pes[41].print_status()
			#self.print_data(current_cycle)
			#	print(self.mode, self.enc_row, self.enc_col)
			#if self.get_percent_complete() > 0.14:
			#	self.memory.print_status()
			#	self.pes[0].print_status()
			#	print(self.memory.ports["DRAM_out"].is_current_request())
			
			#self.pes[0].print_status()
			#self.pes[41].print_status()
			
			done_once = False
			for pe in self.pes:
				if pe.get_state() == "idle":
					self.idle_cycles += 1
				elif pe.get_state() == "loading":
					self.last_pe_active_cycle = current_cycle
					self.loading_cycles += 1
				elif pe.get_state() == "flushing":
					self.flushing_cycles += 1
					self.last_pe_active_cycle = current_cycle
				elif pe.get_state() == "computing":
					self.last_pe_active_cycle = current_cycle
					if pe.get_flag("computing"):
						self.computing_cycles += 1
						if not done_once:
							self.pe_active_cycles += 1
							done_once = True
					else:
						self.waiting_cycles += 1
			
			if self.memory.ports["DRAM_in"].is_current_request():
				self.mem_idle_at_end_cycles = 0
			else:
				self.mem_idle_at_end_cycles += 1
		
		if self.memory.ports["DRAM_in"].is_current_request() or self.memory.ports["DRAM_out"].is_current_request():
			self.memory_active_cycles += 1
		else:
			self.mem_idle_cycles += 1
			if self.mem_idle_cycles >= 100 and current_cycle < self.preload_cycles:
				print("Stopped preloading at", current_cycle)
				self.memory.finish_preload()
				self.mem_idle_cycles += self.preload_cycles - current_cycle
				self.preload_cycles = current_cycle

	def print_data(self, current_cycle):
		print("###############", current_cycle, "###############")
		print("Current tile:", self.current_tile_row, self.current_tile_col, self.current_tile_depth, "out of", self.tile_rows, self.tile_cols, self.tile_depth)
		print("Chunks covered:", self.current_tile_row * self.chunk_rows, self.current_tile_col * self.chunk_cols, self.current_tile_depth * self.chunk_depth, "-", (self.current_tile_row + 1) * self.chunk_rows - 1, (self.current_tile_col + 1) * self.chunk_cols - 1, (self.current_tile_depth + 1) * self.chunk_depth - 1)
		for pe in self.pes:
			pe.print_status()
			self.current_tile.print_pe_data(pe)
		self.memory.print_status()
	
	def get_percent_complete(self):
		if self.estimate:
			return (self.tiles_assigned) / (self.last_tile_estimate) + (1 / (self.last_tile_estimate) * self.current_tile.get_percent_complete())
		else:
			num_total = self.tile_depth * self.tile_rows * self.tile_cols
			return (self.tiles_assigned) / num_total + (1 / num_total) * self.current_tile.get_percent_complete()
	
	def on_complete(self, final_cycles):
		if self.estimate:
			#print(final_cycles, self.estimate_start_cycle, (self.tile_depth * self.tile_rows * self.tile_cols / self.tiles_in_estimate))
			#self.print_data(final_cycles)
			#print()
			print("Final cycles:", final_cycles)
			print("Estimate start cycle:", self.estimate_start_cycle)
			print("Number of tiles total:", self.tile_depth * self.tile_rows * self.tile_cols)
			print("Tiles in estimate:", self.tiles_in_estimate)
			print()
			self.total_cycles = (final_cycles - self.estimate_start_cycle) * (self.tile_depth * self.tile_rows * self.tile_cols / self.tiles_in_estimate)
			self.total_dram_accesses = (self.dram.reads + self.dram.writes) * (self.tile_depth * self.tile_rows * self.tile_cols / self.tiles_in_estimate)
		else:
			print("Cycles spent offloading while PEs are idle:", final_cycles - self.last_pe_active_cycle)
			if self.pipeline_offloading:
				self.pe_idle_at_end_cycles = final_cycles - self.last_pe_active_cycle
				self.total_cycles = self.last_pe_active_cycle - self.preload_cycles
			else:
				self.total_cycles = final_cycles - self.preload_cycles
			self.total_dram_accesses = self.dram.reads + self.dram.writes #self.memory.dram_reads + self.memory.dram_writes
		self.print_on_complete(self.total_cycles)
	
	def print_stats(self):
		print("Chunk dim:", self.x, self.z, self.y)
		print("Chunks per tile dims:", self.chunk_rows, self.chunk_cols, self.chunk_depth)
		print("Tiles in Matrix:", self.tile_rows, self.tile_cols, self.tile_depth)
		#print("Effective size:", self.tile_rows * self.chunk_rows * self.x, self.tile_cols * self.chunk_cols * self.z, self.tile_depth * self.chunk_depth * self.y)
		
		total_comps = 0
		total_chunks = 0
		for pe in self.pes:
			total_comps += pe.num_computations
			total_chunks += pe.num_chunks
		print("Total number of computations:", total_comps)
		print("Total number of chunks computed:", total_chunks)
		
		print("Memory idle cycles:", self.mem_idle_cycles)
		print("DRAM In idle-at-end cycles:", self.mem_idle_at_end_cycles)
		print("Preload cycles:", self.preload_cycles)
		print("Idle cycles:", self.idle_cycles, "(", self.idle_cycles / self.num_PEs, ")")
		print("Waiting for data cycles:", self.waiting_cycles, "(", self.waiting_cycles / self.num_PEs, ")")
		print("Loading cycles:", self.loading_cycles, "(", self.loading_cycles / self.num_PEs, ")")
		print("Flushing cycles:", self.flushing_cycles, "(", self.flushing_cycles / self.num_PEs, ")")
		print("Computing cycles:", self.computing_cycles, "(", self.computing_cycles / self.num_PEs, ")")
	
	def print_memory_usage(self):
		print("Memory-logged DRAM reads/writes:", self.memory.dram_reads + self.memory.dram_writes)
		print("DRAM reads/writes:", self.dram.reads + self.dram.writes)
		print("DRAM reads:", self.dram.reads)
		print("DRAM writes:", self.dram.writes)
		print("Buffer reads/writes:", self.memory.reads + self.memory.writes)
		bank_rw = 0
		out_rw = 0
		for pe in self.pes:
			bank_rw += pe.num_bank_reads + pe.num_bank_writes
			out_rw += pe.num_out_reads + pe.num_out_writes
		print("PE Banks reads/writes:", bank_rw)
		print("PE Out Bank reads/writes:", out_rw)
	
	def print_on_complete(self, final_cycles):
		self.print_stats()
		if not (self.sparse_map is None):
			#print(self.sparse_map.shape)
			#print(np.count_nonzero(self.sparse_map), self.A_cols_B_rows, self.num_PEs, self.num_heads)
			ideal_cycles = np.count_nonzero(self.sparse_map) * self.A_cols_B_rows / self.num_PEs / self.num_heads
		else:
			ideal_cycles = self.A_rows * self.A_cols_B_rows * self.B_cols / self.num_PEs * (1 - self.sparsity) / self.num_heads
		print("Final cycle count:", final_cycles)
		print("Ideal cycle count:", ideal_cycles)
		print("PE active cycles:", self.pe_active_cycles)
		print("Memory active cycles:", self.memory_active_cycles)
		print("Utilization:", ideal_cycles / final_cycles * 100, "%")
		print("Input bandwidth utilization:", self.memory.actual_input_bandwidth_used / self.memory.ideal_input_bandwidth_used * 100, "%")
		print("Output bandwidth utilization:", self.memory.actual_output_bandwidth_used / self.memory.ideal_output_bandwidth_used * 100, "%")
		print("Data to be Offloaded:", self.memory.get_offload_size())
		print("Global Buffer Size:", self.memory.get_min_buffer_size())
		print("Global Buffer Banks:", self.memory.get_min_buffer_banks())
		print("Total PE Memory:", self.num_PEs * (self.num_banks_per_PE * self.size_banks_per_PE + (self.num_banks_per_PE / 2) ** 2))
		self.print_memory_usage()
	
	def get_results(self):
		return [self.total_cycles, self.total_dram_accesses, self.pe_active_cycles, self.memory_active_cycles, self.mem_idle_cycles, self.pe_idle_at_end_cycles, self.memory.get_offload_size()]

class Dec_Tile:

	def __init__(self, memory, index_A, index_B, length, width, num_chunks_to_decode, single_PE_length, orig_heads, comp_heads, chunk_is_A, out_size, clear_weights=False):
		self.index_A = index_A
		self.index_B = index_B
		self.length = length
		self.width = width
		self.single_PE_length = single_PE_length
		self.orig_heads = orig_heads
		self.comp_heads = comp_heads
		self.num_chunks_to_decode = num_chunks_to_decode
		self.clear_weights = clear_weights
		self.chunk_is_A = chunk_is_A
		self.out_size = out_size
		
		self.chunks_setup = False
		self.num_chunks_assigned = 0
		self.num_chunks_finished = 0
		
		self.base = "B"
		if self.chunk_is_A:
			self.base = "A"
		
		# setup chunk sets
		memory.add_chunk(self.base + "_dec_weights", comp_heads * orig_heads, source_is_dram=True)
		
		for i in range(self.num_chunks_to_decode):
			memory.add_chunk(self.base + "_dec_" + str(i), self.single_PE_length * self.comp_heads, source_is_dram=True)
			memory.add_chunk(self.base + "_dec_out_" + str(i), self.single_PE_length * self.orig_heads, source_is_dram=False)
		
		#print(self.num_chunks_to_decode)
		#memory.print_status()
		
		self.pe_mappings = {}
		
		#print(self.clear_weights)
	
	# tries to setup chunk sets; returns True if successful, False otherwise
	# if it's not successful, it's because all the output chunks aren't full
	def setup_chunk_sets(self, memory):
		
		for i in range(self.num_chunks_to_decode):
			memory.add_chunk(self.base + "_dec_" + str(i), self.single_PE_length * self.comp_heads, source_is_dram=True)
		
		self.chunks_setup = True
	
	# takes decoded chunks and reformats them into usable tiles
	def reformat_chunk_sets(self, memory):
		old_chunk_names = [self.base + "_dec_out_" + str(i) for i in range(self.num_chunks_to_decode)]
		new_chunk_names = []
		for i in range(self.index_A, self.index_A + self.length):
			for j in range(self.index_B, self.index_B + self.width):
				new_chunk_names.append(self.base + "_" + str(i) + "_" + str(j))
		
		#print(len(old_chunk_names), self.single_PE_length, self.orig_heads, len(new_chunk_names), self.out_size)
		memory.reformat_chunks(old_chunk_names, new_chunk_names, self.out_size)
		#print("Number of chunks made:", self.length * self.width)
		#print("just reformatted chunks!")
		#memory.print_status()
	
	def get_assignment(self, memory, pe):
		
		if pe.name in self.pe_mappings:
			self.num_chunks_finished += 1
			#print("finished dec assignment", self.num_chunks_finished, "out of", self.num_chunks_to_decode)
			memory.erase_chunk(self.pe_mappings[pe.name])
			del self.pe_mappings[pe.name]
			#print(self.num_chunks_finished)
			if self.num_chunks_finished == self.num_chunks_to_decode:
				self.reformat_chunk_sets(memory)
				if self.clear_weights:
					memory.erase_chunk(self.base + "_dec_weights")
					#print("Cleared weights")
		
		if self.num_chunks_assigned < self.num_chunks_to_decode:
			#print("set operation:", pe.name, ":", self.base + "_dec_" + str(self.num_chunks_assigned))
			memory.set_chunk_for_port(pe.ports["in_A"].connection.source_port.name, self.base + "_dec_" + str(self.num_chunks_assigned))
			memory.set_chunk_for_port(pe.ports["in_B"].connection.source_port.name, self.base + "_dec_weights")
			memory.set_chunk_for_port(pe.ports["out"].connection.sink_port.name, self.base + "_dec_out_" + str(self.num_chunks_assigned))
			#print("assigning dec pe operation")
			pe.set_operation(self.single_PE_length, self.comp_heads, self.orig_heads, False, False, True, False, True)
			self.pe_mappings[pe.name] = self.base + "_dec_" + str(self.num_chunks_assigned)
			self.num_chunks_assigned += 1
			#print("giving out dec assignments", self.num_chunks_assigned, "out of", self.num_chunks_to_decode)
	
	def finished(self, memory):
		done = self.num_chunks_finished >= self.num_chunks_to_decode
		return done

class Enc_Tile:

	# takes out chunks from (index_A, index_B) - (index_A + length, index_B + width) of size (single_PE_length x orig_heads) and multiplies them by encode weights of size (orig_heads x comp_heads)

	def __init__(self, memory, index_A, index_B, length, width, num_chunks_to_encode, single_PE_length, orig_heads, comp_heads, clear_weights=False, out_tag=""):
		self.index_A = index_A
		self.index_B = index_B
		self.length = length
		self.width = width
		self.single_PE_length = single_PE_length
		self.orig_heads = orig_heads
		self.comp_heads = comp_heads
		self.num_chunks_to_encode = num_chunks_to_encode
		self.clear_weights = clear_weights
		self.out_tag = out_tag
		
		self.chunks_setup = False
		self.num_chunks_assigned = 0
		self.num_chunks_finished = 0
		
		memory.add_chunk("enc_weights", comp_heads * orig_heads)
		
		self.pe_mappings = {}
		
		#print(self.num_chunks_to_encode)
	
	# tries to setup chunk sets; returns True if successful, False otherwise
	# if it's not successful, it's because all the output chunks aren't full
	def setup_chunk_sets(self, memory):
		#memory.print_status()
		
		# make sure output chunk sets are full:
		for i in range(self.index_A, self.index_A + self.length):
			for j in range(self.index_B, self.index_B + self.width):
				chunk_name_O = "O_" + str(i) + "_" + str(j) + "_" + self.out_tag
				if not chunk_name_O in memory.chunks:
					#print(chunk_name_O, "not in memory.chunks")
					return False
				if not memory.chunks[chunk_name_O].full:
					#print(chunk_name_O, "not full")
					return False
		
		# reformat old output chunk sets into format for encoding
		old_chunk_names = []
		for i in range(self.index_A, self.index_A + self.length):
			for j in range(self.index_B, self.index_B + self.width):
				chunk_name_O = "O_" + str(i) + "_" + str(j) + "_" + self.out_tag
				old_chunk_names.append(chunk_name_O)
		new_chunk_names = ["enc_" + str(i) for i in range(self.num_chunks_to_encode)]
		
		memory.reformat_chunks(old_chunk_names, new_chunk_names, self.single_PE_length * self.orig_heads)
		
		for i in range(self.num_chunks_to_encode):
			memory.add_chunk("enc_out_" + str(self.index_A) + "_" + str(self.index_B) + "_" + str(i), self.single_PE_length * self.comp_heads, source_is_dram=False)
		
		#print("Adding new chunks!")
		
		self.chunks_setup = True
	
	def get_assignment(self, memory, pe):
		if not self.chunks_setup:
			self.setup_chunk_sets(memory)
		
		if pe.name in self.pe_mappings:
			self.num_chunks_finished += 1
			memory.erase_chunk(self.pe_mappings[pe.name][0])
			memory.give_save_chunk_request(self.pe_mappings[pe.name][1])
			del self.pe_mappings[pe.name]
		
		if self.chunks_setup and self.num_chunks_assigned < self.num_chunks_to_encode:
			
			memory.set_chunk_for_port(pe.ports["in_A"].connection.source_port.name, "enc_" + str(self.num_chunks_assigned))
			memory.set_chunk_for_port(pe.ports["in_B"].connection.source_port.name, "enc_weights")
			memory.set_chunk_for_port(pe.ports["out"].connection.sink_port.name, "enc_out_" + str(self.index_A) + "_" + str(self.index_B) + "_" + str(self.num_chunks_assigned))
			pe.set_operation(self.single_PE_length, self.orig_heads, self.comp_heads, False, False, True, False, True)
			#print("set pe operation for pe", pe.name, "for encoder:", "enc_" + str(self.num_chunks_assigned))
			#memory.print_status()
			self.pe_mappings[pe.name] = ["enc_" + str(self.num_chunks_assigned), "enc_out_" + str(self.index_A) + "_" + str(self.index_B) + "_" + str(self.num_chunks_assigned)]
			self.num_chunks_assigned += 1
			#if self.num_chunks_assigned == self.num_chunks_to_encode:
			#	print("Assigned each chunk in encoder!")
	
	def finished(self, memory):
		done = self.num_chunks_finished >= self.num_chunks_to_encode
		#print(self.num_chunks_finished, self.num_chunks_to_encode)
		if done and self.clear_weights:
			#print("Clearing encoder weights")
			memory.erase_chunk("enc_weights")
		return done

class Tile:
	# index_A, index_B, index_W -> coordinates of tile
	# length, width, depth -> dimensions of tile
	# chunk_rows, chunk_cols, chunk_width -> size of chunk
	# O_load_option -> 0 = create new, 1 = load from DRAM, 2 = present in buffer
	# dataflow -> inner dataflow (e.g. how this tile handles computations inside) - Output-Stationary
	#def __init__(self, memory, index_A, index_B, index_W, act_rows, act_cols, act_depth, chunk_rows, chunk_cols, chunk_width, A_clear_at_end, B_clear_at_end, O_load_option, O_save_to_dram, dataflow, sparsity, A_load_from_DRAM=False, B_load_from_DRAM=False, out_tag="", A_transfer_scale=1, B_transfer_scale=1, O_transfer_scale=1, num_heads=1, sparse_map=None):
	def __init__(self, memory, offset_A, offset_B, offset_W, act_rows, act_cols, act_depth, chunk_rows, chunk_cols, chunk_depth, A_clear_at_end, B_clear_at_end, O_load_option, O_save_to_dram, dataflow, sparsity, A_load_from_DRAM=False, B_load_from_DRAM=False, out_tag="", A_transfer_scale=1, B_transfer_scale=1, O_transfer_scale=1, num_heads=1, sparse_map=None):
		#self.index_A = index_A # coordinates of tile by # of chunks
		#self.index_B = index_B
		#self.index_W = index_W
		self.offset_A = offset_A
		self.offset_B = offset_B
		self.offset_W = offset_W
		#self.act_A = self.index_A * chunk_rows # coordinates of tile by individual row/column
		#self.act_B = self.index_B * chunk_cols
		#self.act_W = self.index_W * chunk_width
		self.act_rows = act_rows # size of tile by individual rows/columns
		self.act_cols = act_cols
		self.act_depth = act_depth
		#print("Starting tile:", act_rows * act_cols * act_depth)
		self.length = math.ceil(act_rows / chunk_rows) # size of tile by # of chunks
		self.width = math.ceil(act_cols / chunk_cols)
		self.depth = math.ceil(act_depth / chunk_depth)
		self.chunk_rows = chunk_rows
		self.chunk_cols = chunk_cols
		self.chunk_depth = chunk_depth
		self.A_clear_at_end = A_clear_at_end
		self.B_clear_at_end = B_clear_at_end
		self.O_load_option = O_load_option
		self.O_save_to_dram = O_save_to_dram
		self.dataflow = dataflow
		self.sparsity = sparsity
		self.A_load_from_DRAM = A_load_from_DRAM
		self.B_load_from_DRAM = B_load_from_DRAM
		self.out_tag = out_tag
		self.sparse_map = sparse_map
		
		self.A_transfer_scale = A_transfer_scale
		self.B_transfer_scale = B_transfer_scale
		self.O_transfer_scale = O_transfer_scale
		
		self.num_heads = num_heads
		
		self.chunk_sets = []
		
		self.pe_mapping = {} # PE name -> ChunkSet
		self.A_chunk_list = []
		self.B_chunk_list = []
		self.Out_chunk_list = set()
		
		# finished flags
		self.num_finished_chunks = 0
		self.num_chunks_total = -1 #self.length * self.width * self.depth
		#print(self.num_chunks_total)
		#print("Number of A chunks needed:", self.length * self.depth)
		#print("Number of B chunks needed:", self.width * self.depth)
		#print("Number of output chunks needed:", self.width * self.length)
		#print(self.width, self.length, self.depth)
	
	def setup_chunk_sets(self, memory, preload=False):
		#print("setting up new chunk set", self.index_A, self.index_B, self.index_W)
		#print("number of save chunk requests:", len(memory.save_chunk_requests))
		#print("number of chunks total:", len(memory.chunks))
		#print(self.length, self.width, self.depth, self.num_chunks_total)
		# generate ChunkSets
		for ii in range(self.length):
			#print(ii)
			i = self.offset_A + ii * self.chunk_rows
			for jj in range(self.width):
				j = self.offset_B + jj * self.chunk_cols
				
				for kk in range(self.depth):
					k = self.offset_W + kk * self.chunk_depth
					chunk_name_A = "A_" + str(i) + "_" + str(k)
					chunk_name_B = "B_" + str(k) + "_" + str(j)
					self.A_chunk_list.append(chunk_name_A)
					self.B_chunk_list.append(chunk_name_B)
				
				#print(i, j, i % self.num_heads, j % self.num_heads)
				if i % self.num_heads == j % self.num_heads: # this assumes that each consecutive chunk is assigned to a different head; thus, the head that chunk (i,k) or (k,j) applies to is i%num_heads or j%num_heads; thus, only add a chunk_set if the two input chunks are from the same head
					for kk in range(self.depth):
						k = self.offset_W + kk * self.chunk_depth
						chunk_name_A = "A_" + str(i) + "_" + str(k)
						chunk_name_B = "B_" + str(k) + "_" + str(j)
						chunk_name_O = "O_" + str(i) + "_" + str(j) + "_" + self.out_tag
						#if i == self.index_A and j == self.index_B and k == self.index_W:
						#	print(chunk_name_A, chunk_name_B)
						#print(self.chunk_rows, i, ii, chunk_name_A)
						#self.A_chunk_list.append(chunk_name_A)
						#self.B_chunk_list.append(chunk_name_B)
						
						x = self.chunk_rows
						if ii == self.length - 1:
							x = self.act_rows - ii * self.chunk_rows
						y = self.chunk_cols
						if jj == self.width - 1:
							y = self.act_cols - jj * self.chunk_cols
						z = self.chunk_depth
						if kk == self.depth - 1:
							z = self.act_depth - kk * self.chunk_depth
						
						row_start = i
						row_end = row_start + x
						col_start = j
						col_end = col_start + y
						
						#if i == 0 and k == 0:
						#	print(row_start, row_end, col_start, col_end)
						
						# always load the data, but only create a chunk set if it's not 0s
						if not (self.sparse_map is None):
							sub_map = self.sparse_map[row_start:row_end, col_start:col_end]
							#if sub_map.shape[0] != x or sub_map.shape[1] != y:
							#	print(self.act_cols, self.index_B * self.chunk_cols, (j - self.index_B), self.chunk_cols)
							#	print(row_start, row_end, col_start, col_end)
							#	print(x, y, sub_map.shape)
							if np.any(sub_map):
								#print("Skipping chunk", chunk_name_O)
								#if self.O_load_option == 1:
								#	print("here")
								cs = ChunkSet(chunk_name_A, chunk_name_B, chunk_name_O, i, j, k, x, y, z)
								self.chunk_sets.append(cs)
						else:
							cs = ChunkSet(chunk_name_A, chunk_name_B, chunk_name_O, i, j, k, x, y, z)
							self.chunk_sets.append(cs)
						
						self.A_chunk_list.append(chunk_name_A)
						self.B_chunk_list.append(chunk_name_B)
						self.Out_chunk_list.add(chunk_name_O)
						
						#print(x, y, z)
						#print(self.chunk_rows, self.chunk_cols, self.chunk_width)
						#cs = ChunkSet(chunk_name_A, chunk_name_B, chunk_name_O, i, j, k, x, y, z)
						#cs.print()
						#self.chunk_sets.append(cs)
						if self.dataflow == "Output-Stationary": # WARNING: This dataflow hasn't been updated with bandwidth & tile size upgrades or with transfer_scale upgrades
							memory.add_chunk(chunk_name_A, self.chunk_rows * self.chunk_depth, source_is_dram=True)
							if not preload:
								memory.update_chunk(chunk_name_A, source_is_dram=True)
							if not self.A_load_from_DRAM and not preload:
								memory.update_chunk(chunk_name_A, source_is_dram=False)
							memory.add_chunk(chunk_name_B, self.chunk_cols * self.chunk_depth, source_is_dram=True)
							if not preload:
								memory.update_chunk(chunk_name_B, source_is_dram=True)
							if not self.B_load_from_DRAM and not preload:
								memory.update_chunk(chunk_name_B, source_is_dram=False)
							if k == self.index_W:
								if self.O_load_option == 1:
									memory.add_chunk(chunk_name_O, self.chunk_rows * self.chunk_cols, source_is_dram=True, remove_on_read=True)
									if not preload:
										memory.update_chunk(chunk_name_O, source_is_dram=True, remove_on_read=True)
								else:
									memory.add_chunk(chunk_name_O, self.chunk_rows * self.chunk_cols, source_is_dram=False, remove_on_read=True)
									if not preload:
										memory.update_chunk(chunk_name_O, source_is_dram=False, remove_on_read=True)
						elif self.dataflow == "None":
							memory.add_chunk(chunk_name_A, x * z, source_is_dram=True, transfer_scale=self.A_transfer_scale)
							if not preload:
								memory.update_chunk(chunk_name_A, source_is_dram=True)
							if not self.A_load_from_DRAM and not preload:
								memory.update_chunk(chunk_name_A, source_is_dram=False)
							memory.add_chunk(chunk_name_B, y * z, source_is_dram=True, transfer_scale=self.B_transfer_scale)
							if not preload:
								memory.update_chunk(chunk_name_B, source_is_dram=True)
							if not self.B_load_from_DRAM and not preload:
								memory.update_chunk(chunk_name_B, source_is_dram=False)
							#if k == self.index_W:
							if self.O_load_option == 1:
								memory.add_chunk(chunk_name_O, x * y, source_is_dram=True, remove_on_read=False, cap_size=True, transfer_scale=self.O_transfer_scale)
								if not preload:
									memory.update_chunk(chunk_name_O, source_is_dram=True, remove_on_read=False, cap_size=True)
							else:
								memory.add_chunk(chunk_name_O, x * y, source_is_dram=False, remove_on_read=False, cap_size=True, transfer_scale=self.O_transfer_scale)
								if not preload:
									memory.update_chunk(chunk_name_O, source_is_dram=False, remove_on_read=False, cap_size=True)
						else:
							raise ValueError(dataflow + " dataflow not supported (inner dataflow)")
		#print("Divided by:", self.num_chunks_total / len(self.chunk_sets))
		self.num_chunks_total = len(self.chunk_sets)
		#print("Num chunks:", self.num_chunks_total)
	
	# erase A and B chunks
	def erase_chunks(self, memory, erase_A, erase_B, erase_Out):
		if erase_A:
			for chunk in self.A_chunk_list:
				memory.erase_chunk(chunk)
				#print("erasing", chunk)
		if erase_B:
			for chunk in self.B_chunk_list:
				memory.erase_chunk(chunk)
		if erase_Out:
			for chunk in self.Out_chunk_list:
				memory.erase_chunk(chunk)
	
	def get_assignment(self, memory, pe):
		# increment finished PEs
		if pe.name in self.pe_mapping and self.pe_mapping[pe.name] != None:
			self.num_finished_chunks += 1
			self.pe_mapping[pe.name] = None
		
		# assign new PE
		if self.dataflow == "Output-Stationary":
			need_new_chunk = False
			old_chunk_set = None
			
			# get old_chunk_set
			if pe.name in self.pe_mapping and self.pe_mapping[pe.name] != None:
				old_chunk_set = self.pe_mapping[pe.name]
			else:
				need_new_chunk = True
			
			# check if there are any chunks with the same output chunk as the old_chunk_set
			if old_chunk_set != None:
				num_available = 0
				chosen_chunk_set = None
				for chunk_set in self.chunk_sets:
					if chunk_set.chunk_name_Out == old_chunk_set.chunk_name_Out:
						num_available += 1
						if chosen_chunk_set == None:
							chosen_chunk_set = chunk_set
				
				# if there are, assign the PE to that chunk_set
				if chosen_chunk_set != None:
					self.pe_mapping[pe.name] = chosen_chunk_set
					memory.set_chunk_for_port(pe.ports["in_A"].connection.source_port.name, chosen_chunk_set.chunk_name_A)
					memory.set_chunk_for_port(pe.ports["in_B"].connection.source_port.name, chosen_chunk_set.chunk_name_B)
					memory.set_chunk_for_port(pe.ports["out"].connection.sink_port.name, chosen_chunk_set.chunk_name_Out)
					pe.set_operation(self.chunk_rows, self.chunk_depth, self.chunk_cols, False, True, num_available == 1, False, False, sparsity = self.sparsity)
					self.chunk_sets.remove(chosen_chunk_set)
					return
			
			# if we get here, we know that the PE needs to start a new output chunk
			
			# first, dispose of the old output chunk (if it exists)
			#if old_chunk_set != None and self.O_save_to_dram == True:
				#if old_chunk_set.chunk_name_Out == "O_38_82":
				#	print("requesting to save O_38_82")
				#	count = 0
				#	for cs in chunk_sets:
				#		if cs.chunk_name_Out == "O_38_82":
				#			count += 1
				#	print(count, "many more chunk sets need O_38_82")
				#memory.give_save_chunk_request(old_chunk_set.chunk_name_Out)
			
			# then, check if there's a new output chunk to start
			chosen_chunk_set = None
			for chunk_set in self.chunk_sets:
				if chunk_set.depth == 0:
					chosen_chunk_set = chunk_set
					break
			
			# if there's a new output chunk to start, start it!
			if chosen_chunk_set != None:
				self.pe_mapping[pe.name] = chosen_chunk_set
				memory.set_chunk_for_port(pe.ports["in_A"].connection.source_port.name, chosen_chunk_set.chunk_name_A)
				memory.set_chunk_for_port(pe.ports["in_B"].connection.source_port.name, chosen_chunk_set.chunk_name_B)
				memory.set_chunk_for_port(pe.ports["out"].connection.sink_port.name, chosen_chunk_set.chunk_name_Out)
				memory.set_chunk_for_port(pe.ports["in_psums"].connection.source_port.name, chosen_chunk_set.chunk_name_Out)
				#print("out port name:", pe.ports["in_psums"].connection.source_port.name, chosen_chunk_set.chunk_name_A)
				psum_load = self.O_load_option != 0 and chosen_chunk_set.depth == 0
				
				# get number of chunk_sets with the same output chunk; if there's only one available, then set psum_flush to true
				num_available = 0
				for chunk_set in self.chunk_sets:
					if chunk_set.chunk_name_Out == chosen_chunk_set.chunk_name_Out:
						num_available += 1
				psum_flush = num_available == 1
				pe.set_operation(self.chunk_rows, self.chunk_depth, self.chunk_cols, psum_load, False, psum_flush, False, False, sparsity = self.sparsity)
				self.chunk_sets.remove(chosen_chunk_set)
			else:
				# if there's no new output chunk to start, unmap this PE
				self.pe_mapping[pe.name] = None
		elif self.dataflow == "None":
			# first give any chunks where depth == 0
			'''
			res = None
			for cs in self.chunk_sets:
				if cs.depth == 0:
					res = cs
					break
			if res != None:
				# assign this chunk
				self.pe_mapping[pe.name] = res
				
				memory.set_chunk_for_port(pe.ports["in_A"].connection.source_port.name, res.chunk_name_A)
				memory.set_chunk_for_port(pe.ports["in_B"].connection.source_port.name, res.chunk_name_B)
				memory.set_chunk_for_port(pe.ports["out"].connection.sink_port.name, res.chunk_name_Out)
				memory.set_chunk_for_port(pe.ports["in_psums"].connection.source_port.name, res.chunk_name_Out)
				pe.set_operation(res.x, res.z, res.y, False, False, True, False, False, sparsity = self.sparsity)
				#pe.set_operation(res.x, res.z, res.y, self.O_load_option != 0, False, True, False, False, sparsity = self.sparsity)
				self.chunk_sets.remove(res)
				#print("Assigned pe", pe.name, "to:", end='')
				#res.print()
				#print(len(self.chunk_sets))
				#if pe.name == "PE_79":
				#	print(res.chunk_name_A, res.chunk_name_B, res.chunk_name_Out)
				
				return
			'''
			# then just pass out the rest in any order
			if len(self.chunk_sets) > 0:
				res = self.chunk_sets[0]
				# assign this chunk
				self.pe_mapping[pe.name] = res
				#print("Assigned pe", pe.name, "to:", end='')
				#res.print()
				memory.set_chunk_for_port(pe.ports["in_A"].connection.source_port.name, res.chunk_name_A)
				memory.set_chunk_for_port(pe.ports["in_B"].connection.source_port.name, res.chunk_name_B)
				memory.set_chunk_for_port(pe.ports["out"].connection.sink_port.name, res.chunk_name_Out)
				memory.set_chunk_for_port(pe.ports["in_psums"].connection.source_port.name, res.chunk_name_Out)
				#pe.set_operation(res.x, res.z, res.y, True, False, True, False, False, sparsity = self.sparsity)
				if self.sparse_map is None:
					pe.set_operation(res.x, res.z, res.y, False, False, True, False, False, sparsity = self.sparsity, sparse_map=None) # changed behavior so that psums aren't loaded onto PEs anymore
				else:
					#print(res.row, res.column, res.x, res.y)
					#row_start = (res.row + self.index_A) * self.chunk_rows
					#row_end = row_start + res.x
					#col_start = (res.column + self.index_B) * self.chunk_cols
					#col_end = col_start + res.y
					
					row_start = res.row #* self.chunk_rows
					row_end = row_start + res.x
					col_start = res.column #* self.chunk_cols
					col_end = col_start + res.y
					
					sub_map = self.sparse_map[row_start:row_end, col_start:col_end]
					#print(sub_map.shape)
					#print(row_start, row_end, col_start, col_end)
					pe.set_operation(res.x, res.z, res.y, False, False, True, False, False, sparsity = self.sparsity, sparse_map=sub_map) # changed behavior so that psums aren't loaded onto PEs anymore
				self.chunk_sets.remove(res)
				#print("Assigned pe", pe.name, "to:", end='')
				#res.print()
				#print(len(self.chunk_sets))
				
				#if pe.name == "PE_79":
				#	print(res.chunk_name_A, res.chunk_name_B, res.chunk_name_Out)
					
				return
		else:
			raise ValueError(dataflow + " dataflow not supported (inner dataflow)")
	
	def finished(self, memory):
		done = (self.num_finished_chunks == self.num_chunks_total)
		if done:
			#print(self.A_clear_at_end, "clearing A")
			self.erase_chunks(memory, self.A_clear_at_end, self.B_clear_at_end, False)
			if self.dataflow == "None" and self.O_save_to_dram:
				for chunk in self.Out_chunk_list:
					memory.give_save_chunk_request(chunk)
		return done
	
	def get_percent_complete(self):
		if self.num_chunks_total == 0:
			return 1
		return self.num_finished_chunks / self.num_chunks_total
	
	def print_pe_data(self, pe):
		if pe.name in self.pe_mapping and self.pe_mapping[pe.name] != None:
			self.pe_mapping[pe.name].print()
		else:
			print("PE not mapped to a chunk set")
	
	def is_pe_assigned(self, pe):
		if pe.name in self.pe_mapping and self.pe_mapping[pe.name] != None:
			return True
		return False
	

class ChunkSet:
	def __init__(self, chunk_name_A, chunk_name_B, chunk_name_Out, row, column, depth, x, y, z):
		self.chunk_name_A = chunk_name_A
		self.chunk_name_B = chunk_name_B
		self.chunk_name_Out = chunk_name_Out
		self.row = row
		self.column = column
		self.depth = depth
		self.x = x
		self.y = y
		self.z = z
	
	def print(self):
		print("A Chunk:", self.chunk_name_A, "B Chunk:", self.chunk_name_B, "Out Chunk:", self.chunk_name_Out)

def calc_actual_dims(A_buffer_size, B_buffer_size, O_buffer_size, x, z, y, A_rows, B_cols, A_cols_B_rows):
	total_A_chunks_per_tile = math.ceil(A_buffer_size / (x * y))
	total_B_chunks_per_tile = math.ceil(B_buffer_size / (z * y))
	total_O_chunks_per_tile = math.ceil(O_buffer_size / (x * z))
	
	chunk_depth = (total_A_chunks_per_tile * total_B_chunks_per_tile / total_O_chunks_per_tile) ** 0.5
	chunk_cols = total_O_chunks_per_tile / total_A_chunks_per_tile * chunk_depth
	chunk_rows = total_A_chunks_per_tile / chunk_depth
	
	chunk_depth = max(1, math.floor(chunk_depth))
	chunk_cols = max(1, math.floor(chunk_cols))
	chunk_rows = max(1, math.floor(chunk_rows))
	
	tile_rows = math.ceil(A_rows / (chunk_rows * x)) # rows of tiles in the entire mult.
	tile_cols = math.ceil(B_cols / (chunk_cols * z))
	tile_depth = math.ceil(A_cols_B_rows / (chunk_depth * y))
	return [tile_rows * chunk_rows * x, tile_cols * chunk_cols * z, tile_depth * chunk_depth * y, tile_rows / chunk_rows, tile_cols / chunk_cols, tile_depth / chunk_depth]

def optimize_params(total_memory, num_banks_per_PE, size_banks_per_PE, A_rows, B_cols, A_cols_B_rows):
	memory_step_size = 100
	ideal_size = A_rows * B_cols * A_cols_B_rows
	best = 100000000000000000000
	best_dim = []
	for A_buffer_size in range(memory_step_size, total_memory - memory_step_size*2 + 1, memory_step_size):
		for B_buffer_size in range(memory_step_size, total_memory - A_buffer_size - memory_step_size + 1, memory_step_size):
			for O_buffer_size in range(memory_step_size, total_memory - A_buffer_size - B_buffer_size + 1, memory_step_size):
				for x in range(1, num_banks_per_PE):
					for z in range(1, num_banks_per_PE - x + 1):
						for y in range(1, size_banks_per_PE + 1):
							act_rows, act_cols, act_depth, a, b, c = calc_actual_dims(A_buffer_size, B_buffer_size, O_buffer_size, x, z, y, A_rows, B_cols, A_cols_B_rows)
							curr = act_rows * act_cols * act_depth
							if curr <= best:
								best = curr
								best_dim = [A_buffer_size, B_buffer_size, O_buffer_size, x, z, y]
	print(best, best_dim, ideal_size)
	print(calc_actual_dims(best_dim[0], best_dim[1], best_dim[2], best_dim[3], best_dim[4], best_dim[5], A_rows, B_cols, A_cols_B_rows))
	return best_dim

def factors(n):
    results = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            results.add(i)
            results.add(int(n/i))
    return results

def optimize_params_2(num_PEs, total_memory, num_banks_per_PE, size_banks_per_PE, A_rows, B_cols, A_cols_B_rows):
	memory_step_size = 100
	ideal_size = A_rows * B_cols * A_cols_B_rows
	best = 0
	best_dim = []
	
	A_rows_factors = factors(A_rows)
	B_cols_factors = factors(B_cols)
	A_cols_B_rows_factors = factors(A_cols_B_rows)
	#c_w = max(factors(size_banks_per_PE))
	
	count = 0
	
	pe_weight = 10
	tile_weight = 1 / (A_rows * B_cols * A_cols_B_rows)
	chunk_weight = 10 / (num_banks_per_PE * size_banks_per_PE)
	
	for t_a in A_rows_factors:
		t_a_factors = factors(t_a)
		for t_b in B_cols_factors:
			t_b_factors = factors(t_b)
			for t_w in A_cols_B_rows_factors:
				if t_a * t_w + t_b * t_w + t_a * t_b > total_memory:
					continue
				c_w = 1
				for i in range(size_banks_per_PE, -1, -1):
					if t_w % i == 0:
						c_w = i
						break
				for x in range(1, num_banks_per_PE):
					z = num_banks_per_PE - x
					for c_a in t_a_factors:
						if c_a > x:
							continue
						for c_b in t_b_factors:
							if c_b > z:
								continue
							count += 1
							
							# for output stationary inner dataflow
							num = ((t_a // c_a) * (t_b // c_b)) % num_PEs
							#print(num)
							tile_size = t_a * t_b * t_w
							chunk_size = c_a * c_b * c_w
							
							score = -num * pe_weight + tile_size * tile_weight + chunk_size * chunk_weight
							
							if score > best:
								best = score
								best_dim = [t_a, t_b, t_w, c_a, c_b, c_w]
							
							'''
							# first, make sure num >= num_PEs
							num = (t_a // c_a) * (t_b // c_b)
							#if num < num_PEs:
							#	score = 1
							#else:
							#	score = t_a * t_b * t_w #c_a * c_b * c_w
							# then rank by c_a and c_b
							res = num % num_PEs
							score = res
							#if res > num_PEs // 2:
							#	score = num_PEs - res
							if score < best:
								best = score
								best_dim = [t_a, t_b, t_w, c_a, c_b, c_w]
							elif num == best:
								#if t_a * t_b * t_w > best_dim[0] * best_dim[1] * best_dim[2]:
								if c_a * c_b * c_w > best_dim[3] * best_dim[4] * best_dim[5]:
									best = score
									best_dim = [t_a, t_b, t_w, c_a, c_b, c_w]
							'''
	
	print("Count:", count)
	print(best, best_dim, ideal_size)
	#print(calc_actual_dims(best_dim[0], best_dim[1], best_dim[2], best_dim[3], best_dim[4], best_dim[5], A_rows, B_cols, A_cols_B_rows))
	return best_dim

# returns total_cycles, utilization, a buffer min size, reads, writes, b buffer min size, reads, writes, o buffer min size, reads, writes, dram accesses, reads, writes
# PE_config = 0 means PE_width and PE_height are given; PE_config = 1 means PE_width and PE_height are switched
def run_MM_dataflow(num_PEs, num_RFs_per_PE, size_RF, off_chip_bandwidth, on_chip_bandwidth, max_sram_size, A_rows, A_cols_B_rows, B_cols, dataflow, t_a, t_b, t_w, c_a, c_b, c_w, estimate=False, encode=False, decode=False, orig_heads=-1, comp_heads=-1, sparsity=0.0, preload_cycles=0, pipeline_offloading=False, share_load="None", num_PE_lanes=1, A_transfer_scale=1, B_transfer_scale=1, O_transfer_scale=1, num_heads=1, load_immediate=False, store_immediate=False, sparse_map=None, memory_target_size=-1, memory_initial_size=0):
	
	state_machine = MMDataflow(num_PEs = num_PEs, num_banks_per_PE = num_RFs_per_PE, size_banks_per_PE = size_RF, off_chip_bandwidth = off_chip_bandwidth, on_chip_bandwidth = on_chip_bandwidth, A_rows = A_rows, A_cols_B_rows = A_cols_B_rows, B_cols = B_cols, dataflow = dataflow, t_a = t_a, t_b = t_b, t_w = t_w, c_a = c_a, c_b = c_b, c_w = c_w, estimate=estimate, encode=encode, decode=decode, orig_heads=orig_heads, comp_heads=comp_heads, sparsity = sparsity, preload_cycles=preload_cycles, pipeline_offloading=pipeline_offloading, share_load=share_load, num_PE_lanes=num_PE_lanes, A_transfer_scale=A_transfer_scale, B_transfer_scale=B_transfer_scale, O_transfer_scale=O_transfer_scale, num_heads=num_heads, load_immediate=load_immediate, store_immediate=store_immediate, sparse_map=sparse_map, memory_target_size=memory_target_size, memory_initial_size=memory_initial_size)
	
	#state_machine = MMDataflow(A_buffer_size = params[0], B_buffer_size = params[1], O_buffer_size = params[2], num_PEs = num_PEs, num_banks_per_PE = 10, size_banks_per_PE = 10, off_chip_bandwidth = 100, on_chip_bandwidth = 10, A_rows = A_rows, A_cols_B_rows = A_cols_B_rows, B_cols = B_cols, dataflow = "Output-Stationary", x = params[3], y = params[5], z = params[4])

	#state_machine = MMDataflow(A_buffer_size = params[0], B_buffer_size = params[1], O_buffer_size = params[2], num_PEs = num_PEs, num_banks_per_PE = 10, size_banks_per_PE = 10, off_chip_bandwidth = 100, on_chip_bandwidth = 10, A_rows = A_rows, A_cols_B_rows = A_cols_B_rows, B_cols = B_cols, dataflow = "B-Stationary", t_a = params[0], t_b = params[1], t_w = params[2], c_a = params[3], c_b = params[4], c_w = params[5])

	run_system(state_machine, state_machine.get_nodes(), 100000000, False)
	
	return state_machine.get_results()

#A_rows = 3025
#A_cols_B_rows = 363
#B_cols = 96
'''
# 196, 576, 192
A_rows = 200
A_cols_B_rows = 600
B_cols = 200
num_PEs = 500
dataflow = 'Output-Stationary'
t_a = 100
t_b = 100
t_w = 100
c_a = 5
c_b = 5
c_w = 10
run_MM_dataflow(num_PEs, 20, 10, 100, 10, 10000, A_rows, A_cols_B_rows, B_cols, dataflow, t_a, t_b, t_w, c_a, c_b, c_w, estimate=True, encode=False, decode=False, orig_heads=10, comp_heads=5, sparsity=0.25)
'''

#Dims: 196, 768, 192 FLOPS: 28901376 Params: 147456 Enc/Dec: 1.0 Sparsity: 0.0
#[11, 65, 48]
#[9, 7, 16]

if __name__ == "__main__":
	sparsity = 0.9
	orig_heads = 10
	comp_heads = 2
	encdec = False
	
	cycles_1, dram_1, c, m = run_MM_dataflow(512, 11, 10, 40, 10, 107776, 198, 64, 198, "Output-Stationary", 140, 140, 140, 1, 1, 10, estimate=False, encode=False, decode=False, orig_heads=orig_heads, comp_heads=comp_heads, sparsity=0.0)
	
	print(cycles_1, dram_1, c, m)
	
	#cycles_2, dram_2 = run_MM_dataflow(100, 11, 10, 100, 10, 1000, 200, 200, 200, "Output-Stationary", 100, 100, 100, 5, 5, 10, estimate=False, encode=False, decode=encdec, orig_heads=orig_heads, comp_heads=comp_heads, sparsity=sparsity)
	
	#ideal_cycles = (200 * 200 * 200 * 2 + 200*200*200*(1-sparsity)) / 100
	#print("Total cycles:", cycles_1 * 2 + cycles_2)
	#print("Ideal cycles:", ideal_cycles)
	#print("Utilization:", ideal_cycles / (cycles_1 * 2 + cycles_2) * 100)
	#print("Total dram:", dram_1 * 2 + dram_2)

#params = optimize_params_2(num_PEs, 10000, 10, 10, A_rows, B_cols, A_cols_B_rows)
#params = [1, 100, 100, 1, 5, 10]
#for num_PEs in [1, 5, 10, 50, 100]:
#	print("\n\n")
#	params = optimize_params_2(num_PEs, 10000, 10, 10, A_rows, B_cols, A_cols_B_rows)
#run_MM_dataflow(A_rows, A_cols_B_rows, B_cols, params, num_PEs)