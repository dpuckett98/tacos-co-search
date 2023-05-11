import numpy as np
import math

import fastarch.conv_helper as ch

from fastarch.src.types import Node, connect, StateMachine, Port
from fastarch.src.buffer_node import BufferNode
from fastarch.src.dram_node import DRAMNode
from fastarch.src.complex_PE_node import ComplexPENode
from fastarch.src.memory_controller_node_v2 import MemoryControllerNode
from fastarch.src.controller import run_system

class ConvDataflow(StateMachine):
	
	# params: [c_in, x/y/ci/co, x/y/ci/co, x/y/ci/co, x/y/ci/co, t_x, t_y, t_ci, t_co, x/y/ci/kx/ky/co, x/y/ci/kx/ky/co, x/y/ci/kx/ky/co, x/y/ci/kx/ky/co, x/y/ci/kx/ky/co, x/y/ci/kx/ky/co, c_x, c_y, c_ci, c_kx, c_ky, c_co]
	def __init__(self, hw_config, layer, params, preload_cycles=0, pipeline_offloading=False):
		if not isinstance(layer, ch.ConvLayer):
			print("Error! ConvDataflow running on non-conv layer")
			return
	
		# configuration
		self.hw_config = hw_config
		self.layer = layer
		self.preload_cycles = preload_cycles
		self.pipeline_offloading = pipeline_offloading
		
		self.act_rows = layer.prows #if params[0] == "cols" else math.ceil(layer.prows // self.hw_config.num_PE_lanes)
		self.act_cols = layer.pcols #if params[0] == "rows" else math.ceil(layer.pcols // self.hw_config.num_PE_lanes)
		#print(self.act_rows, self.act_cols)
		
		self.act_c_out = math.ceil(layer.c_out / self.hw_config.num_PE_lanes)
		
		# unpacking parameters
		
		# params[0] --> PE array dataflow
		if params[0] != "c_out":
			print("Error! ConvDataflow has invalid parameters:", params[0])
			return
		self.share_load = "rows"
		#if params[0] == "rows":
		#	self.share_load = "B"
		#elif params[0] == "cols":
		#	self.share_load = "A"
		#else:
		#	self.share_load = ""
		
		self.lane_dataflow = params[0]
		
		# params[1-4] --> tile dataflow order
		self.dataflow = params[1:5]
		
		# check that the dataflow values are valid & there aren't any duplicates
		got_tx = False
		got_ty = False
		got_tci = False
		got_tco = False
		for d in self.dataflow:
			if d == "x":
				got_tx = True
			elif d == "y":
				got_ty = True
			elif d == "ci":
				got_tci = True
			elif d == "co":
				got_tco = True
		if not (got_tx and got_ty and got_tci and got_tco):
			print("Error! ConvDataflow has wrong tile dataflow params:", self.dataflow)
			return
		
		# params[5-8] --> tile sizes
		self.t_x = params[5] #if params[0] == "rows" else math.ceil(params[5] // self.hw_config.num_PE_lanes)
		self.t_y = params[6] #if params[0] == "cols" else math.ceil(params[6] // self.hw_config.num_PE_lanes)
		self.t_ci = params[7]
		self.t_co = math.ceil(params[8] / self.hw_config.num_PE_lanes)
		
		# params[9-14] --> chunk dataflow order
		self.chunk_dataflow = params[9:15]
		
		# TODO: verify chunk dataflow is valid
		
		# params[15-20] --> chunk sizes
		self.c_x = params[15]
		self.c_y = params[16]
		self.c_ci = params[17]
		self.c_kx = params[18]
		self.c_ky = params[19]
		self.c_co = math.ceil(params[20] / self.hw_config.num_PE_lanes)
		
		# TODO: verify chunk params are valid
		
		# setting up nodes
		
		# DRAM
		self.dram = DRAMNode("DRAM", self.hw_config.off_chip_bandwidth, initialization_time=0)
		
		# PE
		self.pes = [ComplexPENode("PE_" + str(i), self.hw_config.num_RFs_per_PE, self.hw_config.size_RF, self.hw_config.on_chip_bandwidth) for i in range(self.hw_config.num_PEs_per_lane)]
		
		# Memory Controller
		memory_input_ports = ["in_psums_" + str(i) for i in range(self.hw_config.num_PEs_per_lane)]
		memory_output_ports = [item for i in range(self.hw_config.num_PEs_per_lane) for item in ["out_A_" + str(i), "out_B_" + str(i), "out_psums_" + str(i)]]
		self.memory = MemoryControllerNode("MemoryController", self.hw_config.off_chip_bandwidth, self.hw_config.on_chip_bandwidth, memory_input_ports, memory_output_ports, share_load=self.share_load, num_PE_lanes=self.hw_config.num_PE_lanes, load_immediate=False, store_immediate=False, target_size=-1)
		
		# Connections
		connect(self.dram, "out", self.memory, "DRAM_in")
		connect(self.dram, "in", self.memory, "DRAM_out")
		for i in range(self.hw_config.num_PEs_per_lane):
			connect(self.memory, "in_psums_" + str(i), self.pes[i], "out")
			connect(self.memory, "out_A_" + str(i), self.pes[i], "in_A")
			connect(self.memory, "out_B_" + str(i), self.pes[i], "in_B")
			connect(self.memory, "out_psums_" + str(i), self.pes[i], "in_psums")
		
		# initialize tiling parameters
		self.num_tiles_x = math.ceil(self.act_cols / self.t_x)
		self.num_tiles_y = math.ceil(self.act_rows / self.t_y)
		self.num_tiles_ci = math.ceil(layer.c_in / self.t_ci)
		self.num_tiles_co = math.ceil(self.act_c_out / self.t_co)
		
		# per-tile statistics
		# TODO
		#self.ideal_cycles_per_tile = self.A_rows / self.tile_rows * self.A_cols_B_rows / self.tile_depth * self.B_cols / self.tile_cols / self.num_PEs
		self.last_tile_start = self.preload_cycles
		self.last_tile_ideal_input_bandwidth_used = 0
		self.last_tile_actual_input_bandwidth_used = 0
		self.last_tile_ideal_output_bandwidth_used = 0
		self.last_tile_actual_output_bandwidth_used = 0
		
		# initialize tiles
		# TODO
		self.tiles_assigned = 0
		self.output_tiles_created = set()
		self.mode = "processing"
		
		self.current_tile_row = 0
		self.current_tile_col = 0
		self.current_tile_c_in = 0
		self.current_tile_c_out = 0
		self.current_tile = self.generate_tile(0, 0, 0, 0)
		self.current_tile.setup_chunk_sets(self.memory)
		self.load_next_tile()
		
		self.finished_tiles = False
		self.num_tiles = math.ceil(self.act_rows / self.t_y) * math.ceil(self.act_cols / self.t_x) * math.ceil(self.layer.c_in / self.t_ci) * math.ceil(self.act_c_out / self.t_co)
		
		# initialize PE
		self.current_pe_assignments = {} # PE to tile_set
		if preload_cycles == 0:
			for idx, pe in enumerate(self.pes):
				self.assign_chunk(pe, 0)
		
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
				
				# TODO: cleanup end of tile stats
				print("Finished tile -- row:", self.current_tile_row, "col:", self.current_tile_col, "input channel:", self.current_tile_c_in, "output channel:", self.current_tile_c_out)
				print("Cycles:", current_cycle - self.last_tile_start)
				#ideal_cycles_curr_tile = self.current_tile.act_cols * self.current_tile.act_rows * self.current_tile.act_depth / self.num_PEs * (1 - self.sparsity)
				#print(ideal_cycles_curr_tile, self.ideal_cycles_per_tile)
				#print("PE Util:", ideal_cycles_curr_tile / (current_cycle - self.last_tile_start) * 100)
				#print("Input DRAM Bandwidth Util:", (self.last_tile_actual_input_bandwidth_used - self.memory.actual_input_bandwidth_used) / (self.last_tile_ideal_input_bandwidth_used - self.memory.ideal_input_bandwidth_used) * 100)
				#print("Output DRAM Bandwidth Util:", (self.last_tile_actual_output_bandwidth_used - self.memory.actual_output_bandwidth_used) / (self.last_tile_ideal_output_bandwidth_used - self.memory.ideal_output_bandwidth_used) * 100)
				self.last_tile_start = current_cycle
				self.last_tile_ideal_input_bandwidth_used = self.memory.ideal_input_bandwidth_used
				self.last_tile_actual_input_bandwidth_used = self.memory.actual_input_bandwidth_used
				self.last_tile_ideal_output_bandwidth_used = self.memory.ideal_output_bandwidth_used
				self.last_tile_actual_output_bandwidth_used = self.memory.actual_output_bandwidth_used
				
				# if it is finished, then the PE should still be idle
				assert pe.get_state() == "idle" or pe.next_state == "idle"
				
				# find the next tile
				res = self.get_next_coords()
				if res == None:
					self.finished_tiles = True
					return
				self.current_tile_row, self.current_tile_col, self.current_tile_c_in, self.current_tile_c_out = res
				
				# setup the new tile
				self.current_tile = self.generate_tile(self.current_tile_row, self.current_tile_col, self.current_tile_c_in, self.current_tile_c_out)
				self.current_tile.setup_chunk_sets(self.memory)
				self.tiles_assigned += 1
				
				if not self.finished_tiles:
					self.load_next_tile()
				
				# go ahead and assign a new chunk set to the PE
				self.assign_chunk(pe, current_cycle)
	
	def load_next_tile(self):
		# get coords of next tile & preload the data
		res = self.get_next_coords()
		if res != None:
			r, c, ci, co = res
			next_tile = self.generate_tile(r, c, ci, co, preloading=True)
			next_tile.setup_chunk_sets(self.memory, preload=True) # TODO update after making Tile class
	
	# returns 'None' on last one
	# inputs are by tile; e.g. row=2 means this is the third tile
	def get_next_coords(self, row=-1, col=-1, c_in=-1, c_out=-1):
		next_row = self.current_tile_row if row == -1 else row
		next_col = self.current_tile_col if col == -1 else col
		next_c_in = self.current_tile_c_in if c_in == -1 else c_in
		next_c_out = self.current_tile_c_out if c_out == -1 else c_out
		
		maxs = [self.num_tiles_y, self.num_tiles_x, self.num_tiles_ci, self.num_tiles_co]
		vals = [next_row, next_col, next_c_in, next_c_out]
		idxs = [0, 1, 2, 3]
		for i in range(4):
			if self.dataflow[i] == "y":
				idxs[i] = 0
			elif self.dataflow[i] == "x":
				idxs[i] = 1
			elif self.dataflow[i] == "ci":
				idxs[i] = 2
			elif self.dataflow[i] == "co":
				idxs[i] = 3
		
		#print(idxs)
		
		vals[idxs[0]] += 1
		if vals[idxs[0]] >= maxs[idxs[0]]:
			vals[idxs[0]] = 0
			vals[idxs[1]] += 1
			#print("incremented", idxs[1])
			if vals[idxs[1]] >= maxs[idxs[1]]:
				vals[idxs[1]] = 0
				#print("reset", idxs[1])
				vals[idxs[2]] += 1
				if vals[idxs[2]] >= maxs[idxs[2]]:
					vals[idxs[2]] = 0
					vals[idxs[3]] += 1
					if vals[idxs[3]] >= maxs[idxs[3]]:
						vals[idxs[3]] = 0
						# finished!
						return None
		
		return [vals[0], vals[1], vals[2], vals[3]]
	
	# returns 'None' if no previous tile
	# inputs are by tile; e.g. row=2 means this is the third tile
	def get_prev_coords(self, row=-1, col=-1, c_in=-1, c_out=-1):
		next_row = self.current_tile_row if row == -1 else row
		next_col = self.current_tile_col if col == -1 else col
		next_c_in = self.current_tile_c_in if c_in == -1 else c_in
		next_c_out = self.current_tile_c_out if c_out == -1 else c_out
		
		maxs = [self.num_tiles_y, self.num_tiles_x, self.num_tiles_ci, self.num_tiles_co]
		vals = [next_row, next_col, next_c_in, next_c_out]
		idxs = [0, 1, 2, 3]
		for i in range(4):
			if self.dataflow[i] == "y":
				idxs[i] = 0
			elif self.dataflow[i] == "x":
				idxs[i] = 1
			elif self.dataflow[i] == "ci":
				idxs[i] = 2
			elif self.dataflow[i] == "co":
				idxs[i] = 3
		
		vals[idxs[0]] -= 1
		if vals[idxs[0]] < 0:
			vals[idxs[0]] = maxs[idxs[0]] - 1
			vals[idxs[1]] -= 1
			if vals[idxs[1]] < 0:
				vals[idxs[1]] = maxs[idxs[1]] - 1
				vals[idxs[2]] -= 1
				if vals[idxs[2]] < 0:
					vals[idxs[2]] = maxs[idxs[2]] - 1
					vals[idxs[3]] -= 1
					if vals[idxs[3]] < 0:
						vals[idxs[3]] = maxs[idxs[3]] - 1
						# finished!
						return None
		
		return [vals[0], vals[1], vals[2], vals[3]]
	
	def generate_tile(self, r, c, ci, co, preloading=False):
		start_x = c * self.t_x
		start_y = r * self.t_y
		start_ci = ci * self.t_ci
		start_kx = 0
		start_ky = 0
		start_co = co * self.t_co
		
		size_x = self.t_x
		if start_x + size_x > self.act_cols:
			size_x = self.act_cols - start_x
		size_y = self.t_y
		if start_y + size_y > self.act_rows:
			size_y = self.act_rows - start_y
		size_ci = self.t_ci
		if start_ci + size_ci > self.layer.c_in:
			size_ci = self.layer.c_in - start_ci
		size_co = self.t_co
		if start_co + size_co > self.act_c_out:
			size_co = self.act_c_out - start_co
		
		#print("Creating tile")
		#print(start_x, start_y, start_ci, start_kx, start_ky, start_co, size_x, size_y, size_ci, size_co)
		
		o_tile_name = str(r) + ", " + str(c) + ", " + str(co)
		prev_coords = self.get_prev_coords(r, c, ci, co)
		if o_tile_name in self.output_tiles_created: # check if O is being created from scratch
			create_output_tile = False #not (r == prev_coords[0] and c == prev_coords[1] and co == prev_coords[3])
		else:
			create_output_tile = True
			if not preloading:
				self.output_tiles_created.add(o_tile_name)
		
		#create_output_tile = True
		#prev_coords = self.get_prev_coords(r, c, ci, co)
		#if prev_coords != None:
		#	if not (r == prev_coords[0] and c == prev_coords[1] and co == prev_coords[3]):
		#		create_output_tile = False
		
		A_clear_at_end = True
		B_clear_at_end = True
		save_output_tile = True
		next_coords = self.get_next_coords(r, c, ci, co)
		if next_coords != None:
			if r == next_coords[0] and c == next_coords[1] and ci == next_coords[2]:
				A_clear_at_end = False
			if ci == next_coords[2] and co == next_coords[3]:
				B_clear_at_end = False
			if r == next_coords[0] and c == next_coords[1] and co == next_coords[3]:
				save_output_tile = False
		
		out_tile = Tile(self.layer, self.lane_dataflow, self.act_rows, self.act_cols, start_x, start_y, start_ci, start_kx, start_ky, start_co, size_x, size_y, size_ci, self.layer.filter_dim, self.layer.filter_dim, size_co, self.chunk_dataflow, self.c_x, self.c_y, self.c_ci, self.c_kx, self.c_ky, self.c_co, A_clear_at_end, B_clear_at_end, create_output_tile, save_output_tile)
		
		return out_tile
	
	def get_nodes(self):
		return [self.dram, self.memory] + [pe for pe in self.pes]

	def finished(self):
		return self.finished_tiles and self.memory.get_flag("idle") == "True"

	def update(self, current_cycle):
		if current_cycle >= self.preload_cycles:
			# handle end of preload
			if current_cycle == self.preload_cycles:
				self.memory.finish_preload()
			
			# update PEs
			if not self.finished_tiles:
				for idx, pe in enumerate(self.pes):
					if pe.next_state == "idle":
						self.assign_chunk(pe, current_cycle)
			
			# collect PE statistics
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
		
		# collect memory statistics
		if self.memory.ports["DRAM_in"].is_current_request() or self.memory.ports["DRAM_out"].is_current_request():
			self.memory_active_cycles += 1
		else:
			self.mem_idle_cycles += 1
			if self.mem_idle_cycles >= 100 and current_cycle < self.preload_cycles:
				print("Stopped preloading at", current_cycle)
				self.memory.finish_preload()
				self.mem_idle_cycles += self.preload_cycles - current_cycle
				self.preload_cycles = current_cycle

	# TODO
	def print_data(self, current_cycle):
		print("###############", current_cycle, "###############")
		print("Current tile:", self.current_tile_row, self.current_tile_col, self.current_tile_depth, "out of", self.tile_rows, self.tile_cols, self.tile_depth)
		print("Chunks covered:", self.current_tile_row * self.chunk_rows, self.current_tile_col * self.chunk_cols, self.current_tile_depth * self.chunk_depth, "-", (self.current_tile_row + 1) * self.chunk_rows - 1, (self.current_tile_col + 1) * self.chunk_cols - 1, (self.current_tile_depth + 1) * self.chunk_depth - 1)
		for pe in self.pes:
			pe.print_status()
			self.current_tile.print_pe_data(pe)
		self.memory.print_status()
	
	# TODO
	def get_percent_complete(self):
		return (self.tiles_assigned) / self.num_tiles + (1 / self.num_tiles) * self.current_tile.get_percent_complete()
	
	# TODO
	def on_complete(self, final_cycles):
		print("Cycles spent offloading while PEs are idle:", final_cycles - self.last_pe_active_cycle)
		if self.pipeline_offloading:
			self.pe_idle_at_end_cycles = final_cycles - self.last_pe_active_cycle
			self.total_cycles = self.last_pe_active_cycle - self.preload_cycles
		else:
			self.total_cycles = final_cycles - self.preload_cycles
		self.total_dram_accesses = self.dram.reads + self.dram.writes #self.memory.dram_reads + self.memory.dram_writes
		#self.print_on_complete(self.total_cycles)
		total_comps = 0
		total_chunks = 0
		for pe in self.pes:
			total_comps += pe.num_computations
			total_chunks += pe.num_chunks
		print("Total number of computations:", total_comps, total_comps*self.hw_config.num_PE_lanes)
		print("Total number of chunks computed:", total_chunks)
		print("Final cycle count:", final_cycles)
		ideal_cycles = self.layer.get_flops() / (self.hw_config.num_PEs_per_lane * self.hw_config.num_PE_lanes)
		print("Ideal cycle count:", ideal_cycles)
		print("Utilization:", ideal_cycles / final_cycles * 100, "%")
		print("Memory active cycles:", self.memory_active_cycles)
		print("Memory-logged DRAM reads/writes:", self.memory.dram_reads + self.memory.dram_writes)
		print("Memory-logged DRAM reads:", self.memory.dram_reads)
		print("Memory-logged DRAM writes:", self.memory.dram_writes)
		print("DRAM reads/writes:", self.dram.reads + self.dram.writes)
		print("DRAM reads:", self.dram.reads)
		print("DRAM writes:", self.dram.writes)
	
	# TODO
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
	
	# TODO
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
	
	# TODO
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
	
	# TODO
	def get_results(self):
		return [self.total_cycles, self.total_dram_accesses, self.pe_active_cycles, self.memory_active_cycles, self.mem_idle_cycles, self.pe_idle_at_end_cycles, self.memory.get_offload_size()]

# Whole thing is TODO
class Tile:
	
	# inputs: layer, start position (actual position, not by # chunks), total size (actual size, not by # chunks), individual chunk size
	# this assumes step_size == 1
	def __init__(self, layer, lane_dataflow, act_rows, act_cols, start_x, start_y, start_ci, start_kx, start_ky, start_co, size_x, size_y, size_ci, size_kx, size_ky, size_co, dataflow, c_x, c_y, c_ci, c_kx, c_ky, c_co, A_clear_at_end, B_clear_at_end, create_output_tile, save_output_tile):
		
		# initialization
		self.layer = layer
		self.lane_dataflow = lane_dataflow
		self.act_rows = act_rows
		self.act_cols = act_cols
		self.start_x = start_x
		self.start_y = start_y
		self.start_ci = start_ci
		self.start_kx = start_kx
		self.start_ky = start_ky
		self.start_co = start_co
		self.size_x = size_x
		self.size_y = size_y
		self.size_ci = size_ci
		self.size_kx = size_kx
		self.size_ky = size_ky
		self.size_co = size_co
		self.c_x = c_x
		self.c_y = c_y
		self.c_ci = c_ci
		self.c_kx = c_kx
		self.c_ky = c_ky
		self.c_co = c_co
		self.A_clear_at_end = A_clear_at_end
		self.B_clear_at_end = B_clear_at_end
		self.create_output_tile = create_output_tile
		self.save_output_tile = save_output_tile
		self.dataflow = dataflow
		
		#print(self.start_x, self.start_y, self.start_ci, self.start_co, self.A_clear_at_end, self.B_clear_at_end, self.save_output_tile, self.create_output_tile)
		
		self.chunk_sets = []
		
		self.pe_mapping = {} # PE name -> ChunkSet
		self.A_chunk_list = []
		self.B_chunk_list = []
		self.Out_chunk_list = set()
		
		# finished flags
		self.num_finished_chunks = 0
		self.num_chunks_total = -1
	
	def find_pe_instrs(self, c_x=-1, c_y=-1, c_ci=-1, c_kx=-1, c_ky=-1, c_co=-1):
		
		# check if input chunk size is a vector or matrix
		num_greater = 0
		if c_x > 1:
			num_greater += 1
		if c_y > 1:
			num_greater += 1
		if c_ci > 1:
			num_greater += 1
		if num_greater > 1:
			input_vector = False
		else:
			input_vector = True
		#print(input_vector)
		if input_vector:
			# find a and w from the input
			a = max(c_x, c_y)
			w = c_ci #max(c_x, c_y, c_ci)
			
			# find b from the weight chunk params
			res = sorted([c_ci, c_kx, c_ky, c_co], reverse=True)
			if res[0] == w:
				b = res[1]
			else:
				b = res[0]
		else:
			# find b and w from the weights
			b = max(c_kx, c_ky, c_co)
			w = c_ci
			
			# find a from the input chunk params
			res = sorted([c_x, c_y, c_ci], reverse=True)
			
			if res[0] == w:
				a = res[1]
			else:
				a = res[0]
		
		return [a, b, w]
	
	def setup_chunk_sets(self, memory, preload=False):
		
		# generate ChunkSets in the order specified by the dataflow
		starts = [self.start_x, self.start_y, self.start_ci, self.start_kx, self.start_ky, self.start_co]
		ends = [self.start_x + self.size_x, self.start_y + self.size_y, self.start_ci + self.size_ci, self.start_kx + self.size_kx, self.start_ky + self.size_ky, self.start_co + self.size_co]
		steps = [self.c_x, self.c_y, self.c_ci, self.c_kx, self.c_ky, self.c_co]
		idxs = [0, 1, 2, 3, 4, 5]
		for i in range(6):
			if self.dataflow[i] == "x":
				idxs[i] = 0
			elif self.dataflow[i] == "y":
				idxs[i] = 1
			elif self.dataflow[i] == "kx":
				idxs[i] = 2
			elif self.dataflow[i] == "ky":
				idxs[i] = 3
			elif self.dataflow[i] == "ci":
				idxs[i] = 4
			elif self.dataflow[i] == "co":
				idxs[i] = 5
		
		#print(starts)
		#print(ends)
		#print(steps)
		
		for i0 in range(starts[idxs[0]], ends[idxs[0]], steps[idxs[0]]):
			for i1 in range(starts[idxs[1]], ends[idxs[1]], steps[idxs[1]]):
				for i2 in range(starts[idxs[2]], ends[idxs[2]], steps[idxs[2]]):
					for i3 in range(starts[idxs[3]], ends[idxs[3]], steps[idxs[3]]):
						for i4 in range(starts[idxs[4]], ends[idxs[4]], steps[idxs[4]]):
							for i5 in range(starts[idxs[5]], ends[idxs[5]], steps[idxs[5]]):
								x_index = idxs.index(0)
								if x_index == 0:
									x = i0
								elif x_index == 1:
									x = i1
								elif x_index == 2:
									x = i2
								elif x_index == 3:
									x = i3
								elif x_index == 4:
									x = i4
								elif x_index == 5:
									x = i5
								
								y_index = idxs.index(1)
								if y_index == 0:
									y = i0
								elif y_index == 1:
									y = i1
								elif y_index == 2:
									y = i2
								elif y_index == 3:
									y = i3
								elif y_index == 4:
									y = i4
								elif y_index == 5:
									y = i5
								
								ci_index = idxs.index(2)
								if ci_index == 0:
									ci = i0
								elif ci_index == 1:
									ci = i1
								elif ci_index == 2:
									ci = i2
								elif ci_index == 3:
									ci = i3
								elif ci_index == 4:
									ci = i4
								elif ci_index == 5:
									ci = i5
								
								kx_index = idxs.index(3)
								if kx_index == 0:
									kx = i0
								elif kx_index == 1:
									kx = i1
								elif kx_index == 2:
									kx = i2
								elif kx_index == 3:
									kx = i3
								elif kx_index == 4:
									kx = i4
								elif kx_index == 5:
									kx = i5
								
								ky_index = idxs.index(4)
								if ky_index == 0:
									ky = i0
								elif ky_index == 1:
									ky = i1
								elif ky_index == 2:
									ky = i2
								elif ky_index == 3:
									ky = i3
								elif ky_index == 4:
									ky = i4
								elif ky_index == 5:
									ky = i5
								
								co_index = idxs.index(5)
								if co_index == 0:
									co = i0
								elif co_index == 1:
									co = i1
								elif co_index == 2:
									co = i2
								elif co_index == 3:
									co = i3
								elif co_index == 4:
									co = i4
								elif co_index == 5:
									co = i5
								
								chunk_name_A = "A_" + str(x) + "_" + str(y) + "_" + str(ci)
								chunk_name_B = "B_" + str(ci) + "_" + str(kx) + "_" + str(ky) + "_" + str(co)
								chunk_name_O = "O_" + str(x) + "_" + str(y) + "_" + str(co)
								self.A_chunk_list.append(chunk_name_A)
								self.B_chunk_list.append(chunk_name_B)
								self.Out_chunk_list.add(chunk_name_O)
								
								lx = x
								rx = x + self.c_x
								if rx > self.start_x + self.size_x:
									rx = self.start_x + self.size_x
								ly = y
								ry = y + self.c_y
								if ry > self.start_y + self.size_y:
									ry = self.start_y + self.size_y
								
								# check leftmost x
								if kx > lx:
									lx = kx
								# check rightmost x
								if rx > self.layer.pcols - self.layer.filter_dim + kx + 1:
									#print("rx", rx, self.act_cols - self.layer.filter_dim + kx + 1)
									rx = self.layer.pcols - self.layer.filter_dim + kx + 1
								
								# check leftmost y
								if ky > ly:
									ly = ky
								# check rightmost y
								if ry > self.layer.prows - self.layer.filter_dim + ky + 1:
									#print("ry", ry, self.act_rows - self.layer.filter_dim + ky + 1)
									ry = self.layer.prows - self.layer.filter_dim + ky + 1
								
								# check for kx and ky
								c_kx = self.c_kx
								c_ky = self.c_ky
								if c_kx + kx > self.start_kx + self.size_kx:
									c_kx = self.start_kx + self.size_kx - kx
								if c_ky + ky > self.start_ky + self.size_ky:
									c_ky = self.start_ky + self.size_ky - ky
								
								# check for ci and co
								c_ci = self.c_ci
								c_co = self.c_co
								if c_ci + ci > self.start_ci + self.size_ci:
									c_ci = self.start_ci + self.size_ci - ci
								if c_co + co > self.start_co + self.size_co:
									c_co = self.start_co + self.size_co - co
								
								# if this needs to be skipped, then skip it!
								if rx - lx <= 0 or ry - ly <= 0:
									#print("here", rx, lx, ry, ly)
									continue
								
								# generate a, b, and w
								a, b, w = self.find_pe_instrs(rx-lx, ry-ly, c_ci, c_kx, c_ky, c_co)
								
								# add chunks to memory
								memory.add_chunk(chunk_name_A, max(0, a*w), source_is_dram=True)
								memory.add_chunk(chunk_name_B, max(0, b*w), source_is_dram=True)
								if self.create_output_tile:
									memory.add_chunk(chunk_name_O, max(0, a*b), source_is_dram=False)
								else:
									memory.add_chunk(chunk_name_O, max(0, a*b), source_is_dram=True)
									memory.update_chunk(chunk_name_O, source_is_dram=True)
								
								# create the chunk set; [base case: the current chunk from A and B are multiplied together to make O (assuming each chunk is the same size)]
								cs = ChunkSet(chunk_name_A, chunk_name_B, chunk_name_O, a, b, w)
								self.chunk_sets.append(cs)
		
		self.num_chunks_total = len(self.chunk_sets)
		#print("finished creating chunks:", self.num_chunks_total)
		
		#a_size = 0
		#b_size = 0
		#o_size = 0
		#a_names = set()
		#b_names = set()
		#o_names = set()
		#for chunk in self.chunk_sets:
		#	if not chunk.chunk_name_A in a_names:
		#		a_size += chunk.a*chunk.w
		#		a_names.add(chunk.chunk_name_A)
		#	if not chunk.chunk_name_B in b_names:
		#		b_size += chunk.b*chunk.w
		#		b_names.add(chunk.chunk_name_B)
		#	if not chunk.chunk_name_Out in o_names:
		#		o_size += chunk.a*chunk.b
		#		o_names.add(chunk.chunk_name_Out)
		#print("A:", a_size, "B:", b_size, "O:", o_size)
	
	# erase A, B, and Out chunks
	def erase_chunks(self, memory, erase_A, erase_B, erase_Out):
		if erase_A:
			for chunk in self.A_chunk_list:
				memory.erase_chunk(chunk)
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
		if len(self.chunk_sets) > 0:
			res = self.chunk_sets[0]
			# assign this chunk
			self.pe_mapping[pe.name] = res
			memory.set_chunk_for_port(pe.ports["in_A"].connection.source_port.name, res.chunk_name_A)
			memory.set_chunk_for_port(pe.ports["in_B"].connection.source_port.name, res.chunk_name_B)
			memory.set_chunk_for_port(pe.ports["out"].connection.sink_port.name, res.chunk_name_Out)
			memory.set_chunk_for_port(pe.ports["in_psums"].connection.source_port.name, res.chunk_name_Out)
			pe.set_operation(res.a, res.w, res.b, False, False, True, False, False, sparsity = 0, sparse_map=None) # changed behavior so that psums aren't loaded onto PEs anymore
			
			# remove the old chunk set
			self.chunk_sets.remove(res)

	def finished(self, memory):
		done = (self.num_finished_chunks == self.num_chunks_total)
		if done:
			self.erase_chunks(memory, self.A_clear_at_end, self.B_clear_at_end, False)
			if self.save_output_tile:
				for chunk in self.Out_chunk_list:
					memory.give_save_chunk_request(chunk)
		return done
	
	def get_percent_complete(self):
		if self.num_chunks_total == 0:
			return 1
		#print(self.num_finished_chunks)
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
	def __init__(self, chunk_name_A, chunk_name_B, chunk_name_Out, a, b, w):
		self.chunk_name_A = chunk_name_A
		self.chunk_name_B = chunk_name_B
		self.chunk_name_Out = chunk_name_Out
		self.a = a
		self.b = b
		self.w = w
	
	def print(self):
		print("A Chunk:", self.chunk_name_A, "B Chunk:", self.chunk_name_B, "Out Chunk:", self.chunk_name_Out)

# returns total_cycles, utilization, a buffer min size, reads, writes, b buffer min size, reads, writes, o buffer min size, reads, writes, dram accesses, reads, writes
# PE_config = 0 means PE_width and PE_height are given; PE_config = 1 means PE_width and PE_height are switched
def run_conv_dataflow(hw_config, layer, params):
	
	state_machine = ConvDataflow(hw_config, layer, params)
	
	run_system(state_machine, state_machine.get_nodes(), 100000000, False)
	
	return state_machine.get_results()

if __name__ == "__main__":
	pass