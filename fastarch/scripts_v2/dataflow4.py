import numpy as np
import math

from src.types import Node, connect, StateMachine, Port
from src.buffer_node import BufferNode
from src.dram_node import DRAMNode
from src.complex_PE_node import ComplexPENode
from src.memory_controller_node_v2 import MemoryControllerNode
from src.controller import run_system

# TODO
# (optimization): Start loading data for the next tile once the current tile is finished (remove "idle" cycles)
# (optimization): If the tiling parameters don't exactly fit the input matrices, then reduce the size of the final tiles/chunks to exactly fit the input matrices
#	- then, parameter sweep may no longer have to optimize for an exact fit, and can instead optimize for inter-PE utilization and intra-PE utilization (e.g. making sure the % of PEs active is high, and making sure the % of time a PE spends computing vs loading or flushing is high)
# (optimization???): Vary the # of multipliers inside a PE

# Does matrix-matrix multiplication using multiple PEs
# dataflow is either "output-matrix" or "output-vector"
# PE can consume 1 chunk of A and 1 chunk of B to produce 1 chunk of O
# Chip can consume 1 tile of A and 1 tile of B to produce 1 partial tile of O

# outer dataflow (e.g. how tiles are loaded/offloaded -- A-Stationary, B-Stationary, or Output-Stationary)
# inner dataflow (e.g. how chunks are loaded/offloaded in a single tile -- A-Stationary, B-Stationary, Output-Stationary)
# PE dataflow (e.g. how chunks are computed inside a PE -- A-Stationary, B-Stationary, or Output-Stationary) --> this one probably is unnecessary
class MMDataflow(StateMachine):

	def __init__(self, num_PEs, num_banks_per_PE, size_banks_per_PE, off_chip_bandwidth, on_chip_bandwidth, A_rows, A_cols_B_rows, B_cols, dataflow, t_a, t_b, t_w, c_a, c_b, c_w, estimate):
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
		self.inner_dataflow = "Output-Stationary"
		self.estimate = estimate
		
		# setting up nodes
		
		# DRAM
		self.dram = DRAMNode("DRAM", self.off_chip_bandwidth, initialization_time=0)
		
		# PE
		self.pes = [ComplexPENode("PE_" + str(i), self.num_banks_per_PE, self.size_banks_per_PE, self.on_chip_bandwidth) for i in range(self.num_PEs)]
		
		# Memory Controller
		memory_input_ports = ["in_psums_" + str(i) for i in range(self.num_PEs)]
		memory_output_ports = [item for i in range(self.num_PEs) for item in ["out_A_" + str(i), "out_B_" + str(i), "out_psums_" + str(i)]]
		self.memory = MemoryControllerNode("MemoryController", self.off_chip_bandwidth, self.on_chip_bandwidth, memory_input_ports, memory_output_ports)
		
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
		
		self.tile_rows = math.ceil(A_rows / t_a)
		self.tile_cols = math.ceil(B_cols / t_b)
		self.tile_depth = math.ceil(A_cols_B_rows / t_w)
		self.chunk_rows = math.ceil(t_a / c_a)
		self.chunk_cols = math.ceil(t_b / c_b)
		self.chunk_depth = math.ceil(t_w / c_w)
		
		#print(self.tile_rows, self.tile_cols, self.tile_depth)
		
		# initialize tile_sets & memory controller
		'''
		self.tiles = [[[None for k in range(self.tile_depth)] for j in range(self.tile_cols)] for i in range(self.tile_rows)]
		for i in range(self.tile_rows):
			for j in range(self.tile_cols):
				for k in range(self.tile_depth):
					if self.dataflow == "Output-Stationary":
						O_load_option = 2
						if k == 0:
							O_load_option = 0
						O_save_to_dram = False
						if k == self.tile_depth - 1:
							O_save_to_dram = True
						self.tiles[i][j][k] = Tile(self.memory, i * self.chunk_rows, j * self.chunk_cols, k * self.chunk_depth, self.chunk_rows, self.chunk_cols, self.chunk_depth, self.x, self.z, self.y, A_clear_at_end=True, B_clear_at_end=True, O_load_option=O_load_option, O_save_to_dram=O_save_to_dram, dataflow=self.inner_dataflow)
					elif self.dataflow == "A-Stationary":
						A_clear_at_end = False
						if j == self.tile_cols - 1:
							A_clear_at_end = True
						B_clear_at_end = True
						#if self.tile_cols == 1:
						#	B_clear_at_end = False
						O_load_option = 1
						if k == 0:
							O_load_option = 0
						O_save_to_dram = True
						self.tiles[i][j][k] = Tile(self.memory, i * self.chunk_rows, j * self.chunk_cols, k * self.chunk_depth, self.chunk_rows, self.chunk_cols, self.chunk_depth, self.x, self.z, self.y, A_clear_at_end=A_clear_at_end, B_clear_at_end=B_clear_at_end, O_load_option=O_load_option, O_save_to_dram=O_save_to_dram, dataflow=self.inner_dataflow)
					elif self.dataflow == "B-Stationary":
						A_clear_at_end = True
						#if i == self.tile_rows - 1:
							#A_clear_at_end = True
						B_clear_at_end = False
						if i == self.tile_rows - 1:
							B_clear_at_end = True
						#if self.tile_cols == 1:
						#	B_clear_at_end = False
						O_load_option = 1
						if k == 0:
							O_load_option = 0
						O_save_to_dram = True
						self.tiles[i][j][k] = Tile(self.memory, i * self.chunk_rows, j * self.chunk_cols, k * self.chunk_depth, self.chunk_rows, self.chunk_cols, self.chunk_depth, self.x, self.z, self.y, A_clear_at_end=A_clear_at_end, B_clear_at_end=B_clear_at_end, O_load_option=O_load_option, O_save_to_dram=O_save_to_dram, dataflow=self.inner_dataflow)
					else:
						raise Exception(self.dataflow + " dataflow is not supported")
		'''
		self.current_tile = self.generate_tile(0, 0, 0) #self.tiles[0][0][0]
		self.current_tile.setup_chunk_sets(self.memory)
		self.current_tile_row = 0
		self.current_tile_col = 0
		self.current_tile_depth = 0
		
		self.finished_tiles = False
		self.tiles_assigned = 0
		
		# initialize PE
		self.current_pe_assignments = {} # PE to tile_set
		for pe in self.pes:
			self.assign_chunk(pe, 0)
		
		self.total_cycles = -1
		self.last_tile_start = 0
		
		self.idle_cycles = 0 # idle because not assigned a task
		self.waiting_cycles = 0 # idle because waiting for data
		self.loading_cycles = 0
		self.flushing_cycles = 0
		self.computing_cycles = 0
		
	
	def assign_chunk(self, pe, current_cycle):
		if not self.finished_tiles:
			# give the PE an assignment
			self.current_tile.get_assignment(self.memory, pe)
			
			# check if the current tile is finished
			if self.current_tile.finished(self.memory):
				#print((current_cycle - self.last_tile_start), self.chunk_depth * self.x * self.chunk_cols * self.z * self.chunk_rows * self.y / self.num_PEs)
				self.last_tile_start = current_cycle
				#print(">", end='', flush=True)
				
				# if it is finished, then the PE should still be idle
				assert pe.get_state() == "idle"
				# assign the next tile
				if self.dataflow == "Output-Stationary":
					# clear old A and B data
					#self.current_tile.erase_chunks(self.memory, True, True, False)
					
					self.current_tile_depth += 1
					if self.current_tile_depth == self.tile_depth:
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
				if self.estimate:
					self.finished_tiles = True
					return
				
				# setup the new tile
				self.current_tile = self.generate_tile(self.current_tile_row, self.current_tile_col, self.current_tile_depth) #self.tiles[self.current_tile_row][self.current_tile_col][self.current_tile_depth]
				self.current_tile.setup_chunk_sets(self.memory)
				self.tiles_assigned += 1
				# go ahead and assign a new chunk set to the PE
				self.assign_chunk(pe, current_cycle)
	
	def generate_tile(self, i, j, k):
		out_tile = None
		if self.dataflow == "Output-Stationary":
			O_load_option = 2
			if k == 0:
				O_load_option = 0
			O_save_to_dram = False
			if k == self.tile_depth - 1:
				O_save_to_dram = True
			out_tile = Tile(self.memory, i * self.chunk_rows, j * self.chunk_cols, k * self.chunk_depth, self.chunk_rows, self.chunk_cols, self.chunk_depth, self.x, self.z, self.y, A_clear_at_end=True, B_clear_at_end=True, O_load_option=O_load_option, O_save_to_dram=O_save_to_dram, dataflow=self.inner_dataflow)
		elif self.dataflow == "A-Stationary":
			A_clear_at_end = False
			if j == self.tile_cols - 1:
				A_clear_at_end = True
			B_clear_at_end = True
			#if self.tile_cols == 1:
			#	B_clear_at_end = False
			O_load_option = 1
			if k == 0:
				O_load_option = 0
			O_save_to_dram = True
			out_tile = Tile(self.memory, i * self.chunk_rows, j * self.chunk_cols, k * self.chunk_depth, self.chunk_rows, self.chunk_cols, self.chunk_depth, self.x, self.z, self.y, A_clear_at_end=A_clear_at_end, B_clear_at_end=B_clear_at_end, O_load_option=O_load_option, O_save_to_dram=O_save_to_dram, dataflow=self.inner_dataflow)
		elif self.dataflow == "B-Stationary":
			A_clear_at_end = True
			#if i == self.tile_rows - 1:
				#A_clear_at_end = True
			B_clear_at_end = False
			if i == self.tile_rows - 1:
				B_clear_at_end = True
			#if self.tile_cols == 1:
			#	B_clear_at_end = False
			O_load_option = 1
			if k == 0:
				O_load_option = 0
			O_save_to_dram = True
			out_tile = Tile(self.memory, i * self.chunk_rows, j * self.chunk_cols, k * self.chunk_depth, self.chunk_rows, self.chunk_cols, self.chunk_depth, self.x, self.z, self.y, A_clear_at_end=A_clear_at_end, B_clear_at_end=B_clear_at_end, O_load_option=O_load_option, O_save_to_dram=O_save_to_dram, dataflow=self.inner_dataflow)
		else:
			raise Exception(self.dataflow + " dataflow is not supported")
		
		return out_tile
	
	def get_nodes(self):
		return [self.dram, self.memory] + [pe for pe in self.pes]

	def finished(self):
		if self.estimate:
			pes_idle = True
			for pe in self.pes:
				if pe.get_state() != "idle":
					pes_idle = False
					break
			return self.finished_tiles and pes_idle
		return self.finished_tiles and self.memory.get_flag("idle") == "True"

	def update(self, current_cycle):
		for pe in self.pes:
			if pe.get_state() == "idle":
				self.assign_chunk(pe, current_cycle)
		
		#self.print_data(current_cycle)
		
		for pe in self.pes:
			if pe.get_state() == "idle":
				if self.current_tile.is_pe_assigned(pe):
					self.waiting_cycles += 1
				else:
					self.idle_cycles += 1
			elif pe.get_state() == "loading":
				self.loading_cycles += 1
			elif pe.get_state() == "flushing":
				self.flushing_cycles += 1
			elif pe.get_state() == "computing":
				self.computing_cycles += 1

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
			return self.current_tile.get_percent_complete()
		else:
			num_total = self.tile_depth * self.tile_rows * self.tile_cols
			return (self.tiles_assigned) / num_total + (1 / num_total) * self.current_tile.get_percent_complete()
	
	def on_complete(self, final_cycles):
		if self.estimate:
			self.total_cycles = final_cycles * self.tile_depth * self.tile_rows * self.tile_cols
		else:
			self.total_cycles = final_cycles
		#self.print_on_complete(self.total_cycles)
	
	def print_stats(self):
		print("Chunk dim:", self.x, self.z, self.y)
		print("Chunks per tile dims:", self.chunk_rows, self.chunk_cols, self.chunk_depth)
		print("Matrix dims:", self.tile_rows, self.tile_cols, self.tile_depth)
		print("Effective size:", self.tile_rows * self.chunk_rows * self.x, self.tile_cols * self.chunk_cols * self.z, self.tile_depth * self.chunk_depth * self.y)
		
		print("Idle cycles:", self.idle_cycles)
		print("Waiting for data cycles:", self.idle_cycles)
		print("Loading cycles:", self.loading_cycles)
		print("Flushing cycles:", self.flushing_cycles)
		print("Computing cycles:", self.computing_cycles)
	
	def print_memory_usage(self):
		print("DRAM reads/writes:", self.dram.reads + self.dram.writes)
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
	
		ideal_cycles = self.A_rows * self.A_cols_B_rows * self.B_cols / self.num_PEs
		print("Final cycle count:", final_cycles)
		print("Ideal cycle count:", ideal_cycles)
		print("Utilization:", ideal_cycles / final_cycles * 100, "%")
		print("Global Buffer Size:", self.memory.get_min_buffer_size())
		print("Global Buffer Banks:", self.memory.get_min_buffer_banks())
		print("Total PE Memory:", self.num_PEs * self.num_banks_per_PE * self.size_banks_per_PE)
		self.print_memory_usage()
	
	def get_results(self):
		return self.total_cycles

class Tile:
	# index_A, index_B, index_W -> coordinates of tile
	# length, width, depth -> dimensions of tile
	# chunk_rows, chunk_cols, chunk_width -> size of chunk
	# O_load_option -> 0 = create new, 1 = load from DRAM, 2 = present in buffer
	# dataflow -> inner dataflow (e.g. how this tile handles computations inside) - Output-Stationary
	def __init__(self, memory, index_A, index_B, index_W, length, width, depth, chunk_rows, chunk_cols, chunk_width, A_clear_at_end, B_clear_at_end, O_load_option, O_save_to_dram, dataflow):
		self.index_A = index_A
		self.index_B = index_B
		self.index_W = index_W
		self.length = length
		self.width = width
		self.depth = depth
		self.chunk_rows = chunk_rows
		self.chunk_cols = chunk_cols
		self.chunk_width = chunk_width
		self.A_clear_at_end = A_clear_at_end
		self.B_clear_at_end = B_clear_at_end
		self.O_load_option = O_load_option
		self.O_save_to_dram = O_save_to_dram
		self.dataflow = dataflow
		
		self.chunk_sets = []
		
		self.pe_mapping = {} # PE name -> ChunkSet
		self.A_chunk_list = []
		self.B_chunk_list = []
		self.Out_chunk_list = []
		
		# finished flags
		self.num_finished_chunks = 0
		self.num_chunks_total = self.length * self.width * self.depth
	
	def setup_chunk_sets(self, memory):
		#print(self.length, self.width, self.depth, self.num_chunks_total)
		# generate ChunkSets
		for i in range(self.index_A, self.index_A + self.length):
			for j in range(self.index_B, self.index_B + self.width):
				for k in range(self.index_W, self.index_W + self.depth):
					chunk_name_A = "A_" + str(i) + "_" + str(k)
					chunk_name_B = "B_" + str(k) + "_" + str(j)
					chunk_name_O = "O_" + str(i) + "_" + str(j)
					self.A_chunk_list.append(chunk_name_A)
					self.B_chunk_list.append(chunk_name_B)
					self.Out_chunk_list.append(chunk_name_O)
					self.chunk_sets.append(ChunkSet(chunk_name_A, chunk_name_B, chunk_name_O, i - self.index_A, j - self.index_B, k - self.index_W))
					if self.dataflow == "Output-Stationary":
						memory.add_chunk(chunk_name_A, self.chunk_rows * self.chunk_width, source_is_dram=True)
						memory.update_chunk(chunk_name_A, source_is_dram=True)
						memory.add_chunk(chunk_name_B, self.chunk_cols * self.chunk_width, source_is_dram=True)
						memory.update_chunk(chunk_name_B, source_is_dram=True)
						if k == self.index_W:
							if self.O_load_option == 1:
								memory.add_chunk(chunk_name_O, self.chunk_rows * self.chunk_cols, source_is_dram=True, remove_on_read=True)
								memory.update_chunk(chunk_name_O, source_is_dram=True, remove_on_read=True)
							else:
								memory.add_chunk(chunk_name_O, self.chunk_rows * self.chunk_cols, source_is_dram=False, remove_on_read=True)
								memory.update_chunk(chunk_name_O, source_is_dram=False, remove_on_read=True)
					else:
						raise ValueError(dataflow + " dataflow not supported (inner dataflow)")
	
	# erase A and B chunks
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
					pe.set_operation(self.chunk_rows, self.chunk_width, self.chunk_cols, False, True, num_available == 1, False, False)
					self.chunk_sets.remove(chosen_chunk_set)
					return
			
			# if we get here, we know that the PE needs to start a new output chunk
			
			# first, dispose of the old output chunk (if it exists)
			if old_chunk_set != None and self.O_save_to_dram == True:
				memory.give_save_chunk_request(old_chunk_set.chunk_name_Out)
			
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
				#print("out port name:", pe.ports["in_psums"].connection.source_port.name)
				psum_load = self.O_load_option != 0 and chosen_chunk_set.depth == 0
				
				# get number of chunk_sets with the same output chunk; if there's only one available, then set psum_flush to true
				num_available = 0
				for chunk_set in self.chunk_sets:
					if chunk_set.chunk_name_Out == chosen_chunk_set.chunk_name_Out:
						num_available += 1
				psum_flush = num_available == 1
				pe.set_operation(self.chunk_rows, self.chunk_width, self.chunk_cols, psum_load, False, psum_flush, False, False)
				self.chunk_sets.remove(chosen_chunk_set)
			else:
				# if there's no new output chunk to start, unmap this PE
				self.pe_mapping[pe.name] = None
		else:
			raise ValueError(dataflow + " dataflow not supported (inner dataflow)")
	
	def finished(self, memory):
		done = self.num_finished_chunks == self.num_chunks_total
		if done:
			self.erase_chunks(memory, self.A_clear_at_end, self.B_clear_at_end, False)
		return done
	
	def get_percent_complete(self):
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
	def __init__(self, chunk_name_A, chunk_name_B, chunk_name_Out, row, column, depth):
		self.chunk_name_A = chunk_name_A
		self.chunk_name_B = chunk_name_B
		self.chunk_name_Out = chunk_name_Out
		self.row = row
		self.column = column
		self.depth = depth
	
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
def run_MM_dataflow(num_PEs, num_RFs_per_PE, size_RF, off_chip_bandwidth, on_chip_bandwidth, max_sram_size, A_rows, A_cols_B_rows, B_cols, dataflow, t_a, t_b, t_w, c_a, c_b, c_w, estimate=False):
	
	state_machine = MMDataflow(num_PEs = num_PEs, num_banks_per_PE = num_RFs_per_PE, size_banks_per_PE = size_RF, off_chip_bandwidth = off_chip_bandwidth, on_chip_bandwidth = on_chip_bandwidth, A_rows = A_rows, A_cols_B_rows = A_cols_B_rows, B_cols = B_cols, dataflow = dataflow, t_a = t_a, t_b = t_b, t_w = t_w, c_a = c_a, c_b = c_b, c_w = c_w, estimate=estimate)
	
	#state_machine = MMDataflow(A_buffer_size = params[0], B_buffer_size = params[1], O_buffer_size = params[2], num_PEs = num_PEs, num_banks_per_PE = 10, size_banks_per_PE = 10, off_chip_bandwidth = 100, on_chip_bandwidth = 10, A_rows = A_rows, A_cols_B_rows = A_cols_B_rows, B_cols = B_cols, dataflow = "Output-Stationary", x = params[3], y = params[5], z = params[4])

	#state_machine = MMDataflow(A_buffer_size = params[0], B_buffer_size = params[1], O_buffer_size = params[2], num_PEs = num_PEs, num_banks_per_PE = 10, size_banks_per_PE = 10, off_chip_bandwidth = 100, on_chip_bandwidth = 10, A_rows = A_rows, A_cols_B_rows = A_cols_B_rows, B_cols = B_cols, dataflow = "B-Stationary", t_a = params[0], t_b = params[1], t_w = params[2], c_a = params[3], c_b = params[4], c_w = params[5])

	run_system(state_machine, state_machine.get_nodes(), 100000000, False)
	
	return state_machine.get_results()

#A_rows = 3025
#A_cols_B_rows = 363
#B_cols = 96
#A_rows = 1
#A_cols_B_rows = 1000
#B_cols = 1000
#num_PEs = 50

#params = optimize_params_2(num_PEs, 10000, 10, 10, A_rows, B_cols, A_cols_B_rows)
#params = [1, 100, 100, 1, 5, 10]
#for num_PEs in [1, 5, 10, 50, 100]:
#	print("\n\n")
#	params = optimize_params_2(num_PEs, 10000, 10, 10, A_rows, B_cols, A_cols_B_rows)
#run_MM_dataflow(A_rows, A_cols_B_rows, B_cols, params, num_PEs)