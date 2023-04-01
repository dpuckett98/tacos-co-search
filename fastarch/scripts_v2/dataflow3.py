import numpy as np
import math

from src.types import Node, connect, StateMachine, Port
from src.buffer_node import BufferNode
from src.dram_node import DRAMNode
from src.complex_PE_node import ComplexPENode
from src.memory_controller_node_v2 import MemoryControllerNode
from src.controller import run_system

# Does matrix-matrix multiplication using multiple PEs
# dataflow is either "output-matrix" or "output-vector"
# PE can consume 1 chunk of A and 1 chunk of B to produce 1 chunk of O
# Chip can consume 1 tile of A and 1 tile of B to produce 1 partial tile of O

# outer dataflow (e.g. how tiles are loaded/offloaded -- A-Stationary, B-Stationary, or Output-Stationary)
# inner dataflow (e.g. how chunks are loaded/offloaded in a single tile -- A-Stationary, B-Stationary, Output-Stationary)
# PE dataflow (e.g. how chunks are computed inside a PE -- A-Stationary, B-Stationary, or Output-Stationary) --> this one probably is unnecessary
class MMDataflow(StateMachine):

	def __init__(self, num_PEs, num_banks_per_PE, size_banks_per_PE, off_chip_bandwidth, on_chip_bandwidth, A_rows, A_cols_B_rows, B_cols):
		# configuration and initialization
		self.num_PEs = num_PEs
		self.num_banks_per_PE = num_banks_per_PE
		self.size_banks_per_PE = size_banks_per_PE
		self.off_chip_bandwidth = off_chip_bandwidth
		self.on_chip_bandwidth = on_chip_bandwidth
		self.A_rows = A_rows
		self.A_cols_B_rows = A_cols_B_rows
		self.B_cols = B_cols
		#self.dataflow = dataflow
		
		self.A_buffer_size = 100
		self.B_buffer_size = 100
		self.O_buffer_size = 100
		
		# setting up nodes
		
		# DRAM
		self.dram = DRAMNode("DRAM", self.off_chip_bandwidth, initialization_time=0)
		
		# PE
		self.pes = [ComplexPENode("PE", self.num_banks_per_PE, self.size_banks_per_PE, self.on_chip_bandwidth) for i in range(self.num_PEs)]
		
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
		self.x = self.num_banks_per_PE // 2
		self.z = self.num_banks_per_PE // 2 #- self.x
		self.y = self.size_banks_per_PE
		# columns & rows & depth per A and B buffer... e.g. A buffer holds 1 A tile = chunk_rows * chunk_depth chunks at once, B buffer holds chunk_cols * chunk_depth chunks at once
		self.total_A_chunks_per_tile = math.ceil(self.A_buffer_size / (self.x * self.y))
		self.total_B_chunks_per_tile = math.ceil(self.B_buffer_size / (self.z * self.y))
		self.total_O_chunks_per_tile = math.ceil(self.O_buffer_size / (self.x * self.z))
		
		self.chunk_depth = (self.total_A_chunks_per_tile * self.total_B_chunks_per_tile / self.total_O_chunks_per_tile) ** 0.5
		self.chunk_cols = self.total_O_chunks_per_tile / self.total_A_chunks_per_tile * self.chunk_depth
		self.chunk_rows = self.total_A_chunks_per_tile / self.chunk_depth
		
		self.chunk_depth = math.floor(self.chunk_depth)
		self.chunk_cols = math.floor(self.chunk_cols)
		self.chunk_rows = math.floor(self.chunk_rows)
		
		self.tile_rows = math.ceil(self.A_rows / (self.chunk_rows * self.x)) # rows of tiles in the entire mult.
		self.tile_cols = math.ceil(self.B_cols / (self.chunk_cols * self.z))
		self.tile_depth = math.ceil(self.A_cols_B_rows / (self.chunk_depth * self.y))
		
		print("Chunk dim:", self.x, self.y, self.z)
		print("Chunks per tile dims:", self.chunk_rows, self.chunk_cols, self.chunk_depth)
		print("Matrix dims:", self.tile_rows, self.tile_cols, self.tile_depth)
		
		print("A buffer utilization:", (self.x * self.y) * self.chunk_rows * self.chunk_depth / self.A_buffer_size * 100, "%")
		print("B buffer utilization:", (self.z * self.y) * self.chunk_cols * self.chunk_depth / self.B_buffer_size * 100, "%")
		print("O buffer utilization:", (self.x * self.z) * self.chunk_rows * self.chunk_cols / self.O_buffer_size * 100, "%")
		
		# initialize tile_sets & memory controller
		# create an empty list for each tile
		self.super_tile_sets = [[[[] for i in range(self.tile_depth)] for j in range(self.tile_cols)] for k in range(self.tile_rows)]
		for i in range(math.ceil(self.A_rows / self.x)):
			for j in range(math.ceil(self.B_cols / self.z)):
				for k in range(math.ceil(self.A_cols_B_rows / self.y)):
					#print(i, j, k)
					tile_row = math.floor(i / self.chunk_rows)
					tile_col = math.floor(j / self.chunk_cols)
					tile_depth = math.floor(k / self.chunk_depth)
					self.super_tile_sets[tile_row][tile_col][tile_depth].append(Tile("A_" + str(i) + "_" + str(k), "B_" + str(k) + "_" + str(j), "Out_" + str(i) + "_" + str(j), self.x, self.z, self.y, k % self.chunk_depth == 0))
					self.memory.add_chunk("A_" + str(i) + "_" + str(k), self.x * self.y)
					self.memory.add_chunk("B_" + str(k) + "_" + str(j), self.z * self.y)
					self.memory.add_chunk("Out_" + str(i) + "_" + str(j), self.x * self.z)
		
		self.super_tile_pos_rows = 0
		self.super_tile_pos_cols = 0
		self.super_tile_pos_depth = 0
		self.tile_sets = self.super_tile_sets[0][0][0]
		self.finished_tile_sets = []
		print(len(self.tile_sets))
		
		self.num_tile_sets = math.ceil(self.A_rows / self.x) * math.ceil(self.B_cols / self.z) * math.ceil(self.A_cols_B_rows / self.y)
		self.num_tile_sets_completed = 0
		
		# initialize PE
		self.current_pe_assignments = {} # PE to tile_set
		for pe in self.pes:
			self.choose_next_tile_set(pe)
		
		self.total_cycles = -1
	
	def choose_next_tile_set(self, pe):
		# give PE a command from the current tile
		done = self.select_from_tile(pe)
		
		# if the PE didn't get a tile...
		if not done:
			# if the current tile has been finished, select a new tile and give the PE a command from it
			if len(self.tile_sets) == 0:
				self.select_next_tile()
				self.select_from_tile(pe)
	
	# assigns tile_set to the given PE; returns False when no tile set assigned
	def select_from_tile(self, pe):
		# if there are tile_sets left using the same output chunk, assign that tile_set
		#print("Assigning tile_set")
		if pe in self.current_pe_assignments.keys() and self.current_pe_assignments[pe] != None:
			self.finished_tile_sets.append(self.current_pe_assignments[pe])
			#print("PE found in PE_Assignments")
			new_tile_set = None
			num_available_tiles = 0
			old_tile_set = self.current_pe_assignments[pe]
			for tile_set in self.tile_sets:
				if tile_set.chunk_name_Out == old_tile_set.chunk_name_Out:
					if new_tile_set == None:
						new_tile_set = tile_set
					num_available_tiles += 1
			if new_tile_set != None:
				#print("Number of available tiles:", num_available_tiles)
				#print("Chosen tile:", new_tile_set.
				self.current_pe_assignments[pe] = new_tile_set
				self.memory.set_chunk_for_port(pe.ports["in_A"].connection.source_port.name, new_tile_set.chunk_name_A)
				self.memory.set_chunk_for_port(pe.ports["in_B"].connection.source_port.name, new_tile_set.chunk_name_B)
				self.memory.set_chunk_for_port(pe.ports["out"].connection.sink_port.name, new_tile_set.chunk_name_Out)
				pe.set_operation(new_tile_set.rows, new_tile_set.width, new_tile_set.columns, False, True, num_available_tiles == 1, False, False)
				self.tile_sets.remove(new_tile_set)
				print("Removed one", len(self.tile_sets))
				return True
			
			# if PE used to have an assignment but there's nothing left, then the corresponding output chunk was finished -- we can go ahead and offload that chunk
			#print(self.current_pe_assignments[pe].chunk_name_Out)
			#self.memory.give_save_chunk_request(self.current_pe_assignments[pe].chunk_name_Out)
		#print("Starting new tile set")
		# otherwise, assign a tile set that is at the beginning of a run (e.g. initializing_out_chunk==True)
		new_tile_set = None
		num_available_tiles = 0
		for tile_set in self.tile_sets:
			if new_tile_set == None:
				if tile_set.initializing_out_chunk:
					new_tile_set = tile_set
					num_available_tiles += 1
			else:
				if tile_set.chunk_name_Out == new_tile_set.chunk_name_Out:
					num_available_tiles += 1
		if new_tile_set != None:
			self.current_pe_assignments[pe] = tile_set
			self.memory.set_chunk_for_port(pe.ports["in_A"].connection.source_port.name, tile_set.chunk_name_A)
			self.memory.set_chunk_for_port(pe.ports["in_B"].connection.source_port.name, tile_set.chunk_name_B)
			self.memory.set_chunk_for_port(pe.ports["out"].connection.sink_port.name, tile_set.chunk_name_Out)
			pe.set_operation(tile_set.rows, tile_set.width, tile_set.columns, False, False, num_available_tiles == 1, False, False)
			self.tile_sets.remove(tile_set)
			print("Removed one", len(self.tile_sets))
			return True
		# if none of those exist, leave the PE idle & remove its assignment
		self.current_pe_assignments[pe] = None
		return False
	
	def select_next_tile(self):
		print("here")
		# output stationary
		# clear all A and B
		for tile_set in self.finished_tile_sets:
			self.memory.erase_chunk(tile_set.chunk_name_A)
			self.memory.erase_chunk(tile_set.chunk_name_B)
		# if this isn't the last tile for this output:
		if self.super_tile_pos_depth + 1 < self.tile_depth:
			print("Getting new tile in same output row")
			# set the new tile_set list
			self.super_tile_pos_depth += 1
			self.tile_sets = self.super_tile_sets[self.super_tile_pos_rows][self.super_tile_pos_cols][self.super_tile_pos_depth]
		# if this is the last tile
		else:
			print("Getting new tile in different output row")
			# offload the output chunks
			for tile_set in self.finished_tile_sets:
				self.memory.give_save_chunk_request(tile_set.chunk_name_A)
			# set the new tile_set list
			self.super_tile_pos_depth = 0
			self.super_tile_pos_cols += 1
			if self.super_tile_pos_cols == self.tile_cols:
				self.super_tile_pos_cols = 0
				self.super_tile_pos_rows += 1
				if self.super_tile_pos_rows == self.tile_rows:
					print("finished")
					# if we get here, then we're done -- we've finished the last tile
					self.finished_tile_sets = []
					return
			self.tile_sets = self.super_tile_sets[self.super_tile_pos_rows][self.super_tile_pos_cols][self.super_tile_pos_depth]
		# reset finished tile sets
		self.finished_tile_sets = []
	
	def get_nodes(self):
		return [self.dram, self.memory] + [pe for pe in self.pes]

	def finished(self):
		#print(self.num_tile_sets_completed, self.num_tile_sets, self.memory.get_flag("idle"))
		#for pe in self.pes:
		#	if pe.get_state() != "idle":
		#		return False
		#if self.memory.get_flag("idle") != "True":
		#	return False
		#return True
		return self.num_tile_sets_completed == self.num_tile_sets and self.memory.get_flag("idle") == "True"

	def update(self, current_cycle):
		for pe in self.pes:
			if pe.get_state() == "idle":
				had_tile = (self.current_pe_assignments[pe] != None)
				self.choose_next_tile_set(pe)
				if had_tile:
					self.num_tile_sets_completed += 1
	
		#self.print_data(current_cycle)

	def print_data(self, current_cycle):
		print("###############", current_cycle, "###############")
		print("Number of Chunks Processed:", self.num_tile_sets_completed, "out of", self.num_tile_sets)
		for pe in self.pes:
			pe.print_status()
		self.memory.print_status()
	
	def get_percent_complete(self):
		return self.num_tile_sets_completed / self.num_tile_sets
	
	def on_complete(self, final_cycles):
		self.total_cycles = final_cycles
		self.print_on_complete(final_cycles)
	
	def print_on_complete(self, final_cycles):
		ideal_cycles = self.A_rows * self.A_cols_B_rows * self.B_cols / self.num_PEs
		print("Final cycle count:", final_cycles)
		print("Ideal cycle count:", ideal_cycles)
		print("Utilization:", ideal_cycles / final_cycles * 100, "%")
		print("Global Buffer Size:", self.memory.get_min_buffer_size())
		print("Total PE Memory:", self.num_PEs * self.num_banks_per_PE * self.size_banks_per_PE)

class Tile:
	def __init__(self, chunk_name_A, chunk_name_B, chunk_name_Out, rows, columns, width, initializing_out_chunk):
		self.chunk_name_A = chunk_name_A
		self.chunk_name_B = chunk_name_B
		self.chunk_name_Out = chunk_name_Out
		self.rows = rows
		self.columns = columns
		self.width = width
		self.initializing_out_chunk = initializing_out_chunk

# returns total_cycles, utilization, a buffer min size, reads, writes, b buffer min size, reads, writes, o buffer min size, reads, writes, dram accesses, reads, writes
# PE_config = 0 means PE_width and PE_height are given; PE_config = 1 means PE_width and PE_height are switched
def run_MM_dataflow(A_rows, A_cols_B_rows, B_cols):
	
	state_machine = MMDataflow(num_PEs = 1, num_banks_per_PE = 10, size_banks_per_PE = 10, off_chip_bandwidth = 100, on_chip_bandwidth = 10, A_rows = A_rows, A_cols_B_rows = A_cols_B_rows, B_cols = B_cols)

	run_system(state_machine, state_machine.get_nodes(), 100000000, False)

run_MM_dataflow(100, 100, 100)