import numpy as np
import math

from src.types import Node, connect, StateMachine, Port
from src.buffer_node import BufferNode
from src.dram_node import DRAMNode
from src.complex_PE_node import ComplexPENode
from src.memory_controller_node import MemoryControllerNode
from src.controller import run_system

# Does matrix-matrix multiplication using multiple PEs
class MMDataflow(StateMachine):

	def __init__(self, num_PEs, num_banks_per_PE, size_banks_per_PE, bandwidth, A_rows, A_cols_B_rows, B_cols):
		# configuration and initialization
		self.num_PEs = num_PEs
		self.num_banks_per_PE = num_banks_per_PE
		self.size_banks_per_PE = size_banks_per_PE
		self.bandwidth = bandwidth
		self.A_rows = A_rows
		self.A_cols_B_rows = A_cols_B_rows
		self.B_cols = B_cols
		
		# setting up nodes
		
		# DRAM
		self.dram = DRAMNode("DRAM", self.bandwidth, initialization_time=0)
		
		# PE
		self.pes = [ComplexPENode("PE", self.num_banks_per_PE, self.size_banks_per_PE, self.bandwidth) for i in range(self.num_PEs)]
		
		# Memory Controller
		memory_input_ports = ["in_psums_" + str(i) for i in range(self.num_PEs)]
		memory_output_ports = [item for i in range(self.num_PEs) for item in ["out_A_" + str(i), "out_B_" + str(i), "out_psums_" + str(i)]]
		self.memory = MemoryControllerNode("MemoryController", bandwidth, bandwidth, memory_input_ports, memory_output_ports)
		
		# Connections
		connect(self.dram, "out", self.memory, "DRAM_in")
		connect(self.dram, "in", self.memory, "DRAM_out")
		for i in range(self.num_PEs):
			connect(self.memory, "in_psums_" + str(i), self.pes[i], "out")
			connect(self.memory, "out_A_" + str(i), self.pes[i], "in_A")
			connect(self.memory, "out_B_" + str(i), self.pes[i], "in_B")
			connect(self.memory, "out_psums_" + str(i), self.pes[i], "in_psums")
		
		# initialize PE
		for pe in self.pes:
			pe.set_operation(self.A_rows // 2, self.A_cols_B_rows, self.B_cols // 2, False, True, False, False)
		self.num_chunks_processing = self.num_PEs
		self.num_chunks_processed = 0
		self.num_chunks_to_process = 4
		
		# initialize memory
		for i in range(self.num_PEs):
			self.memory.add_output_group("A_" + str(i), ["out_A_" + str(i)])
			self.memory.add_output_group("B_" + str(i), ["out_B_" + str(i)])
			self.memory.add_output_group("psums_" + str(i), ["out_psums_" + str(i)])
			self.memory.add_input_group("psums_" + str(i), ["in_psums_" + str(i)])
		
		self.total_cycles = -1
	
	def get_nodes(self):
		return [self.dram, self.memory] + [pe for pe in self.pes]

	def finished(self):
		return self.num_chunks_processed == self.num_chunks_to_process and self.memory.get_flag("idle") == "True"

	def update(self, current_cycle):
		for pe in self.pes:
			if pe.get_state() == "idle":
				if pe.get_flag("finished"):
					self.num_chunks_processed += 1
				if self.num_chunks_processing < self.num_chunks_to_process:
					pe.set_operation(self.A_rows // 2, self.A_cols_B_rows, self.B_cols // 2, False, True, False, False)
					self.num_chunks_processing += 1
	
		self.print_data(current_cycle)

	def print_data(self, current_cycle):
		print("###############", current_cycle, "###############")
		print("Number of Chunks Processed:", self.num_chunks_processed, "out of", self.num_chunks_to_process)
		for pe in self.pes:
			pe.print_status()
		self.memory.print_status()
	
	def on_complete(self, final_cycles):
		self.total_cycles = final_cycles
		self.print_on_complete(final_cycles)
	
	def print_on_complete(self, final_cycles):
		print("Final cycle count:", final_cycles)
		ideal_cycles = self.A_rows * self.A_cols_B_rows * self.B_cols / self.num_PEs
		print("Utilization:", ideal_cycles / final_cycles * 100, "%")

# returns total_cycles, utilization, a buffer min size, reads, writes, b buffer min size, reads, writes, o buffer min size, reads, writes, dram accesses, reads, writes
# PE_config = 0 means PE_width and PE_height are given; PE_config = 1 means PE_width and PE_height are switched
def run_MM_dataflow(A_rows, A_cols_B_rows, B_cols):
	
	state_machine = MMDataflow(2, (A_rows + B_cols) // 2, A_cols_B_rows, 10, A_rows, A_cols_B_rows, B_cols)

	run_system(state_machine, state_machine.get_nodes(), 100000000, False)

run_MM_dataflow(10, 10, 10)