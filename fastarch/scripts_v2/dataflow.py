import numpy as np
import math

from src.types import Node, connect, StateMachine, Port
from src.buffer_node import BufferNode
from src.dram_node import DRAMNode
from src.complex_PE_node import ComplexPENode
from src.controller import run_system

# Does matrix-matrix multiplication on one PE
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
		self.dram_a = DRAMNode("DRAM A", self.bandwidth, initialization_time=0)
		self.dram_b = DRAMNode("DRAM B", self.bandwidth, initialization_time=0)
		self.dram_psums = DRAMNode("DRAM psums", self.bandwidth, initialization_time=0)
		
		# PE
		self.pe = ComplexPENode("PE", self.num_banks_per_PE, self.size_banks_per_PE, self.bandwidth)
		
		# Connections
		connect(self.dram_a, "out", self.pe, "in_A")
		connect(self.dram_b, "out", self.pe, "in_B")
		connect(self.dram_psums, "out", self.pe, "in_psums")
		connect(self.dram_psums, "in", self.pe, "out")
		
		# initialize PE
		self.pe.set_operation(self.A_rows // 2, self.A_cols_B_rows, self.B_cols // 2, False, True, False, False)
		self.num_chunks_processed = 0
		self.num_chunks_to_process = 4
		
		self.total_cycles = -1
	
	def get_nodes(self):
		return [self.dram_a, self.dram_b, self.dram_psums, self.pe]

	def finished(self):
		return self.num_chunks_processed == self.num_chunks_to_process

	def update(self, current_cycle):
		if self.pe.get_state() == "idle":
			self.num_chunks_processed += 1
			if self.num_chunks_processed < self.num_chunks_to_process:
				self.pe.set_operation(self.A_rows // 2, self.A_cols_B_rows, self.B_cols // 2, False, True, False, False)
	
		self.print_data(current_cycle)

	def print_data(self, current_cycle):
		print("###############", current_cycle, "###############")
		self.pe.print_status()
	
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
	
	state_machine = MMDataflow(1, (A_rows + B_cols) / 2, A_cols_B_rows, 10, A_rows, A_cols_B_rows, B_cols)

	run_system(state_machine, state_machine.get_nodes(), 1000000000)

run_MM_dataflow(10, 10, 10)