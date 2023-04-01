from .types import Node, Port

class DRAMNode(Node):

	def __init__(self, name, transmit_rate, initialization_time=0, percent_bandwidth_out=0.7):
		Node.__init__(self, name, [Port("in", False, False, transmit_rate), Port("out", True, False, transmit_rate)])
		#Node.__init__(self, name, [Port("in", False, False, int(transmit_rate * (1 - percent_bandwidth_out))), Port("out", True, False, int(transmit_rate * percent_bandwidth_out))])
		self.initialization_time = initialization_time
		
		# stats
		self.reads = 0
		self.writes = 0
	
	# Sends 1 bitwidth worth of data from this Node to the requester & resets the ready signal
	def send_data(self, port_name, amount):
		self.reads += amount
		return
	
	# Receive up to 1 bitwidth worth of data from the source to this node
	def receive_data(self, port_name, amount):
		self.writes += amount
		return
	
	# DRAM should never be in charge of a connection
	def step_request(self, current_cycle):
		return
	
	# DRAM passes a bunch of data whenever asked
	def step_ready(self, current_cycle):
		if self.ports["in"].connection != None and self.ports["in"].is_current_request():
			self.ports["in"].act_max(current_cycle)
		if self.ports["out"].connection != None and self.ports["out"].is_current_request():
			self.ports["out"].act_max(current_cycle)
		
		# ready to send data initialization_time cycles after it has been requested
		#for key in self.sinks:
		#	if (not self.sinks_request[key] == None) and self.sinks_request[key][1] <= current_cycle - self.initialization_time:
		#		self.sinks_ready[key] = self.sinks_bitwidth[key]
	
	def print_status(self):
		return