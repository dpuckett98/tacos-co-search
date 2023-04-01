from src.types import Node, Port

class MemoryControllerNode(Node):

	# on_chip_bandwidth is bandwidth between the memory controller and on-chip nodes
	# off_chip_bandwidth is bandwidth between the memory controller and DRAM
	# DRAM ports are called "DRAM_in" and "DRAM_out"
	# the number and name of on-chip ports are given in "input_port_names" and "output_port_names"
	def __init__(self, name, off_chip_bandwidth, on_chip_bandwidth, input_port_names, output_port_names):
		# create ports & initialize node
		input_ports = [Port(name, False, False, on_chip_bandwidth) for name in input_port_names]
		output_ports = [Port(name, True, False, on_chip_bandwidth) for name in output_port_names]
		DRAM_ports = [Port("DRAM_in", False, True, off_chip_bandwidth), Port("DRAM_out", True, True, off_chip_bandwidth)]
		Node.__init__(self, name, DRAM_ports + input_ports + output_ports)
		
		self.input_groups = []
		self.output_groups = []
		
		self.DRAM_in_group = None
		self.DRAM_out_group = None
		
		# buffer data
		self.min_buffer_size = 0
		self.min_buffer_banks = 0
		
		# stats
		self.reads = 0
		self.writes = 0
	
	def add_output_group(self, group_name, port_names):
		# make sure direction of ports matches the direction of the group
		for port_name in port_names:
			assert self.ports[port_name].is_source == True
		
		# remove ports from other groups
		for group in self.output_groups:
			for port_name in port_names:
				if self.ports[port_name] in group.ports:
					group.ports.remove(self.ports[port_name])
		
		# create new group
		self.output_groups.append(OutputGroup(group_name, [self.ports[name] for name in port_names]))
	
	def add_input_group(self, group_name, port_names):
		# make sure direction of ports matches the direction of the group
		for port_name in port_names:
			assert self.ports[port_name].is_source == False
		
		# remove ports from other groups
		for group in self.input_groups:
			for port_name in port_names:
				if self.ports[port_name] in group.ports:
					group.ports.remove(self.ports[port_name])
		
		# create new group
		self.input_groups.append(InputGroup(group_name, [self.ports[name] for name in port_names]))
	
	def send_data(self, port_name, amount): # TODO
		# Send data to DRAM
		# if there's a group trying to send data to DRAM, then inform the group data was sent
		if port_name == "DRAM_out" and self.DRAM_out_group != None:
			self.DRAM_out_group.send_data(amount)
			# reset if that group has finished sending data
			if self.DRAM_out_group.request_sent == False:
				self.DRAM_out_group = None
			return
		
		# Send data to PEs
		# get group the data is being sent from
		group = None
		for g in self.output_groups:
			if self.ports[port_name] in g.ports:
				group = g
				break
		if group == None:
			raise ValueError("Port name not found in output groups: " + port_name)
		
		# write data to that port
		group.send_data(port_name, amount)
	
	def receive_data(self, port_name, amount):
		
		# receive data from DRAM
		if port_name == "DRAM_in":
			self.DRAM_in_group.receive_data(amount)
			return
		
		# receive data from input ports
		# first find the group it's in
		for group in self.input_groups:
			if self.ports[port_name] in group.ports:
				# tell the group it got data
				group.receive_data(amount)
				# if this port is finished, then tell the group
				if not self.ports[port_name].is_current_request():
					group.finish_port()
				return
	
	
	def step_request(self, current_cycle): # TODO

		# request data from DRAM
		# if DRAM_in isn't being used...
		if not self.ports["DRAM_in"].is_current_request():
			self.DRAM_in_group = None
			# for each group...
			for group in self.output_groups:
				# if the group hasn't already requested data...
				if not group.request_sent:
					# check if this group needs data
					need_data = False
					amount_needed = -1
					for port in group.ports:
						if port.is_current_request():
							need_data = True
							amount_needed = port.get_elements_requested()
							break
					# if this group does neeed data, send a request for the data and mark this group as receiving data
					if need_data:
						group.request_sent = True
						group.current_request_amount = amount_needed
						self.ports["DRAM_in"].act(amount_needed, current_cycle)
						self.DRAM_in_group = group
						break
		
		# request to send data to DRAM
		if self.DRAM_out_group == None:
			# for each group...
			for group in self.input_groups:
				# if the group hasn't already asked for data...
				if not group.request_sent:
					# if the group needs to send data...
					if group.ports_done == len(group.ports):
						group.request_sent = True
						self.ports["DRAM_out"].act(group.current_stored, current_cycle)
						self.DRAM_out_group = group
						break
	
	def step_ready(self, current_cycle):
		
		# signal ready to receive data from DRAM
		# for each output group...
		for group in self.output_groups:
			# if it currently has data to send...
			if group.request_sent:
				# send data to each port if it's asked for data & doesn't have all the current data yet
				for index, port in enumerate(group.ports):
					if port.is_current_request() and group.sent_to_ports[index] < group.current_stored:
						amount_to_send = min(group.current_stored - group.sent_to_ports[index], port.get_transmit_rate())
						port.act(amount_to_send, current_cycle)
		
		# TODO: DRAM write requests
		# signal ready to receive data from PEs
		# for each input group
		for group in self.input_groups:
			# if it isn't currently flushing data...
			if not group.request_sent:
				# receive data from each port if it's asked to send it
				for port in group.ports:
					if port.is_current_request():
						port.act_max(current_cycle)
	
	def post_step(self, current_cycle):
		self.set_flag("idle", "True")
		for port in self.ports.values():
			if port.is_current_request():
				self.set_flag("idle", "False")
				return
	
	def print_status(self): # TODO
		print("--", self.name, "--")
		print("Output groups:")
		for group in self.output_groups:
			group.print_state()
		print("Input groups:")
		for group in self.input_groups:
			group.print_state()

# Assume that a request coming in from one of these ports will be duplicated on the other ports
# Thus, if there's no current request (from this group), whenever a port asks for data, request for that amount of data to be loaded from DRAM
# As data is available from DRAM, send it out to each port that is ready for it
# E.g. if port n has requested data & sent_to_ports[n] < current_stored_data, then can send up to current_stored_data - sent_to_ports[n] data to port n
# At the same time, port m may have just sent a request, and may be sent different data
# Keep track of amount of data loaded & amount of data sent to each group
class OutputGroup:
	def __init__(self, name, ports):
		self.name = name
		self.ports = ports
		
		self.sent_to_ports = [0 for port in self.ports]
		self.current_stored = 0
		self.request_sent = False
		self.current_request_amount = -1
		self.ports_done = 0
	
	def receive_data(self, amount):
		self.current_stored += amount
	
	def send_data(self, port_name, data_amount):
		# get the port index
		port_index = -1
		for index, port in enumerate(self.ports):
			if port.name == port_name:
				port_index = index
				break
		
		self.sent_to_ports[port_index] += data_amount
		if self.sent_to_ports[index] == self.current_request_amount:
			self.ports_done += 1
			if self.ports_done == len(self.ports):
				self.reset()
	
	def reset(self):
		self.sent_to_ports = [0 for port in self.ports]
		self.current_stored = 0
		self.request_sent = False
		self.current_request_amount = -1
		self.ports_done = 0
	
	def print_state(self):
		print("Group:", self.name)
		print("Request sent:", self.request_sent)
		print("Current stored:", self.current_stored)
		print("Current request amount:", self.current_request_amount)
		print("Ports done:", self.ports_done)

# Assuming each port in the Input Group will send data to the memory buffer to be sent to DRAM
# The size of each request can be different
# The InputGroup waits for each port to finish sending one request, then sends the entire chunk of data to DRAM
class InputGroup:
	def __init__(self, name, ports):
		self.name = name
		self.ports = ports
		
		self.current_stored = 0
		self.request_sent = False
		self.ports_done = 0
	
	def receive_data(self, amount):
		self.current_stored += amount
	
	def finish_port(self):
		self.ports_done += 1
	
	def send_data(self, amount):
		self.current_stored -= amount
		if self.current_stored == 0:
			self.reset()
	
	def reset(self):
		self.current_stored = 0
		self.request_sent = False
		self.ports_done = 0
	
	def print_state(self):
		print("Group:", self.name)
		print("Request sent:", self.request_sent)
		print("Current stored:", self.current_stored)
		print("Ports done:", self.ports_done)