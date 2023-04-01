
def connect(first, first_port_name, second, second_port_name):
	connection = Connection(first, first_port_name, second, second_port_name)
	#source.add_sink(sink, bitwidth)
	#sink.add_source(source, bitwidth)

class Node:

	def __init__(self, name, port_definitions):
		self.name = name
		self.state = ""
		self.ports = {}
		for pd in port_definitions:
			pd.set_node(self)
			self.ports[pd.name] = pd
		#self.source_port_names = source_port_names
		#self.sink_port_names = sink_port_names
		#self.ports = {}
		
		#self.sources = {}
		#self.sources_bitwidth = {} # bitwidth of connection with a source
		#self.sinks = {}
		#self.sinks_bitwidth = {} # bitwidth of connection with a sink
		#self.sinks_ready = {} # This is the number of bits this node is ready to send to the corresponding sink; default value is 0, max value is corresponding bitwidth
		#self.sinks_request = {} # Holds [the amount of data a sink has asked for, the cycle the request was sent on]
		
		self.flags = {} # dictionary of output flags that can be hooked up for state engine
	
	def add_port(self, port_name, connection):
		self.ports[port_name].set_connection(connection)
		#self.sources[obj.name] = obj
		#self.sources_bitwidth[obj.name] = bitwidth
	
	#def add_sink(self, port_name, connection):
	#	self.sinks[port_name] = connection
		#self.sinks[obj.name] = obj
		#self.sinks_bitwidth[obj.name] = bitwidth
		#self.sinks_ready[obj.name] = 0
		#self.sinks_request[obj.name] = None
	
	#def get_source_connection(self, port_name):
	#	return self.sources[port_name]
	
	#def get_sink_connection(self, port_name):
	#	return self.sinks[port_name]
	
	#def request(self, requester, data_amt, current_cycle):
	#	self.sinks_request[requester.name] = [data_amt, current_cycle]
	
	# Sends up to 1 bitwidth worth of data from this Node to the requester & resets the ready signal
	#def send_data(self, requester, amount):
	#	self.sinks_request[requester.name][0] -= amount
	#	if self.sinks_request[requester.name][0] <= 0:
	#		self.sinks_request[requester.name] = None
	#	self.sinks_ready[requester.name] = 0
	#	self._send_data(requester, amount)
	
	# handles sending amount of data from port_name
	def send_data(self, port_name, amount):
		raise NotImplementedError("send_data function not implemented")
	
	# handles receiving amount of data from port_name
	def receive_data(self, port_name, amount):
		raise NotImplementedError("receive_data function not implemented")
	
	#def get_ready_data(self, other):
	#	return self.sinks_ready[other]
	
	def get_state(self):
		return self.state
	
	def set_state(self, new_state):
		self.state = new_state
	
	# Optional method; handles any node updates before any requests are updated, but after state machine is updated
	def pre_step(self, current_cycle):
		return
	
	# This method should only modifies requests
	def step_request(self, current_cycle):
		raise NotImplementedError("step_request function not implemented")
	
	# This method should only modify ready signals
	def step_ready(self, current_cycle):
		raise NotImplementedError("step_ready function not implemented")
	
	# Optional method; run after all of the current cycle's requests & readys have been processed
	def post_step(self, current_cycle):
		return
	
	def get_flag(self, name):
		return self.flags[name]
	
	def set_flag(self, name, val):
		self.flags[name] = val
	
	def print_status(self):
		return

class Port:
	def __init__(self, name, is_source, is_requester, transmit_rate):
		self.name = name
		self.is_source = is_source
		self.is_requester = is_requester
		self.connection = None
		self.transmit_rate = transmit_rate
		self.node = None
	
	def set_node(self, node):
		self.node = node
	
	def set_connection(self, connection):
		self.connection = connection
	
	# transmits as much data as possible
	def act_max(self, cycle):
		self.connection.act(self.node, min(max(1, self.connection.transmit_rate), self.connection.current_request_amount), cycle)
	
	# transmits specific amount of data
	def act(self, amount, cycle):
		self.connection.act(self.node, amount, cycle)
	
	def is_current_request(self):
		return self.connection.current_request
	
	def get_elements_requested(self):
		return self.connection.current_request_amount
	
	def get_transmit_rate(self):
		return self.transmit_rate

# handles communication between two nodes (requester and responder)
# data flows from requester to responder if requester_source is True; otherwise it flows from responder to requester
class Connection:
	
	def __init__(self, first, first_port_name, second, second_port_name):
		first_port = first.ports[first_port_name]
		second_port = second.ports[second_port_name]
		assert first_port.is_requester != second_port.is_requester # one port must be a requester, the other must be a responder
		assert first_port.is_source != second_port.is_source # one port must be a source, the other must be a sink
		assert first_port.transmit_rate == second_port.transmit_rate # ports must have the same transmit rates
		
		# add connection to nodes
		first.add_port(first_port.name, self)
		second.add_port(second_port.name, self)
		
		# get requester/responder
		self.requester = first if first_port.is_requester else second
		self.responder = first if not first_port.is_requester else second
		# get source/sink
		self.source = first if first_port.is_source else second
		self.sink = first if not first_port.is_source else second
		# get source/sink ports
		self.source_port = first_port if first_port.is_source else second_port
		self.sink_port = first_port if not first_port.is_source else second_port
		# update other constructor variables
		self.transmit_rate = first_port.transmit_rate
		
		# prep internal vars
		self.current_request = False
		self.current_request_amount = -1
		self.current_request_cycle = -1
		self.current_ready = False
		self.current_ready_amount = -1
		
		# handling transmit rate < 1
		self.avail_bandwidth = 0
		
		#self.source = self.requester if requester_source == True else self.responder
		#self.sink = self.requester if requester_source == False else self.responder
		#self.source_port = self.requester_port if requester_source == True else self.responder_port
		#self.sink_port = self.requester_port if requester_source == False else self.responder_port
	
	# each node calls this function; requester is routed to _make_request, and responder is routed to _set_ready
	def act(self, node, amount, cycle):
		if node == self.requester:
			self._make_request(amount, cycle)
		elif node == self.responder:
			self._set_ready(amount)
		else:
			raise Exception("Node must be requester or responder")
	
	# requester should make the request
	# amount is the number of elements to be transmitted
	# TODO in future if needed: make "request queue" where additional requests can be queued (maybe extend the Connection class to preserve old behavior and enable different implementations of request queues?)
	def _make_request(self, amount, cycle):
		assert self.current_request == False # a new request should only be made when the old one is finished
		self.current_request = True
		self.current_request_amount = amount
		self.current_request_cycle = cycle
	
	# ready should be updated each cycle
	def _set_ready(self, amount):
		#print(self.responder.name)
		#print(self.source_port.name)
		#print(amount)
		assert self.current_request == True # should only flip ready if a request is made
		assert amount <= max(1, self.transmit_rate) # ready amount must be within the transmit rate, unless the transmit rate is less than 1
		assert amount <= self.current_request_amount # ready amount must be less than or equal to the amount requested
		assert amount >= 0 # ready amount must be positive
		self.current_ready = True
		self.current_ready_amount = amount
	
	# designed to be called by controller after each node has been updated
	# sends data from source to sink if a request has been made and the responder is ready
	def update(self, cycle):
		if self.current_ready and self.current_request:
			# update avail_bandwidth
			self.avail_bandwidth += self.transmit_rate
			#print(self.avail_bandwidth)
			# if there's enough "stored" bandwidth to transmit, then transmit
			if self.avail_bandwidth >= self.current_ready_amount:
				# remove used bandwidth
				self.avail_bandwidth -= self.current_ready_amount
				# update request; reset if request is fulfilled
				self.current_request_amount -= self.current_ready_amount
				if self.current_request_amount == 0:
					self.current_request = False
					self.current_request_amount = -1
					self.current_request_cycle = -1
				# tell the nodes that they sent/received the data
				self.source.send_data(self.source_port.name, self.current_ready_amount)
				self.sink.receive_data(self.sink_port.name, self.current_ready_amount)
				# reset ready
				self.current_ready = False
				self.current_ready_amount = -1

class StateMachine:

	def finished(self):
		raise NotImplementedError("StateMachine finished function not implemented")

	def update(self, current_cycle):
		raise NotImplementedError("StateMachine update function not implemented")
	
	def on_complete(self, final_cycles):
		raise NotImplementedError("StateMachine on_complete function not implemented")
	
	def get_percent_complete(self):
		raise NotImplementedError("StateMachine get_percent_complete function not implemented")