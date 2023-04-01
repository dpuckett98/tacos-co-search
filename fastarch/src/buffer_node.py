from .types import Node, Port
#from logger import log

# States:
# "filling" sends a request for data whenever it is empty and flushes data when it is read (e.g. acts similarly to a FIFO)
# "stationary" does not request for data, and doesn't flush data when read
class BufferNode(Node):

	def __init__(self, name, max_size, transmit_rate, out_requester, initial_size=0):
		Node.__init__(self, name, [Port("in", False, True, transmit_rate), Port("out", True, out_requester, transmit_rate)])
		self.max_size = max_size
		self.current_size = initial_size
		
		# initialize flags
		self.flags["request count"] = 0
		self.set_flag("elements delivered", 0)
		
		# stats
		self.max_used = 0
		self.reads = 0
		self.writes = 0
		self.accesses = 0
	
	# Sends 1 bitwidth worth of data from this Node to the requester & resets the ready signal
	def send_data(self, port_name, amount):
		self.reads += amount
		self.set_flag("elements delivered", self.get_flag("elements delivered") + amount)
		if self.state != "stationary":
			self.current_size -= amount
		assert self.current_size >= 0 # buffer should never be in the negative
	
	# Receive up to 1 bitwidth worth of data from the source to this node
	def receive_data(self, port_name, amount):
		self.current_size += amount
		self.writes += amount
		if self.current_size > self.max_used:
			self.max_used = self.current_size
		assert self.current_size <= self.max_size # buffer should never be over full
	
	def step_request(self, current_cycle):
		#assert len(self.sinks) == 1 and len(self.sources) == 1 # BufferNode must have 1 source and 1 sink
		
		#source = self.sources[list(self.sources.keys())[0]]
		#sink = self.sinks[list(self.sinks.keys())[0]]
		
		# send request for data if not already requested, empty and if in "filling" state
		if self.state == "filling" and self.current_size == 0:
			if not self.ports["in"].is_current_request():
				self.ports["in"].act(self.max_size, current_cycle)
				#source.request(self, self.max_size - self.current_size, current_cycle)
				self.flags["request count"] += 1
				self.accesses += 1
		
		# if output requester, then request for data to be read when it is full
		if self.ports["out"].is_requester and self.current_size == self.max_size:
			self.ports["out"].act(self.current_size, current_cycle)
		
		
	def step_ready(self, current_cycle):
		#source = self.sources[list(self.sources.keys())[0]]
		#sink = self.sinks[list(self.sinks.keys())[0]]
		
		# ready to send data if it has been requested & there is currently data in the buffer
		if not self.ports["out"].is_requester and self.ports["out"].is_current_request() and self.current_size > 0:
			self.ports["out"].act(min(self.current_size, self.ports["out"].get_transmit_rate(), self.ports["out"].get_elements_requested()), current_cycle)
		#if (not self.sinks_request[sink.name] == None) and self.current_size > 0:
		#	self.sinks_ready[sink.name] = min(self.current_size, self.sinks_bitwidth[sink.name])
	
	def post_step(self, current_cycle):
		#log(current_cycle, self.name, "state", self.state)
		#log(current_cycle, self.name, "size", str(self.current_size))
		pass
	
	def print_status(self):
		print(self.name, "current data:", self.current_size)
		print(self.name, "state:", self.state)
	
	def empty(self):
		self.current_size = 0