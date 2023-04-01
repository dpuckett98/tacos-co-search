from src.types import Node, Port

# Defines PE Array of width and height, doing a matrix multiply of width x tile_length by height x tile_length, with an initialization cost of initialization_time
# Matrix A (in port A) corresponds to height; matrix B (in port B) corresponds to width
# Computing mode: read in width + height elements from separate sources & multiply for tile_length cycles, then switch to flushing mode
# Flushing mode: height data elements are flushed out for width cycles
class PENode(Node):

	def __init__(self, name, width, height, tile_length, transmit_rate, initialization_time=0):
		Node.__init__(self, name, [Port("in_A", False, True, transmit_rate), Port("in_B", False, True, transmit_rate), Port("out", True, False, transmit_rate)])
		self.initialization_time = initialization_time
		self.width = width
		self.height = height
		self.tile_length = tile_length
		#self.A_buf_name = A_buf_name # corresponds to height
		#self.B_buf_name = B_buf_name # corresponds to width
		# amount of data currently stored in A, B, and O buffers:
		self.curr_A_buffer = 0
		self.curr_B_buffer = 0
		self.curr_O_buffer = 0
		self.mode = "computing"
		self.computing_counter = -self.initialization_time
		self.flushing_counter = 0
		
		# setting flags
		self.set_flag("elements processed", 0)
		self.set_flag("elements input", 0)
		#self.set_flag("tiles processed", 0) # Maybe need this?
	
	# Sends 1 bitwidth worth of data from this Node to the requester & resets the ready signal
	def send_data(self, port_name, amount):
		# empty O buffer
		self.curr_O_buffer -= amount
		self.set_flag("elements processed", self.get_flag("elements processed") + amount)
		assert self.curr_O_buffer >= 0 # amount of data stored in O buffer cannot be negative
		return
	
	# Receive up to 1 bitwidth worth of data from the source to this node
	def receive_data(self, port_name, amount):
		# fill A or B Buffer
		if port_name == "in_A":
			self.curr_A_buffer += amount
			if self.curr_A_buffer == self.height and self.curr_B_buffer == self.width:
				self.set_flag("elements input", self.get_flag("elements input") + 1)
			assert self.curr_A_buffer <= self.height
		elif port_name == "in_B":
			self.curr_B_buffer += amount
			if self.curr_A_buffer == self.height and self.curr_B_buffer == self.width:
				self.set_flag("elements input", self.get_flag("elements input") + 1)
			assert self.curr_B_buffer <= self.width
		else:
			raise ValueError("Port should either be in_A or in_B")
		return
	
	def pre_step(self, current_cycle):
		
		# if both A and B are filled and the full tile hasn't been computed, then do computation -> empty A and B buffers & increment counter
		if self.mode == "computing":
			if self.curr_A_buffer == self.height and self.curr_B_buffer == self.width and self.computing_counter < self.tile_length:
				self.curr_A_buffer = 0
				self.curr_B_buffer = 0
				self.computing_counter += 1
				# if counter is equal to tile length, then switch modes to flushing buffer & reset counter
				if self.computing_counter == self.tile_length:
					self.mode = "flushing"
					self.computing_counter = -self.initialization_time # this is where initialization_time is counted
		elif self.mode == "flushing":
			if self.curr_O_buffer == 0:
				self.curr_O_buffer = self.height
				self.flushing_counter += 1
				if self.flushing_counter == self.width:
					self.flushing_counter = 0
					self.mode = "computing"
	
	def step_request(self, current_cycle):
		
		# if either A or B buffer is empty, request for it to be filled
		if self.curr_A_buffer == 0 and not self.ports["in_A"].is_current_request():
			self.ports["in_A"].act(self.height, current_cycle)
			#self.sources[self.A_buf_name].request(self, self.height, current_cycle)
		if self.curr_B_buffer == 0 and not self.ports["in_B"].is_current_request():
			self.ports["in_B"].act(self.width, current_cycle)
			#self.sources[self.B_buf_name].request(self, self.width, current_cycle)
		
		
	def step_ready(self, current_cycle):
		
		# if O buffer is filled, flip ready
		if self.ports["out"].is_current_request() and self.curr_O_buffer > 0:
			self.ports["out"].act(min(self.curr_O_buffer, self.ports["out"].get_transmit_rate()), current_cycle)
		#if (not self.sinks_request[list(self.sinks)[0]] == None) and self.curr_O_buffer > 0:
		#	self.sinks_ready[list(self.sinks)[0]] = min(self.curr_O_buffer, self.sinks_bitwidth[list(self.sinks)[0]])
		

	def print_status(self):
		print("Mode:", self.mode)
		print("Computing counter:", self.computing_counter)
		print("Flushing counter:", self.flushing_counter)
		print("A Buf Elements:", self.curr_A_buffer)
		print("B Buf Elements:", self.curr_B_buffer)
		print("O Buf Elements:", self.curr_O_buffer)