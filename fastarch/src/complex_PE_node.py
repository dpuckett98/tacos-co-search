import numpy as np

from .types import Node, Port

# Performs matrix multiplication with input data
class ComplexPENode(Node):

	def __init__(self, name, num_banks, size_banks, bandwidth):
		Node.__init__(self, name, [Port("in_A", False, True, bandwidth), Port("in_B", False, True, bandwidth), Port("in_psums", False, True, bandwidth), Port("out", True, True, bandwidth)])
		
		# parameters
		self.num_banks = num_banks
		self.size_banks = size_banks
		self.output_buffer_size = (self.num_banks * self.size_banks / 2) ** 2
		self.bandwidth = bandwidth
		
		# operation-specific parameters
		self.x = -1
		self.y = -1
		self.z = -1
		self.psum_load = False
		self.psum_reuse = False
		self.psum_flush = False
		self.reuse_A = False
		self.reuse_B = False
		self.x_loop = 0
		self.y_loop = 0
		self.z_loop = 0
		self.B_requested = False
		self.A_requested = False
		self.psums_load_requested = False
		self.psums_write_requested = False
		self.sparsity = 0.0
		self.sparse_map = None
		
		# state variables (initialized when operation selected)
		self.A_buffer = None
		self.B_buffer = None
		self.out_buffer = None
		self.current_A_buffer = 0
		self.current_B_buffer = 0
		self.current_out_buffer = 0
		self.fixed_out_buffer_size = False
		
		self.next_state = "idle"
		self.set_state("idle") # possible states: idle, computing, flushing
		self.set_flag("computing", False)
		
		# tracking variables
		self.num_chunks = 0 # number of chunks processed
		self.num_computations = 0 # number of MACs
		self.num_bank_writes = 0 # number of times data is written to any input bank
		self.num_bank_reads = 0 # number of times data is read from any input bank
		self.num_out_writes = 0 # number of times data is written to output bank
		self.num_out_reads = 0 # number of times data is read from output bank
	
	def set_operation(self, x, y, z, psum_load, psum_reuse, psum_flush, reuse_A, reuse_B, sparsity=0.0, sparse_map=None):
		#print(x*y + z*y, x*y*z*(1-sparsity))
		#if self.name == "PE_0":
		#	print(sparse_map)
		# reset old operation
		self.num_chunks += 1
		#print("Starting new op: num_comps:", self.num_computations)
		self.B_requested = False
		self.A_requested = False
		self.psums_load_requested = False
		self.psums_write_requested = False
		self.x_loop = 0
		self.y_loop = 0
		self.z_loop = 0
		self.sparsity = sparsity
		self.sparse_map = sparse_map
		# verify input fits with PE
		assert x + z <= self.num_banks
		assert y <= self.size_banks
		# reset A, B, out buffers if parameters change
		if x != self.x or y != self.y:
			self.A_buffer = [[False for i in range(y)] for j in range(x)] # 2D array; x rows, y columns
			self.current_A_buffer = 0
		if z != self.z or y != self.y:
			self.B_buffer = [[False for i in range(y)] for j in range(z)] # 2D array; z rows, y columns
			self.current_B_buffer = 0
		if x != self.x or z != self.z:
			self.out_buffer = [[False for i in range(z)] for j in range(x)] # 2D array; x rows, z columns
			self.current_out_buffer = 0
		# copy over parameters
		self.x = x
		self.y = y
		self.z = z
		self.psum_load = psum_load
		self.psum_reuse = psum_reuse
		self.psum_flush = psum_flush
		self.reuse_A = reuse_A
		self.reuse_B = reuse_B
		# initialize state variables
		if not self.reuse_A or self.A_buffer == None:
			self.A_buffer = [[False for i in range(self.y)] for j in range(self.x)] # 2D array; x rows, y columns
			self.current_A_buffer = 0
		if not self.reuse_B or self.B_buffer == None:
			self.B_buffer = [[False for i in range(self.y)] for j in range(self.z)] # 2D array; z rows, y columns
			self.current_B_buffer = 0
		if self.out_buffer == None or self.psum_load: # reset if it's never been initialized or if we're loading new data -- otherwise, just keep the old data
			self.out_buffer = [[False for i in range(self.z)] for j in range(self.x)] # 2D array; x rows, z columns
			self.current_out_buffer = 0
		if self.psum_load or self.psum_reuse: # don't change the output buffer size if we're reusing psums
			self.fixed_out_buffer_size = True
		else:
			self.fixed_out_buffer_size = False
		if self.reuse_B and self.current_B_buffer > 0:
			self.B_requested = True
		if self.reuse_A and self.current_A_buffer > 0:
			self.A_requested = True
		# start running operation
		if self.psum_load:
			self.next_state = "loading"
			#self.set_state("loading")
		else:
			self.next_state = "computing"
			#self.set_state("computing")
		
		# advance to the next non-zero element
		if not (self.sparse_map is None):
			finished_comp = False
			while self.sparse_map[self.x_loop, self.z_loop] == 0 and not finished_comp:
				self.out_buffer[self.x_loop][self.z_loop] = True
				finished_comp = self.inc_loop()
				#print(self.x_loop, self.z_loop, self.sparse_map.shape)
	
	# Sends 1 bitwidth worth of data from this Node to the requester & resets the ready signal
	def send_data(self, port_name, amount):
		assert port_name == "out"
		
		num_counted = amount
		
		# loop backwards over out buffer, marking off first amount True elements
		for x_val in reversed(range(self.x)):
			for z_val in reversed(range(self.z)):
				if self.out_buffer[x_val][z_val]:
					self.out_buffer[x_val][z_val] = False
					self.current_out_buffer -= 1
					num_counted -= 1
					self.num_out_reads += 1
					if num_counted == 0:
						break
			if num_counted == 0:
				break
		
		assert num_counted == 0
	
	# Receive up to 1 bitwidth worth of data from the source to this node
	def receive_data(self, port_name, amount):
		#print(self.name, port_name, amount, self.current_B_buffer, self.x, self.y, self.z)
		num_counted = amount
		
		# loop forwards over buffer, marking first amount False elements
		if port_name == "in_A":
			for x_val in range(self.x):
				for y_val in range(self.y):
					if not self.A_buffer[x_val][y_val]:
						self.A_buffer[x_val][y_val] = True
						self.current_A_buffer += 1
						num_counted -= 1
						self.num_bank_writes += 1
						if num_counted == 0:
							break
				if num_counted == 0:
					break
		
		elif port_name == "in_B":
			for z_val in range(self.z):
				for y_val in range(self.y):
					if not self.B_buffer[z_val][y_val]:
						self.B_buffer[z_val][y_val] = True
						self.current_B_buffer += 1
						num_counted -= 1
						self.num_bank_writes += 1
						if num_counted == 0:
							break
				if num_counted == 0:
					break
		
		elif port_name == "in_psums":
			for x_val in range(self.x):
				for z_val in range(self.z):
					if not self.out_buffer[x_val][z_val]:
						self.out_buffer[x_val][z_val] = True
						#print("Fixed_out_buffer_size:", self.fixed_out_buffer_size)
						#if not self.fixed_out_buffer_size:
						self.current_out_buffer += 1
						num_counted -= 1
						self.num_out_writes += 1
						if num_counted == 0:
							break
				if num_counted == 0:
					break
		
		else:
			raise ValueError("Port should either be in_A or in_B")
		#if num_counted != 0:
			#print(self.name, port_name, num_counted)
		assert num_counted == 0

	# returns "True" if it's done
	def inc_loop(self):
		self.y_loop += 1
		if self.y_loop >= self.y:
			self.y_loop = 0
			self.num_out_writes += 1 # increment number of writes, because one value has been written to memory
			if not self.fixed_out_buffer_size:
				self.current_out_buffer += 1
			self.z_loop += 1
			if self.z_loop >= self.z:
				self.z_loop = 0
				self.x_loop += 1
				if self.x_loop >= self.x:
					self.x_loop = 0
					return True
		return False

	def pre_step(self, current_cycle):
		
		self.set_flag("finished", False)
		self.set_state(self.next_state)
		
		if self.get_state() == "computing":
			
			if not (self.sparse_map is None):
				# start at a non-zero element; compute if it's loaded
				if self.A_buffer[self.x_loop][self.y_loop] and self.B_buffer[self.z_loop][self.y_loop]: # and (not self.psum_load or self.out_buffer[self.x_loop][self.z_loop]):
					self.set_flag("computing", True)
					self.num_computations += 1 # increment number of computations
					self.num_bank_reads += 2 # increment number of reads
					self.out_buffer[self.x_loop][self.z_loop] = True # mark output as available
					
					# advance to the next non-zero element
					finished_comp = self.inc_loop()
					while self.sparse_map[self.x_loop, self.z_loop] == 0 and not finished_comp:
						self.out_buffer[self.x_loop][self.z_loop] = True
						finished_comp = self.inc_loop()
					
					# if there's no more non-zero elements, flip to the next state
					if finished_comp:
						# transition to next state
						if self.psum_flush:
							if True: # no flushing -- each element is flushed as soon as it is computed
								for i in range(self.x):
									for j in range(self.z):
										self.out_buffer[i][j] = True
								self.next_state = "idle"
								#self.set_state("idle")
								self.set_flag("finished", True)
							else:
								self.next_state = "flushing"
								#self.set_state("flushing")
						else:
							self.next_state = "idle"
							#self.set_state("idle")
							self.set_flag("finished", True)
				else:
					self.set_flag("computing", False)
			else:
				# compute, if able
				#print(self.out_buffer[self.x_loop][self.z_loop])
				if self.A_buffer[self.x_loop][self.y_loop] and self.B_buffer[self.z_loop][self.y_loop]: # and (not self.psum_load or self.out_buffer[self.x_loop][self.z_loop]):
					self.set_flag("computing", True)
					#if self.x_loop == 0 and self.y_loop == 0 and self.z_loop == 0 and self.name == "PE_99":
					#	print(self.name, "is starting comp at", current_cycle)
					self.num_computations += 1 # increment number of computations
					self.num_bank_reads += 2 # increment number of reads
					self.out_buffer[self.x_loop][self.z_loop] = True # mark output as available
					# step through computing loops
					self.y_loop += 1
					if self.y_loop >= self.y:
						self.y_loop = 0
						self.num_out_writes += 1 # increment number of writes, because one value has been written to memory
						#print("Fixed_out_buffer_size:", self.fixed_out_buffer_size)
						if not self.fixed_out_buffer_size:
							self.current_out_buffer += 1
						self.z_loop += 1
						if self.z_loop >= self.z: #* ((1 - self.sparsity)):
							self.z_loop = 0
							self.x_loop += 1
							if abs(self.x_loop - self.x * (1 - self.sparsity)) < 0.1 or self.x_loop >= self.x * (1 - self.sparsity): #self.x_loop >= int(self.x * ((1 - self.sparsity))):
								self.x_loop = 0
								
								# fix sparsity variables
								if self.sparsity > 0.0:
									for i in range(self.x):
										for j in range(self.z):
											self.out_buffer[i][j] = True
								
								# transition to next state
								if self.psum_flush:
									if True: # no flushing -- each element is flushed as soon as it is computed
										for i in range(self.x):
											for j in range(self.z):
												self.out_buffer[i][j] = True
										self.next_state = "idle"
										#self.set_state("idle")
										self.set_flag("finished", True)
									else:
										self.next_state = "flushing"
										#self.set_state("flushing")
								else:
									self.next_state = "idle"
									#self.set_state("idle")
									self.set_flag("finished", True)
				else:
					self.set_flag("computing", False)
					#print(self.A_buffer[self.x_loop][self.y_loop], self.B_buffer[self.z_loop][self.y_loop])
		# check for transition from loading to computing
		elif self.get_state() == "loading":
			done = True
			for i in range(self.x):
				for j in range(self.z):
					if self.out_buffer[i][j] == False:
						done = False
						break
				if not done:
					break
			if done:
				self.next_state = "idle"
				#self.set_state("computing")
		
		# check for transition from flusing to idle
		elif self.get_state() == "flushing":
			done = True
			for i in range(self.x):
				for j in range(self.z):
					if self.out_buffer[i][j] == True:
						done = False
						break
				if not done:
					break
			if done:
				#self.set_state("idle")
				self.next_state = "idle"
				self.set_flag("finished", True)
		
	
	def step_request(self, current_cycle):
		
		if self.get_state() == "computing":
			# request data for each buffer, if it hasn't already been requested
			if not self.A_requested:
				self.ports["in_A"].act(self.x * self.y, current_cycle)
				self.A_requested = True
			if not self.B_requested:
				self.ports["in_B"].act(self.z * self.y, current_cycle)
				self.B_requested = True
		elif self.get_state() == "loading":
			if self.psum_load and not self.psums_load_requested: # only request data for psum buffer if psum_load is True
				self.ports["in_psums"].act(self.x * self.z, current_cycle)
				self.psums_load_requested = True
				#print("requesting data")
		elif self.get_state() == "flushing":
			# request to push data, if it hasn't already been requested (psum_flush check should be redundant)
			if self.psum_flush and not self.psums_write_requested:
				self.ports["out"].act(self.x * self.z, current_cycle)
				self.psums_write_requested = True

	def step_ready(self, current_cycle):
		# does nothing -- ComplexPENode actively controls all of its ports
		return

	def print_status(self):
		print("--", self.name, "--")
		print("State:", self.state)
		print("x Loop Counter:", self.x_loop)
		print("y Loop Counter:", self.y_loop)
		print("z Loop Counter:", self.z_loop)
		print("Current A Buffer Size:", self.current_A_buffer)
		print("Current B Buffer Size:", self.current_B_buffer)
		print("Current Out Buffer Size:", self.current_out_buffer)
		# maybe add something to compute and print out number of elements in each buffer?