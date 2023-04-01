from .types import Node, Port

class MemoryControllerNode(Node):

	# on_chip_bandwidth is bandwidth between the memory controller and on-chip nodes
	# off_chip_bandwidth is bandwidth between the memory controller and DRAM
	# DRAM ports are called "DRAM_in" and "DRAM_out"
	# the number and name of on-chip ports are given in "input_port_names" and "output_port_names"
	# share_load is "A" if elements of A are loaded by each PE lane, "B" if elements of B are loaded by each PE lane, and "None" if there is no shared loading
	def __init__(self, name, off_chip_bandwidth, on_chip_bandwidth, input_port_names, output_port_names, share_load="None", num_PE_lanes=1, load_immediate=False, store_immediate=False):
		# create ports & initialize node
		input_ports = [Port(name, False, False, on_chip_bandwidth) for name in input_port_names]
		output_ports = [Port(name, True, False, on_chip_bandwidth) for name in output_port_names]
		DRAM_ports = [Port("DRAM_in", False, True, off_chip_bandwidth), Port("DRAM_out", True, True, off_chip_bandwidth)]
		#DRAM_ports = [Port("DRAM_in", False, True, int(off_chip_bandwidth * percent_bandwidth_in)), Port("DRAM_out", True, True, int(off_chip_bandwidth * (1 - percent_bandwidth_in)))]
		#print(output_port_names)
		Node.__init__(self, name, DRAM_ports + input_ports + output_ports)
		
		self.off_chip_bandwidth = off_chip_bandwidth
		
		self.share_load = share_load
		self.num_PE_lanes = num_PE_lanes
		
		self.load_immediate = load_immediate
		self.store_immediate = store_immediate
		
		self.chunks = {} # chunk name to chunk object
		
		self.DRAM_in_chunk = None
		self.DRAM_out_chunk = None
		
		self.port_chunks = {} # port name to chunk object
		self.load_chunk_queue = []
		self.save_chunk_queue = []
		
		self.save_chunk_requests = [] # list of chunk names
		
		# buffer data
		self.min_buffer_size = 0
		self.min_buffer_banks = 0
		
		# stats
		self.reads = 0
		self.writes = 0
		
		self.dram_reads = 0
		self.dram_writes = 0
		
		self.ideal_input_bandwidth_used = 0
		self.actual_input_bandwidth_used = 0
		self.ideal_output_bandwidth_used = 0
		self.actual_output_bandwidth_used = 0
		
		# flags
		self.set_flag("idle", "False")
		
		self.in_turn = True
	
	# adds a chunk with the following parameters if it exists
	def add_chunk(self, chunk_name, size, source_is_dram=True, remove_on_read=False, cap_size=False, transfer_scale=1):
		# add chunk, ignoring duplicates
		if not chunk_name in self.chunks.keys():
			#print("Added chunk:", chunk_name)
			self.chunks[chunk_name] = Chunk(chunk_name, size, source_is_dram, remove_on_read, cap_size, transfer_scale)
	
	def update_chunk(self, chunk_name, source_is_dram=True, remove_on_read=False, cap_size=False):
		self.chunks[chunk_name].source_is_dram = source_is_dram
		self.chunks[chunk_name].remove_on_read = remove_on_read
		self.chunks[chunk_name].cap_size = cap_size
	
	def set_chunk_for_port(self, port_name, chunk_name):
		self.port_chunks[port_name] = self.chunks[chunk_name]
		#print("Set chunk for port:", port_name, chunk_name)
	
	def give_save_chunk_request(self, chunk_name):
		#del self.chunks[chunk_name]
		#assert self.chunks[chunk_name].full == True
		if not chunk_name in self.save_chunk_requests:
			#if chunk_name == "O_38_81":
			#	print("Saving chunk O_38_81")
			self.save_chunk_requests.append(chunk_name)
	
	def erase_chunk(self, chunk_name):
		#self.chunks[chunk_name].erase()
		if chunk_name in self.chunks:
			del self.chunks[chunk_name]
	
	def reformat_chunks(self, old_chunks, new_chunks, new_chunk_dim, source_is_dram=False, remove_on_read=True):
		size = 0
		for chunk_name in old_chunks:
			assert self.chunks[chunk_name].full
			size += self.chunks[chunk_name].current_size
			self.erase_chunk(chunk_name)
		#print(size, len(new_chunks), new_chunk_dim)
		#assert size == len(new_chunks) * new_chunk_dim
		
		for chunk_name in new_chunks:
			self.add_chunk(chunk_name, new_chunk_dim)
			self.chunks[chunk_name].add_data(new_chunk_dim)
		
		#print("Reformatted chunks")
		#print("Deleted:", old_chunks)
		#print("Added:", new_chunks)
		#self.print_status()
	
	def send_data(self, port_name, amount):
		self.reads += amount
		
		#print(port_name)
		#print(self.ports[port_name].is_source)
		# if port_name is DRAM_out
		if port_name == "DRAM_out":
			self.actual_output_bandwidth_used += amount
			self.dram_writes += amount
			
			while amount > 0:
				if len(self.save_chunk_queue) == 0:
					return
				#assert len(self.save_chunk_queue) > 0
				chunk = self.save_chunk_queue[0]
				amt_to_remove = min(amount, chunk.current_size * chunk.transfer_scale)
				chunk.remove_data(amt_to_remove)
				if chunk.empty:
					del self.save_chunk_queue[0]
				amount -= amt_to_remove
			
			'''
			# grab the corresponding chunk and tell it that "amount" has been removed from it
			self.DRAM_out_chunk.remove_data(amount)
			if self.DRAM_out_chunk.empty:
			#	print("removing", self.DRAM_out_chunk.name)
				#print("Removed", self.DRAM_out_chunk.name)
				del self.chunks[self.DRAM_out_chunk.name]
			'''
		# otherwise...
		elif self.port_chunks[port_name].remove_on_read:
			#if self.port_chunks[port_name].name == "O_0_0":
			#	print("Sending data from O_0_0 at request of port", port_name)
			# get the chunk corresponding to that port and tell it that "amount" has been removed from it
			self.port_chunks[port_name].remove_data(amount)
	
	def receive_data(self, port_name, amount):
		self.writes += amount
		
		# if port_name is DRAM_in
		if port_name == "DRAM_in":
			#print("Total amount received:", amount, "amount left:")
			self.actual_input_bandwidth_used += amount
			while amount > 0:
				if len(self.load_chunk_queue) == 0:
					#print("Load queue is empty with", amount, "left")
					return
				#assert len(self.load_chunk_queue) > 0
				chunk = self.load_chunk_queue[0]
				if chunk.name[0] == self.share_load: # if this is part of matrix B, then increase amount loaded by # of PE lanes times
					#print(chunk.max_size, chunk.current_size, amount)
					if amount * self.num_PE_lanes / chunk.transfer_scale < (chunk.max_size - chunk.current_size):
						#if chunk.name == "B_0_0":
						#	print(amount * self.num_PE_lanes / chunk.transfer_scale)
						#	print("Amount left:", self.ports["DRAM_in"].get_elements_requested())
						amt_to_add = amount
						chunk.add_data(amt_to_add * self.num_PE_lanes)
					else:
						amt_to_add = (chunk.max_size - chunk.current_size) * chunk.transfer_scale / self.num_PE_lanes
						chunk.add_data(amt_to_add * self.num_PE_lanes)
					#print(amt_to_add, chunk.current_size)
					#amt_to_add = min(amount, (chunk.max_size - chunk.current_size) // 8)
					self.dram_reads += amt_to_add
				else:
					amt_to_add = min(amount, (chunk.max_size - chunk.current_size) * chunk.transfer_scale)
					chunk.add_data(amt_to_add)
					self.dram_reads += amt_to_add * self.num_PE_lanes
					#print(self.num_PE_lanes)
				if chunk.full:
					del self.load_chunk_queue[0]
				amount -= amt_to_add
			
			# grab the corresponding port and tell it that "amount" has been added
			#self.DRAM_in_chunk.add_data(amount)
		# else
		else:
			# grab the corresponding port and tell it that "amount" has been added
			self.port_chunks[port_name].add_data(amount)
	
	def step_request(self, current_cycle):
		
		# v3, new behavior: all of the DRAM bandwidth is used to load or all is used to store. Loading takes priority over storing
		
		if self.load_immediate:
			for chunk in self.chunks.values():
				if chunk.empty and chunk.source_is_dram:
					chunk.make_full()
		
		if not self.ports["DRAM_in"].is_current_request() and not self.ports["DRAM_out"].is_current_request():
			if self.in_turn:
				
				# first check any input requests
				if len(self.load_chunk_queue) > 0:
					#for chunk in self.load_chunk_queue:
					#	chunk.make_full()
					#self.load_chunk_queue.clear()
					print(self.load_chunk_queue[0].name)
					print(self.load_chunk_queue[0].current_size, self.load_chunk_queue[0].max_size)
				assert len(self.load_chunk_queue) == 0
				total_size = 0
				for chunk in self.chunks.values():
					if chunk.empty and chunk.source_is_dram:
						assert chunk.remove_on_read == False
						self.load_chunk_queue.append(chunk)
						if chunk.name[0] == self.share_load:
							total_size += chunk.max_size * chunk.transfer_scale / self.num_PE_lanes
						else:
							total_size += chunk.max_size * chunk.transfer_scale
				if total_size != 0:
					self.ports["DRAM_in"].act(total_size, current_cycle)
				else:
					if self.store_immediate:
						for chunk_name in self.save_chunk_requests:
							chunk = self.chunks[chunk_name]
							if chunk_name[0] == "O":
								chunk.make_full()
							chunk.remove_data(chunk.current_size)
						self.save_chunk_requests.clear()
					# then, if there are no input requests, offload outputs:
					assert len(self.save_chunk_queue) == 0
					total_size = 0
					for chunk_name in self.save_chunk_requests:
						chunk = self.chunks[chunk_name]
						if chunk_name[0] == "O":
							chunk.make_full()
							#chunk.current_size = chunk.max_size
							#chunk.full = True
						if not chunk.full:
							print(chunk.name, chunk.current_size, chunk.max_size)
						assert chunk.full
						total_size += chunk.current_size * chunk.transfer_scale
						self.save_chunk_queue.append(chunk)
						#if total_size > 1000:
						#	break
					for chunk in self.save_chunk_queue:
						if chunk.name in self.save_chunk_requests:
							self.save_chunk_requests.remove(chunk.name)
					#del self.save_chunk_requests[0:count]
					#self.save_chunk_requests.clear()
					self.ports["DRAM_out"].act(total_size, current_cycle)
			else:
				if self.store_immediate:
					for chunk_name in self.save_chunk_requests:
						chunk = self.chunks[chunk_name]
						if chunk_name[0] == "O":
							chunk.make_full()
						chunk.remove_data(chunk.current_size)
					self.save_chunk_requests.clear()
				assert len(self.save_chunk_queue) == 0
				total_size = 0
				for chunk_name in self.save_chunk_requests:
					chunk = self.chunks[chunk_name]
					# quick fix while removing flushing from PEs
					if chunk_name[0] == "O":
						chunk.make_full()
					#	chunk.current_size = chunk.max_size
					#	chunk.full = True
					assert chunk.full
					total_size += chunk.current_size * chunk.transfer_scale
					self.save_chunk_queue.append(chunk)
				self.save_chunk_requests.clear()
				if total_size != 0:
					self.ports["DRAM_out"].act(total_size, current_cycle)
				else:
					if self.load_immediate:
						for chunk in self.chunks.values():
							if chunk.empty and chunk.source_is_dram:
								chunk.make_full()
					
					if len(self.load_chunk_queue) > 0:
						#for chunk in self.load_chunk_queue:
						#	chunk.make_full()
						#self.load_chunk_queue.clear()
						print(self.load_chunk_queue[0].name, len(self.load_chunk_queue), self.load_chunk_queue[0].transfer_scale)
						print(self.load_chunk_queue[0].current_size, self.load_chunk_queue[0].max_size)
						print(self.ports["DRAM_in"].is_current_request())
					assert len(self.load_chunk_queue) == 0
					total_size = 0
					for chunk in self.chunks.values():
						if chunk.empty and chunk.source_is_dram:
							#if chunk.name == "B_0_140":
							#	print("B_0_140 is getting loaded!")
							assert chunk.remove_on_read == False
							self.load_chunk_queue.append(chunk)
							if chunk.name[0] == self.share_load:
								total_size += chunk.max_size * chunk.transfer_scale / self.num_PE_lanes
								#if chunk.name == "B_0_0":
								#	print("Amount requested:", chunk.max_size * chunk.transfer_scale / self.num_PE_lanes)
							else:
								total_size += chunk.max_size * chunk.transfer_scale
					self.ports["DRAM_in"].act(total_size, current_cycle)
			
			self.in_turn = not self.in_turn
		
		'''
		# v2
		# if dram_in isn't in use...
		if not self.ports["DRAM_in"].is_current_request():
			assert len(self.load_chunk_queue) == 0
			total_size = 0
			for chunk in self.chunks.values():
				if chunk.empty and chunk.source_is_dram:
					#if chunk.name == "B_0_140":
					#	print("B_0_140 is getting loaded!")
					assert chunk.remove_on_read == False
					self.load_chunk_queue.append(chunk)
					total_size += chunk.max_size
			#if total_size != 0:
			#	print(total_size)
			self.ports["DRAM_in"].act(total_size, current_cycle)
		
		if not self.ports["DRAM_out"].is_current_request():
			assert len(self.save_chunk_queue) == 0
			total_size = 0
			for chunk_name in self.save_chunk_requests:
				chunk = self.chunks[chunk_name]
				assert chunk.full
				total_size += chunk.current_size
				self.save_chunk_queue.append(chunk)
			self.save_chunk_requests.clear()
			self.ports["DRAM_out"].act(total_size, current_cycle)
		'''
		
		'''
		# v1
		# if dram_in isn't in use...
		if not self.ports["DRAM_in"].is_current_request():
			self.DRAM_in_chunk = None
			# for each port...
			for port in self.ports.values():
				# if the port isn't DRAM:
				if port.name != "DRAM_out":
					# if the port has an active request...
					#if port.is_current_request():
					# if the port is an output port...
					if port.is_source:
						# if the corresponding chunk isn't full and hasn't already been requested...
						if not self.port_chunks[port.name].full and not self.port_chunks[port.name].requested:
							# if the chunk is sourced from dram...
							if self.port_chunks[port.name].source_is_dram:
								# request data from DRAM for that chunk
								self.DRAM_in_chunk = self.port_chunks[port.name]
								self.ports["DRAM_in"].act(self.port_chunks[port.name].max_size, current_cycle)
								self.port_chunks[port.name].requested = True
								break
		'''
		
		'''
		# if dram_out isn't in use...
		if not self.ports["DRAM_out"].is_current_request():
			self.DRAM_out_chunk = None
			
			# get the first chunk request where the corresponding chunk is full
			chunk = None
			for request in self.save_chunk_requests:
				if self.chunks[request].full:
					chunk = self.chunks[request]
					break
			
			if chunk != None:
				self.save_chunk_requests.remove(chunk.name)
				self.DRAM_out_chunk = chunk
				self.ports["DRAM_out"].act(chunk.max_size, current_cycle)
			
			
			# grab the first store command & store the corresponding chunk
			#if len(self.save_chunk_requests) > 0:
			#	chunk_name = self.save_chunk_requests.pop(0)
			#	self.DRAM_out_chunk = self.chunks[chunk_name]
			#	#if chunk_name == "O_0_0":
			#	#	print("Requesting remove O_0_0")
			#	self.ports["DRAM_out"].act(self.chunks[chunk_name].max_size, current_cycle)
			
		'''
	
	def step_ready(self, current_cycle):
		
		num_banks_used = 0
		
		# for each port
		for port in self.ports.values():
			# if the port isn't DRAM_in or DRAM_out
			if port.name != "DRAM_in" and port.name != "DRAM_out":
				# if the port has an active request...
				if port.is_current_request():
					
					# if it's an input port, flip ready
					if not port.is_source:
						port.act_max(current_cycle)
					# if it's an output port and the corresponding chunk is full, flip ready (OR if the chunk is remove_on_read, then remove it as it's read -- remove_on_read assumes that only one port uses it)
					elif self.port_chunks[port.name].full or self.port_chunks[port.name].remove_on_read:
						#if self.port_chunks[port.name].name == "B_0_140":
						#	print("sending data to B_0_140")
						#if port.name == "out_psums_0":
						#	print(self.port_chunks[port.name].current_size)
						#	print(port.transmit_rate)
						#	print(port.get_elements_requested())
						#print(port.name)
						#print(port.get_elements_requested())
						#print(self.port_chunks[port.name].current_size)
						#print(port.transmit_rate)
						#print("here", port.name)
						amount_to_transmit = min(self.port_chunks[port.name].current_size, max(1, port.transmit_rate), port.get_elements_requested())
						if amount_to_transmit > 0:
							num_banks_used += 1
							port.act(amount_to_transmit, current_cycle)
		
		if num_banks_used > self.min_buffer_banks:
			self.min_buffer_banks = num_banks_used

	def post_step(self, current_cycle):
		# handle idle
		self.set_flag("idle", "True")
		#for port in self.ports.values():
		#	if port.is_current_request():
		#		self.set_flag("idle", "False")
		#		break
		if len(self.save_chunk_requests) > 0:
			self.set_flag("idle", "False")
		for name, chunk in self.chunks.items():
			if not chunk.empty:
				self.set_flag("idle", "False")
		
		# track max size
		current_size = 0
		for name, chunk in self.chunks.items():
			current_size += chunk.current_size
		if current_size > self.min_buffer_size:
			self.min_buffer_size = current_size
		
		# track ideal bandwidth utilization
		self.ideal_input_bandwidth_used += self.off_chip_bandwidth
		self.ideal_output_bandwidth_used += self.off_chip_bandwidth
	
	def print_status(self):
		print("--", self.name, "--")
		for name, chunk in self.chunks.items():
			if not chunk.empty:
				chunk.print_status()
	
	def get_min_buffer_size(self):
		return self.min_buffer_size
	
	def get_min_buffer_banks(self):
		return self.min_buffer_banks

# transfer_scale is used to adjust the size of the chunk while loading; for example, when using encoder/decoder w/ 50% compression then the transfer_scale = 0.5
class Chunk:
	def __init__(self, name, max_size, source_is_dram=True, remove_on_read=False, cap_size=False, transfer_scale=1):
		self.name = name
		self.max_size = max_size
		self.current_size = 0
		self.full = False
		self.empty = True
		self.source_is_dram = source_is_dram
		self.remove_on_read = remove_on_read
		self.cap_size = cap_size
		self.transfer_scale = transfer_scale
		#self.transfer_size = self.max_size * transfer_scale
		
		self.requested = False
	
	def remove_data(self, amount):
		self.current_size -= amount / self.transfer_scale
		if self.current_size < 0.01:
			self.current_size = 0
		#if self.name == "O_0_0":
			#print("removing data:", amount)
		if self.current_size < 0:
			self.print_status()
			#print(amount)
		assert self.current_size >= 0
		if self.current_size < self.max_size:
			self.full = False
		if self.current_size == 0:
			self.empty = True
			self.requested = False
	
	def add_data(self, amount):
		#if self.name == "B_0_0":
		#	print("In chunk:", self.current_size, self.max_size, amount, amount / self.transfer_scale, self.current_size + amount / self.transfer_scale)
		self.current_size += amount / self.transfer_scale
		# deal with floating point addition
		if self.max_size - self.current_size < 0.01:
			self.current_size = self.max_size
		if self.cap_size and self.current_size > self.max_size:
			self.current_size = self.max_size
		#if self.name == "O_0_0":
		#	print("receiving data:", amount)
		#print(self.name, self.max_size, self.current_size, amount)
		assert self.current_size <= self.max_size
		if self.current_size > 0:
			self.empty = False
		if self.current_size == self.max_size:
			self.full = True
			self.source_is_dram = False # once it's been filled from DRAM, don't ask to be filled from DRAM anymore
	
	def make_full(self):
		self.current_size = self.max_size
		self.empty = False
		self.full = True
		self.source_is_dram = False
	
	def erase(self):
		self.current_size = 0
		self.empty = True
		self.full = False
		self.requested = False
	
	def print_status(self):
		print(self.name, ":", self.current_size, "out of", self.max_size)
