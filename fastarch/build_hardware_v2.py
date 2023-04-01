import math
import random

class Hardware:

	# tiling_generator is a function that is called in order to determine the tiling parameters given this hardware and an input layer
	def __init__(self, num_PE_lanes, num_PEs_per_lane, num_RFs_per_PE, size_RF, off_chip_bandwidth, on_chip_bandwidth, total_sram_size):
		self.num_PE_lanes = num_PE_lanes
		self.num_PEs_per_lane = num_PEs_per_lane
		self.num_RFs_per_PE = num_RFs_per_PE
		self.size_RF = size_RF
		self.off_chip_bandwidth = off_chip_bandwidth
		self.on_chip_bandwidth = on_chip_bandwidth
		self.total_sram_size = total_sram_size
	
	def get_single_lane_bandwidth(self):
		return self.off_chip_bandwidth / self.num_PE_lanes
	
	def get_single_lane_SRAM_size(self):
		return self.total_sram_size // self.num_PE_lanes
	
	def print(self):
		print("Hardware Description:")
		print("Num PE lanes:", self.num_PE_lanes)
		print("Num PEs per lane:", self.num_PEs_per_lane)
		print("Num RFs per PE:", self.num_RFs_per_PE)
		print("Size RF:", self.size_RF)
		print("Off-chip bandwidth:", self.off_chip_bandwidth)
		print("On-chip bandwidth:", self.on_chip_bandwidth)
		print("Total SRAM size:", self.total_sram_size)
	
	def get_string(self):
		descrip = "Hardware Description:\n"
		descrip += "Num PE lanes: " + str(self.num_PE_lanes) + "\n"
		descrip += "Num PEs per lane: " + str(self.num_PEs_per_lane) + "\n"
		descrip += "Num RFs per PEs: " + str(self.num_RFs_per_PE) + "\n"
		descrip += "Size RF: " + str(self.size_RF) + "\n"
		descrip += "Off-chip bandwidth: " + str(self.off_chip_bandwidth) + "\n"
		descrip += "On-chip bandwidth: " + str(self.on_chip_bandwidth) + "\n"
		descrip += "Total SRAM size: " + str(self.total_sram_size) + "\n"
		return descrip

def test_tiling_generator(num_PEs, num_RFs_per_PE, size_RF, off_chip_bandwidth, on_chip_bandwidth, max_sram_size, layer):
	params = ['Output-Stationary', 7, 1, 1, 1, 1, 1]
	layer.print()
	print(params)
	return params

def random_tiling_generator(num_PEs, num_RFs_per_PE, size_RF, off_chip_bandwidth, on_chip_bandwidth, max_sram_size, layer):
	t_a_options = list(factors(layer.A_rows)) #list(range(1, layer.A_rows+1))
	t_w_options = list(factors(layer.A_cols_B_rows)) #list(range(1, layer.A_cols_B_rows+1))
	t_b_options = list(factors(layer.B_cols)) #list(range(1, layer.B_cols+1))
	
	if layer.encode:
		orig = list(factors(layer.B_cols))
		t_b_options = []
		for i in orig:
			if i % layer.orig_head_dim == 0:
				t_b_options.append(i)
	
	if layer.decode:
		orig = list(factors(layer.A_cols_B_rows))
		t_w_options = []
		for i in orig:
			if i % layer.orig_head_dim == 0:
				t_w_options.append(i)
			#print(t_w_options)
	
	while True:
		t_a = random.choice(t_a_options)
		t_b = random.choice(t_b_options)
		t_w = int(math.floor((max_sram_size - t_a * t_b) / (t_a + t_b)))
		total = t_a * t_w + t_b * t_w + t_a * t_b
		while total > max_sram_size or t_w % layer.orig_head_dim != 0:
			#t_w = random.choice(t_w_options) #int(math.floor((max_sram_size - t_a * t_b) / (t_a + t_b)))
			t_w -= 1
			total = t_a * t_w + t_b * t_w + t_a * t_b
			#print(t_a, t_b, t_w, max_sram_size, total)
		
		if t_w > 0:
			break
	
	c_a_options = list(factors(t_a)) #list(range(1, t_a+1))
	c_b_options = list(factors(t_b)) #list(range(1, t_b+1))
	total = num_RFs_per_PE + 1
	while total > num_RFs_per_PE:
		c_a = 1 #random.choice(c_a_options)
		c_b = 1 #random.choice(c_b_options)
		total = c_a + c_b
	
	c_w = min(t_w, size_RF)
	
	if layer.decode:
		# make sure c_w evenly divides t_w for decoding purposes
		while t_w % c_w != 0:
			c_w -= 1
	
	'''
	c_w = 1
	for i in list(factors(t_w)):
		if i <= size_RF and i > c_w:
			c_w = i
	#c_w = min(t_w, size_RF)
	'''
	
	params = ["Output-Stationary", t_a, t_b, t_w, c_a, c_b, c_w]
	print(params)
	return params

def simple_tiling_generator(num_PEs, num_RFs_per_PE, size_RF, off_chip_bandwidth, on_chip_bandwidth, max_sram_size, layer):
	return ["Output-Stationary", 10, 10, 10, 1, 1, 1]

def factors(n):
    results = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            results.add(i)
            results.add(int(n/i))
    return results

def complex_tiling_generator(num_PEs, num_banks_per_PE, size_banks_per_PE, off_chip_bandwidth, on_chip_bandwidth, total_memory, layer):
	A_rows = layer.A_rows
	B_cols = layer.B_cols
	A_cols_B_rows = layer.A_cols_B_rows
	
	memory_step_size = 100
	ideal_size = A_rows * B_cols * A_cols_B_rows
	best = 0
	best_dim = []
	
	A_rows_factors = factors(A_rows)
	B_cols_factors = factors(B_cols)
	A_cols_B_rows_factors = factors(A_cols_B_rows)
	#c_w = max(factors(size_banks_per_PE))
	
	count = 0
	
	pe_weight = 10
	tile_weight = 1 / (A_rows * B_cols * A_cols_B_rows)
	chunk_weight = 10 / (num_banks_per_PE * size_banks_per_PE)
	
	for t_a in A_rows_factors:
		t_a_factors = factors(t_a)
		for t_b in B_cols_factors:
			t_b_factors = factors(t_b)
			for t_w in A_cols_B_rows_factors:
				if t_a * t_w + t_b * t_w + t_a * t_b > total_memory:
					continue
				c_w = 1
				for i in range(size_banks_per_PE, -1, -1):
					if t_w % i == 0:
						c_w = i
						break
				for x in range(1, num_banks_per_PE):
					z = num_banks_per_PE - x
					for c_a in t_a_factors:
						if c_a > x:
							continue
						for c_b in t_b_factors:
							if c_b > z:
								continue
							count += 1
							
							# for output stationary inner dataflow
							num = ((t_a // c_a) * (t_b // c_b)) % num_PEs
							#print(num)
							tile_size = t_a * t_b * t_w
							chunk_size = c_a * c_b * c_w
							
							score = -num * pe_weight + tile_size * tile_weight + chunk_size * chunk_weight
							
							if score > best:
								best = score
								best_dim = [t_a, t_b, t_w, c_a, c_b, c_w]
							
	print("Count:", count)
	print(best, best_dim, ideal_size)
	#print(calc_actual_dims(best_dim[0], best_dim[1], best_dim[2], best_dim[3], best_dim[4], best_dim[5], A_rows, B_cols, A_cols_B_rows))
	return ['B-Stationary'] + best_dim

def generate_random_config(num_PEs, total_on_chip_memory, off_chip_bandwidth, on_chip_bandwidth, tiling_generator):
	potential_num_RFs = [4] #list(range(9, 55))
	potential_size_RF = [49] #list(range(8, 49))
	num_RFs = random.choice(potential_num_RFs)
	size_RF = random.choice(potential_size_RF)
	size_sram = total_on_chip_memory - num_RFs * size_RF
	return Hardware(num_PEs, num_RFs, size_RF, off_chip_bandwidth, on_chip_bandwidth, size_sram, tiling_generator)


# generate random possibilities for each combination (aka for each group of combinations)

# on-chip memory (sets num_RFs_per_PE, size_RF, buffer_size)
# Constraints:
# 1. if there's a decoding layer: num_RFs_per_PE >= 1 + comp_heads and size_RF >= orig_heads
# 2. if there's an encoding layer: num_RFs_per_PE >= 1 + orig_heads and size_RF >= comp_heads
# 3. num_RFs_per_PE * size_RF * num_PEs + buffer_size <= total_on_chip_memory
def get_on_chip_memory(num_PEs, total_on_chip_memory, layers):
	# layers is a LayerSet
	
	smallest_num_RFs = 5
	smallest_size_RFs = 5
	for layer in layers.unique_layers:
		if layer[0].encode:
			if smallest_num_RFs < 1 + layer[0].orig_head_dim:
				smallest_num_RFs = 1 + layer[0].orig_head_dim
			if smallest_size_RFs < layer[0].comp_head_dim:
				smallest_size_RFs = layer[0].comp_head_dim
		if layer[0].decode:
			if smallest_num_RFs < 1 + layer[0].comp_head_dim:
				smallest_num_RFs = 1 + layer[0].comp_head_dim
			if smallest_size_RFs < layer[0].orig_head_dim:
				smallest_size_RFs = layer[0].orig_head_dim
	
	largest_num_RFs = 20
	largest_size_RFs = 20
	
	while True:
		num_RFs = random.choice(list(range(smallest_num_RFs, largest_num_RFs + 1)))
		size_RFs = random.choice(list(range(smallest_size_RFs, largest_size_RFs + 1)))
		
		if num_RFs * size_RFs * num_PEs <= total_on_chip_memory - 1000: # make sure there's enough memory for the on-chip buffer
			break
	
	#print(num_PEs, num_RFs, size_RFs)
	max_buffer_size = int(total_on_chip_memory - num_RFs * size_RFs * num_PEs)
	min_buffer_size = 1000 #min(1000, max_buffer_size)
	
	#print(min_buffer_size, max_buffer_size + 1)
	buffer_size = random.choice(list(range(min_buffer_size, max_buffer_size + 1)))
	
	if buffer_size <= 1:
		raise Exception("Buffer size is too small: " + str(buffer_size))
	
	return num_RFs, size_RFs, buffer_size

def get_potential_hardware_configs(num_PEs, total_on_chip_memory, layers):
	smallest_num_RFs = 5
	smallest_size_RFs = 5
	for layer in layers.unique_layers:
		if layer[0].encode:
			if smallest_num_RFs < 1 + layer[0].orig_head_dim:
				smallest_num_RFs = 1 + layer[0].orig_head_dim
			if smallest_size_RFs < layer[0].comp_head_dim:
				smallest_size_RFs = layer[0].comp_head_dim
		if layer[0].decode:
			if smallest_num_RFs < 1 + layer[0].comp_head_dim:
				smallest_num_RFs = 1 + layer[0].comp_head_dim
			if smallest_size_RFs < layer[0].orig_head_dim:
				smallest_size_RFs = layer[0].orig_head_dim
	
	largest_num_RFs = 20
	largest_size_RFs = 20
	
	possibilities = []
	for num_RFs in range(smallest_num_RFs, largest_num_RFs + 1):
		for size_RF in range(smallest_size_RFs, largest_size_RFs + 1):
			if num_RFs * size_RF * num_PEs <= total_on_chip_memory - 1000: # make sure there's enough memory for the on-chip buffer
				possibilities.append([num_RFs, size_RF])
	
	return possibilities

# dataflow (sets... dataflow)
# Constraints:
# 1. if encoding, dataflow must be output-stationary
# Values:
# Output-Stationary, A-Stationary, B-Stationary
def get_dataflow(layer):
	if layer.encode:
		return "Output-Stationary"
	else:
		return random.choice(["Output-Stationary", "A-Stationary", "B-Stationary"])

def get_tile_parameters(buffer_size, layer):
	best_params = []
	best_score = 0
	for i in range(100):
		params = _get_tile_parameters(buffer_size, layer)
		act_A_rows = math.ceil(layer.A_rows / params[0]) * params[0]
		act_A_cols = math.ceil(layer.A_cols_B_rows / params[2]) * params[2]
		act_B_rows = math.ceil(layer.B_cols / params[1]) * params[1]
		score = (sum(params)) / (act_A_rows * act_A_cols * act_B_rows - layer.A_rows * layer.A_cols_B_rows * layer.B_cols + 0.00000001)
		if score > best_score:
			best_params = params
			best_score = score
	
	layer.print()
	print(best_params)
	
	return best_params

# Tile parameters (sets t_a, t_b, t_w)
# Constraints:
# 1. if encoding, t_b % orig_heads == 0
# 2. if decoding, t_w % orig_heads == 0
# 3. t_a * t_w + t_b * t_w + t_a * t_b <= buffer_size
# 4. if encoding, t_b >= orig_heads and t_w >= comp_heads
# 5. if decoding, t_b >= comp_heads and t_w >= orig_heads
def _get_tile_parameters(buffer_size, layer):
	t_a_options = list(range(1, layer.A_rows + 1))
	t_b_options = list(range(1, layer.B_cols + 1))
	t_w_options = list(range(1, layer.A_cols_B_rows + 1))
	#print(t_a_options, t_b_options)
	t_a = -1
	t_b = -1
	t_w = -1
	
	searching = True
	
	while searching:
		searching = False
		
		t_a = random.choice(t_a_options)
		t_b = random.choice(t_b_options)
		t_w = random.choice(t_w_options)
		
		if layer.encode and (t_b % layer.orig_head_dim != 0 or t_b < layer.orig_head_dim or t_w < layer.comp_head_dim):
			searching = True
		elif layer.decode and (t_w % layer.orig_head_dim != 0 or t_b < layer.comp_head_dim or t_w < layer.orig_head_dim):
			#print("hit first")
			searching = True
		elif t_a * t_w + t_b * t_w + t_a * t_b > buffer_size:
			searching = True
		#print("searching tile")
	
	return [t_a, t_b, t_w]

def get_chunk_parameters(t_a, t_b, t_w, num_RFs, size_RF, layer):
	best_params = []
	best_score = 0
	for i in range(100):
		params = _get_chunk_parameters(t_a, t_b, t_w, num_RFs, size_RF, layer)
		#act_A_rows = math.ceil(t_a / params[0]) * params[0]
		#act_A_cols = math.ceil(t_w / params[2]) * params[2]
		#act_B_rows = math.ceil(t_b / params[1]) * params[1]
		#score = 1 / (act_A_rows * act_A_cols * act_B_rows - t_a * t_b * t_w + 0.00000001)
		score = sum(params)
		if score > best_score:
			best_params = params
			best_score = score
	
	#layer.print()
	#print(best_params)
	
	return best_params

# Chunk parameters (sets c_a, c_b, c_w)
# Constraints:
# 1. if decoding, t_w % c_w == 0
# 2. if encoding, c_b >= orig_heads and c_w >= comp_heads
# 3. if decoding, c_b >= comp_heads and c_w >= orig_heads
# 4. c_a <= t_a and c_b <= t_b and c_w <= t_w
# 5. c_a + c_b <= num_RFs and c_w <= size_RF
def _get_chunk_parameters(t_a, t_b, t_w, num_RFs, size_RF, layer):
	c_a_options = list(range(1, min(num_RFs, t_a + 1)))
	c_b_options = list(range(1, min(num_RFs, t_b + 1)))
	c_w_options = list(range(1, min(size_RF + 1, t_w + 1)))
	
	c_a = -1
	c_b = -1
	c_w = -1
	
	searching = True
	
	#print(c_a_options)
	#print(c_b_options)
	
	while searching:
		searching = False
		
		c_a = random.choice(c_a_options)
		c_b = random.choice(c_b_options)
		#c_w = random.choice(c_w_options)
		c_w = min(size_RF, t_w)
		if layer.encode and c_w < layer.comp_head_dim:
			c_w = layer.comp_head_dim
		if layer.decode: # and c_w < layer.orig_head_dim:
			while t_w % c_w != 0:
				c_w -= 1
			assert c_w >= layer.orig_head_dim
			#c_w = layer.orig_head_dim
		
		#print(c_a, c_b, c_w)
		#print(c_w, t_w, t_w % c_w)
		#print(t_w, c_w, c_b, layer.comp_head_dim, c_w, layer.orig_head_dim)
		if layer.encode and (c_b < layer.orig_head_dim or c_w < layer.comp_head_dim):
			searching = True
		elif layer.decode and (t_w % c_w != 0 or c_b < layer.comp_head_dim or c_w < layer.orig_head_dim):
			#print(t_w, c_w, c_b, layer.comp_head_dim, c_w, layer.orig_head_dim)
			searching = True
		elif c_a + c_b > num_RFs or c_w > size_RF:
			searching = True
			
		#raise Exception()
	
	return [c_a, c_b, c_w]