import random

import fastarch.build_models_v2 as bm
import fastarch.build_hardware_v2 as bh

# assumes step size = 1
class ConvLayer:
	def __init__(self, rows, cols, c_in, c_out, filter_dim, step_size, flags = {}):
		self.rows = rows
		self.cols = cols
		self.c_in = c_in
		self.c_out = c_out
		self.filter_dim = filter_dim
		self.step_size = step_size
		
		self.out_rows = (self.rows - self.filter_dim + step_size) // step_size
		self.out_cols = (self.cols - self.filter_dim + step_size) // step_size
		
		# pretend rows & cols; lets us act like step size == 1
		self.prows = self.out_rows + self.filter_dim - 1
		self.pcols = self.out_cols + self.filter_dim - 1
		
		self.flags = flags
		
		self.actual_cycles = -1
		self.actual_memory_accesses = -1
		self.flags = flags
	
	def equals(self, other):
		if not isinstance(other, ConvLayer):
			return False
		return self.rows == other.rows and self.cols == other.cols and self.c_in == other.c_in and self.c_out == other.c_out
	
	def get_flops(self):
		return (self.out_rows * self.out_cols * self.c_out) * ((self.filter_dim ** 2) * self.c_in)
	
	def get_flops_including_extras(self):
		return self.get_flops()
	
	def get_params(self):
		return self.filter_dim ** 2 * self.c_in * self.c_out
	
	def get_ideal_memory_accesses(self):
		return (self.rows * self.cols * self.c_in) * (self.c_in * self.c_out * self.filter_dim ** 2) * (self.out_rows * self.out_cols * self.c_out)
	
	def get_actual_cycles(self):
		return self.actual_cycles
	
	def get_actual_memory_accesses(self):
		return self.actual_memory_accesses
	
	def get_utilization(self, num_PEs):
		return self.get_flops_including_extras() / num_PEs / self.actual_cycles
	
	def get_mem_utilization(self, elems_per_cycle):
		return self.get_ideal_memory_accesses() / elems_per_cycle / self.actual_cycles
	
	def print(self):
		print(self.get_string(), end='')
	
	def get_string(self):
		return self.get_string_no_nl() + "\n"
	
	def get_string_no_nl(self):
		descrip = "Dims: " + str(self.rows) + ", " + str(self.cols) + ", " + str(self.c_in) + ", " + str(self.c_out) + ", " + str(self.filter_dim) + ", " + str(self.step_size) + " FLOPS: " + str(self.get_flops_including_extras()) + " Params: " + str(self.get_params()) + ", Flags:" + str(self.flags)
		return descrip

# generates a random parameter set given a specific hw and layer
def generate_random_param(hw, layer, count):
	# params: [c_in, x/y/ci/co, x/y/ci/co, x/y/ci/co, x/y/ci/co, t_x, t_y, t_ci, t_co, x/y/ci/kx/ky/co, x/y/ci/kx/ky/co, x/y/ci/kx/ky/co, x/y/ci/kx/ky/co, x/y/ci/kx/ky/co, x/y/ci/kx/ky/co, c_x, c_y, c_ci, c_kx, c_ky, c_co]
	
	# generate potential tiling params
	possibilities = []
	for t_x in range(1, layer.pcols):
		for t_y in range(1, layer.prows):
			for t_ci in range(1, layer.c_in):
				for t_co in range(1, layer.c_out):
					if t_x * t_y * t_ci + t_ci * t_co * layer.filter_dim ** 2 <= hw.total_sram_size and t_x * t_y * t_ci + t_ci * t_co * layer.filter_dim ** 2 >= 0.9 * hw.total_sram_size:
						possibilities.append([t_x, t_y, t_ci, t_co])
	
	if len(possibilities) == 0:
		possibilities.append([layer.pcols, layer.prows, layer.c_in, layer.c_out])
	
	while len(possibilities) < count:
		possibilities += possibilities
	
	selected = random.sample(possibilities, count)
	
	results = []
	
	for c in range(count):
		params = ["c_out"]
		params += random.sample(["x", "y", "ci", "co"], 4)
		params += selected[c]
		params += random.sample(["x", "y", "ci", "kx", "ky", "co"], 6)
		
		# c_w is always c_ci
		c_ci = min(random.randint(1, hw.size_RF), params[7])
		
		# c_a is from input
		c_a_choice = random.choice(["c_x", "c_y"])
		c_a_val = random.randint(1, hw.num_RFs_per_PE-1)
		if c_a_choice == "c_x":
			c_a_val = min(c_a_val, params[5])
			c_x = c_a_val
			c_y = 1
		else:
			c_a_val = min(c_a_val, params[6])
			c_x = 1
			c_y = c_a_val
		
		# c_b is from weight
		c_b_choice = random.choice(["c_kx", "c_ky", "c_co"])
		c_b_val = random.randint(1, hw.num_RFs_per_PE - c_a_val)
		if c_b_choice == "c_kx":
			c_b_val = min(c_b_val, layer.filter_dim)
			c_kx = c_b_val
			c_ky = 1
			c_co = 1
		elif c_b_choice == "c_ky":
			c_b_val = min(c_b_val, layer.filter_dim)
			c_ky = c_b_val
			c_kx = 1
			c_co = 1
		elif c_b_choice == "c_co":
			c_b_val = min(c_b_val, params[8])
			c_co = c_b_val
			c_kx = 1
			c_ky = 1
		
		params += [c_x, c_y, c_ci, c_kx, c_ky, c_co]
		
		results.append(params)
	
	return results