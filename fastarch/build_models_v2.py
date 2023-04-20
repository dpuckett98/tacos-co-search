import math
import matplotlib.pyplot as plt

class Model:

	def __init__(self, batch_size, name=""):
		self.layers = []
		self.batch_size = batch_size
		self.name = name

	# performs (batch_size * in_rows x in_elements) times (in_elements x out_elements)
	def add_linear(self, in_elements, out_elements, in_rows=1):
		self.layers.append({
			"type" : "linear",
			"batch_size" : self.batch_size,
			"in_elements" : in_elements,
			"out_elements" : out_elements,
			"in_rows" : in_rows,
			})
		return out_elements
	
	# filter size is filter_dim x filter_dim
	def add_convolution(self, in_width, in_height, in_channels, out_channels, filter_dim, step_size, padding = -1):
		# add padding
		if padding == -1:
			while (in_width - filter_dim + step_size) // step_size != (in_width - filter_dim + step_size) / step_size:
				in_width += 1
			while (in_height - filter_dim + step_size) // step_size != (in_height - filter_dim + step_size) / step_size:
				in_height += 1
		else:
			in_width += padding
			in_height += padding
		self.layers.append({
			"type" : "convolution",
			"batch_size" : self.batch_size,
			"in_width" : in_width,
			"in_height" : in_height,
			"in_channels" : in_channels,
			"out_channels" : out_channels,
			"filter_dim" : filter_dim,
			"step_size" : step_size,
			})
		out_width = (in_width - filter_dim + step_size) // step_size
		out_height = (in_height - filter_dim + step_size) // step_size
		return [out_width, out_height, out_channels]
	
	# comp_amt is the number of heads it is compressed to
	def add_attention(self, in_tokens, token_size, feature_dim, num_heads, comp_q_heads=-1, comp_k_heads=-1, sparsity=0.0, expand_ratio = 1):
		self.layers.append({
			"type" : "attention",
			"batch_size" : self.batch_size,
			"in_tokens" : in_tokens,
			"token_size" : token_size,
			"feature_dim" : feature_dim,
			"num_heads" : num_heads,
			"comp_q_heads" : comp_q_heads,
			"comp_k_heads" : comp_k_heads,
			"sparsity" : sparsity,
			"expand_ratio" : expand_ratio,
			})
		return [in_tokens, token_size]
	
	def add_attention_NASViT(self, in_tokens, token_size, feature_dim, num_heads, comp_q_heads=-1, comp_k_heads=-1, sparsity=0.0, expand_ratio = 1):
		self.layers.append({
			"type" : "attention_NASViT",
			"batch_size" : self.batch_size,
			"in_tokens" : in_tokens,
			"token_size" : token_size,
			"feature_dim" : feature_dim,
			"num_heads" : num_heads,
			"comp_q_heads" : comp_q_heads,
			"comp_k_heads" : comp_k_heads,
			"sparsity" : sparsity,
			"expand_ratio" : expand_ratio,
			})
		return [in_tokens, token_size]
	
	def print(self):
		for idx, layer in enumerate(self.layers):
			print(idx, ":", layer)
	
	def print_out_sizes(self):
		for idx, layer in enumerate(self.layers):
			if layer["type"] == "linear":
				print(idx, ":", layer["in_rows"], layer["out_elements"], "(linear) \t # Params:", layer["in_elements"] * layer["out_elements"])
			if layer["type"] == "convolution":
				out_width = (layer["in_width"] - layer["filter_dim"] + layer["step_size"]) // layer["step_size"]
				out_height = (layer["in_height"] - layer["filter_dim"] + layer["step_size"]) // layer["step_size"]
				print(idx, ":", layer["out_channels"], out_width, out_height, "(convolution)")
			if layer["type"] == "attention":
				print(idx, ":", layer["in_tokens"], layer["token_size"], "(attention)")

# encode/decode variables signal whether or not the layer should be encoded after computing or decoded before computing; encode_dim contains info about how the encoding/decoding dimensions
# sparse_map is None if the Layer is dense; otherwise, it is a 2D matrix of bools of size A_rows x B_cols, signaling whether or not the corresponding output element is sparse
# params = hardware parameters used to run this layer [optional]
class Layer:
	def __init__(self, A_rows, A_cols_B_rows, B_cols, sparsity=0.0, A_weights=False, B_weights=False, init_weights=0, flags = {}, A_transfer_scale=1, B_transfer_scale=1, O_transfer_scale=1, num_heads=1, load_immediate=False, store_immediate=False, params=None):
		self.A_rows = A_rows
		self.A_cols_B_rows = A_cols_B_rows
		self.B_cols = B_cols
		self.sparsity = sparsity
		self.init_weights = init_weights
		self.A_weights = A_weights
		self.B_weights = B_weights
		self.A_transfer_scale = A_transfer_scale
		self.B_transfer_scale = B_transfer_scale
		self.O_transfer_scale = O_transfer_scale
		self.num_heads = num_heads
		self.load_immediate = load_immediate
		self.store_immediate = store_immediate
		
		self.params = params
		
		self.actual_cycles = -1
		self.actual_memory_accesses = -1
		self.flags = flags
	
	def equals(self, other):
		#if "part" in self.flags.keys() and "part" in other.flags.keys():
		#	if self.flags["part"] != other.flags["part"]:
		#		return False
		return self.A_rows == other.A_rows and self.A_cols_B_rows == other.A_cols_B_rows and self.B_cols == other.B_cols and self.sparsity == other.sparsity #and self.flags["type"] == other.flags["type"]
	
	def get_flops(self):
		return self.A_rows * self.A_cols_B_rows * self.B_cols / self.num_heads
	
	def get_flops_including_extras(self):
		#flops = 0
		#if self.encode:
		#	flops += self.A_rows * self.B_cols * self.comp_head_dim
		#if self.decode:
		#	flops += (self.comp_head_dim * self.orig_head_dim) * (self.A_rows // self.orig_head_dim * self.A_cols_B_rows) # number of computations per tile * number of tiles
		#	flops += (self.comp_head_dim * self.orig_head_dim) * (self.A_cols_B_rows // self.orig_head_dim * self.B_cols) # number of computations per tile * number of tiles
		#flops += self.A_rows * self.A_cols_B_rows * self.B_cols * (1 - self.sparsity)
		return self.get_flops() * (1 - self.sparsity) / self.num_heads
	
	def get_params(self):
		total = self.init_weights
		if self.A_weights:
			total += self.A_rows * self.A_cols_B_rows
		if self.B_weights:
			total += self.A_cols_B_rows * self.B_cols
		return total
	
	def get_ideal_memory_accesses(self):
		#if encode or decode or sparse_map != None:
		#	raise Exception("Encode/decode and sparse attention not supported yet")
		# this is loading A, loading B, and saving result
		return self.A_rows * self.A_cols_B_rows + self.A_cols_B_rows * self.B_cols + self.A_rows * self.B_cols
	
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
		#print("Flags:" + str(self.flags) + ", Dims:", str(self.A_rows) + ",", str(self.B_cols) + ",", str(self.A_cols_B_rows), "FLOPS:", self.get_flops(), "Params:", self.get_params(), "Enc/Dec:", self.comp_head_dim / self.orig_head_dim, "Sparsity:", self.sparsity)
	
	def get_string(self):
		#descrip = "Dims: " + str(self.A_rows) + ", " + str(self.B_cols) + ", " + str(self.A_cols_B_rows) + " FLOPS: " + str(self.get_flops()) + " Params: " + str(self.get_params()) + " Sparsity: " + str(self.sparsity) + " Number of Heads: " + str(self.num_heads) + " A transfer scale: " + str(self.A_transfer_scale) + " B transfer scale: " + str(self.B_transfer_scale) + " O transfer scale: " + str(self.O_transfer_scale) + ", Flags:" + str(self.flags) + "\n"
		return self.get_string_no_nl() + "\n"
	
	def get_string_no_nl(self):
		descrip = "Dims: " + str(self.A_rows) + ", " + str(self.B_cols) + ", " + str(self.A_cols_B_rows) + " FLOPS: " + str(self.get_flops()) + " Params: " + str(self.get_params()) + " Sparsity: " + str(self.sparsity) + " Number of Heads: " + str(self.num_heads) + " Load Imm.: " + str(self.load_immediate) + " Store Imm.: " + str(self.store_immediate) + " A transfer scale: " + str(self.A_transfer_scale) + " B transfer scale: " + str(self.B_transfer_scale) + " O transfer scale: " + str(self.O_transfer_scale) + ", Flags:" + str(self.flags)
		return descrip

class LayerSet:
	
	def __init__(self, layers, model_name=""):
		self.layers = layers
		self.model_name = model_name
		
		# generate unique layers
		self.unique_layers = []
		for layer in self.layers:
			add = True
			for idx, unique in enumerate(self.unique_layers):
				if layer.equals(unique[0]):
					layer.flags["u_idx"] = idx
					add = False
					unique[1] += 1
					break
			if add:
				layer.flags["u_idx"] = len(self.unique_layers)
				self.unique_layers.append([layer, 1])

	def get_total_flops(self):
		flops = 0
		for layer in self.layers:
			flops += layer.get_flops()
		return flops
	
	def get_total_flops_including_extras(self):
		flops = 0
		for layer in self.layers:
			flops += layer.get_flops_including_extras()
		return flops
	
	def get_total_params(self):
		total = 0
		for layer in self.layers:
			total += layer.get_params()
		return total
	
	def get_total_ideal_memory_accesses(self):
		total = 0
		for layer in self.layers:
			total += layer.get_ideal_memory_accesses()
		return total
	
	def get_total_cycles(self):
		total = 0
		for layer in self.layers:
			total += layer.get_actual_cycles()
		return total
	
	def get_actual_memory_accesses(self):
		total = 0
		for layer in self.layers:
			total += layer.get_actual_memory_accesses()
		return total
	
	def get_utilization(self, num_PEs):
		ideal_cycles = self.get_total_flops_including_extras() / num_PEs
		return ideal_cycles / self.get_total_cycles() * 100
	
	def get_mem_utilization(self, elems_per_cycle):
		ideal_cycles = self.get_total_ideal_memory_accesses() / elems_per_cycle
		return ideal_cycles / self.get_total_cycles() * 100
	
	def update_layer_params(self, layer, params):
		for l in self.layers:
			if l.equals(layer):
				l.params = params
		for l, num in self.unique_layers:
			if l.equals(layer):
				l.params = params
	
	def update_layer_latency(self, layer, actual_cycles):
		for l in self.layers:
			if l.equals(layer):
				l.actual_cycles = actual_cycles
		for l, num in self.unique_layers:
			if l.equals(layer):
				l.actual_cycles = actual_cycles
	
	def update_layer_dram_accesses(self, layer, actual_accesses):
		for l in self.layers:
			if l.equals(layer):
				l.actual_memory_accesses = actual_accesses
		for l, num in self.unique_layers:
			if l.equals(layer):
				l.actual_memory_accesses = actual_accesses
	
	def print(self):
		print("Layer Set:")
		print("Model name:", self.model_name)
		print("Number of layers:", len(self.layers))
		print("Number of unique layers:", len(self.unique_layers))
		for layer in self.unique_layers:
			print(layer[1], end=": ")
			layer[0].print()
	
	def print_stats(self, num_PEs, elems_per_cycle, params):
		print(get_string_stats(num_PEs, elems_per_cycle, params))
		#print("Layer Set stats:")
		#total_cycles = self.get_total_cycles()
		#print("Total cycles:", total_cycles)
		#print("Number of layers:", len(self.layers))
		#print("Number of unique layers:", len(self.unique_layers))
		#for layer in self.unique_layers:
		#	print("--- Layer ({}x) ---".format(layer[1]))
		#	layer[0].print()
		#	print("Cycles:", layer[0].get_actual_cycles())
		#	print("Utilization: {:.2f}%".format(layer[0].get_utilization(num_PEs) * 100))
		#	print("Mem Utilization: {:.2f}%".format(layer[0].get_mem_utilization(elems_per_cycle) * 100))
		#	print("% of total cycles: {:.2f}%".format(layer[0].get_actual_cycles() * layer[1] / total_cycles * 100))
	
	def get_string_stats(self, num_PEs, elems_per_cycle, params):
		descrip = "Layer Set stats:\n"
		descrip += "Model name: " + self.model_name + "\n"
		descrip += "Total cycles: " + str(self.get_total_cycles()) + "\n"
		descrip += "Average utilization: {:.2f}%\n".format(self.get_utilization(num_PEs))
		descrip += "Average mem utilization: {:.2f}%\n".format(self.get_mem_utilization(elems_per_cycle))
		descrip += "Number of layers: " + str(len(self.layers)) + "\n"
		descrip += "Number of unique layers: " + str(len(self.unique_layers)) + "\n"
		for layer, p in zip(self.unique_layers, params):
			descrip += "--- Layer (" + str(layer[1]) + "x) ---\n"
			descrip += layer[0].get_string()
			descrip += "Cycles: " + str(layer[0].get_actual_cycles()) + "\n"
			descrip += "Utilization: {:.2f}%\n".format(layer[0].get_utilization(num_PEs) * 100)
			descrip += "Mem Utilization: {:.2f}%\n".format(layer[0].get_mem_utilization(elems_per_cycle) * 100)
			descrip += "% of total cycles: {:.2f}%\n".format(layer[0].get_actual_cycles() * layer[1] / self.get_total_cycles() * 100)
			descrip += "Tiling params: " + str(p) + "\n"
		return descrip

def model_to_layer_set(model):
	
	layers = []
	for layer_descrip in model.layers:
		if layer_descrip["type"] == "linear":
			layers.append(Layer(layer_descrip["batch_size"] * layer_descrip["in_rows"], layer_descrip["in_elements"], layer_descrip["out_elements"], B_weights=True, flags={"type":"linear"}))
		elif layer_descrip["type"] == "convolution":
			# convert convolution to matrix mult - more data accesses, but same FLOPS
			out_width = (layer_descrip["in_width"] - layer_descrip["filter_dim"] + layer_descrip["step_size"]) // layer_descrip["step_size"]
			out_height = (layer_descrip["in_height"] - layer_descrip["filter_dim"] + layer_descrip["step_size"]) // layer_descrip["step_size"]
			A_rows = out_width * out_height
			A_cols_B_rows = layer_descrip["filter_dim"] * layer_descrip["filter_dim"] * layer_descrip["in_channels"]
			B_cols = layer_descrip["out_channels"]
			for i in range(layer_descrip["batch_size"]):
				layers.append(Layer(A_rows, A_cols_B_rows, B_cols, init_weights = layer_descrip["out_channels"] * layer_descrip["filter_dim"]**2 * layer_descrip["in_channels"], flags={"type":"convolution"}))
		elif layer_descrip["type"] == "attention":
			for i in range(layer_descrip["batch_size"]):
				# computing Q
				if layer_descrip["comp_q_heads"] != -1 and layer_descrip["comp_q_heads"] != layer_descrip["num_heads"]:
					# with encoding
					# compute Q
					layers.append(Layer(layer_descrip["in_tokens"], layer_descrip["token_size"], layer_descrip["feature_dim"] * layer_descrip["num_heads"], B_weights=True, flags={"type":"attention", "part":"Q"}, store_immediate=True))
					# encode Q
					layers.append(Layer(layer_descrip["in_tokens"] * layer_descrip["feature_dim"], layer_descrip["num_heads"], layer_descrip["comp_q_heads"], B_weights=True, flags={"type":"attention", "part":"encode Q"}, load_immediate=True))
				else:
					# without encoding
					#print("without encoding", layer_descrip["comp_q_heads"], layer_descrip["num_heads"], layer_descrip["sparsity"])
					layers.append(Layer(layer_descrip["in_tokens"], layer_descrip["token_size"], layer_descrip["feature_dim"] * layer_descrip["num_heads"], B_weights=True, flags={"type":"attention", "part":"Q"}))
				
				# computing K
				if layer_descrip["comp_k_heads"] != -1 and layer_descrip["comp_k_heads"] != layer_descrip["num_heads"]:
					# with encoding
					# compute K
					layers.append(Layer(layer_descrip["in_tokens"], layer_descrip["token_size"], layer_descrip["feature_dim"] * layer_descrip["num_heads"], B_weights=True, flags={"type":"attention", "part":"K"}, store_immediate=True))
					# encode K
					layers.append(Layer(layer_descrip["in_tokens"] * layer_descrip["feature_dim"], layer_descrip["num_heads"], layer_descrip["comp_q_heads"], B_weights=True, flags={"type":"attention", "part":"encode K"}, load_immediate=True))
				else:
					# without encoding
					layers.append(Layer(layer_descrip["in_tokens"], layer_descrip["token_size"], layer_descrip["feature_dim"] * layer_descrip["num_heads"], B_weights=True, flags={"type":"attention", "part":"K"}))
				
				# decoding q if necessary
				q_transfer_scale = 1
				if layer_descrip["comp_q_heads"] != -1 and layer_descrip["comp_q_heads"] != layer_descrip["num_heads"]:
					q_transfer_scale = 0.0001
					layers.append(Layer(layer_descrip["in_tokens"] * layer_descrip["feature_dim"], layer_descrip["comp_q_heads"], layer_descrip["num_heads"], B_weights=True, flags={"type":"attention", "part":"decode Q"}, store_immediate=True))
				
				# decoding k if necessary
				k_transfer_scale = 1
				if layer_descrip["comp_k_heads"] != -1 and layer_descrip["comp_k_heads"] != layer_descrip["num_heads"]:
					k_transfer_scale = 0.0001
					layers.append(Layer(layer_descrip["in_tokens"] * layer_descrip["feature_dim"], layer_descrip["comp_k_heads"], layer_descrip["num_heads"], B_weights=True, flags={"type":"attention", "part":"decode K"}, store_immediate=True))
				
				# computing score
				if layer_descrip["comp_q_heads"] != -1 and layer_descrip["comp_q_heads"] != layer_descrip["num_heads"] or layer_descrip["comp_k_heads"] != -1 and layer_descrip["comp_k_heads"] != layer_descrip["num_heads"]:
					# if either Q or K is compressed, then overlap all the heads
					#print(layer_descrip["in_tokens"] * layer_descrip["num_heads"], layer_descrip["feature_dim"], layer_descrip["in_tokens"] * layer_descrip["num_heads"])
					layers.append(Layer(layer_descrip["in_tokens"] * layer_descrip["num_heads"], layer_descrip["feature_dim"], layer_descrip["in_tokens"] * layer_descrip["num_heads"], num_heads=layer_descrip["num_heads"], sparsity=layer_descrip["sparsity"], flags={"type":"attention", "part":"score"}, load_immediate=True))
				else:
					# if neither Q nor K is compressed, then compute each head separately (improves tiling over overlapped heads)
					for i in range(layer_descrip["num_heads"]):
						#print(layer_descrip["in_tokens"], layer_descrip["feature_dim"], layer_descrip["in_tokens"])
						layers.append(Layer(layer_descrip["in_tokens"], layer_descrip["feature_dim"], layer_descrip["in_tokens"], sparsity=layer_descrip["sparsity"], flags={"type":"attention", "part":"score"}))
			
			# V
			for i in range(layer_descrip["batch_size"]):
				layers.append(Layer(layer_descrip["in_tokens"], layer_descrip["token_size"], layer_descrip["feature_dim"] * layer_descrip["num_heads"] * layer_descrip["expand_ratio"], B_weights=True, flags={"type":"attention", "part":"V"}))
			
			# ignoring softmax
			
			# multiply by value vectors
			for i in range(layer_descrip["batch_size"] * layer_descrip["num_heads"]):
				layers.append(Layer(layer_descrip["in_tokens"], layer_descrip["in_tokens"], layer_descrip["feature_dim"] * layer_descrip["expand_ratio"], sparsity=layer_descrip["sparsity"], flags={"type":"attention", "part":"score * V"}))
			
			# out MLP (DeiT and Transformer include this in MHA)
			for i in range(layer_descrip["batch_size"]):
				layers.append(Layer(layer_descrip["in_tokens"], layer_descrip["feature_dim"] * layer_descrip["num_heads"] * layer_descrip["expand_ratio"], layer_descrip["token_size"], B_weights=True, flags={"type":"attention", "part":"MLP"}))
		elif layer_descrip["type"] == "attention_NASViT":
			for i in range(layer_descrip["batch_size"]):
				# computing Q
				if layer_descrip["comp_q_heads"] != -1 and layer_descrip["comp_q_heads"] != layer_descrip["num_heads"]:
					# with encoding
					# compute Q
					layers.append(Layer(layer_descrip["in_tokens"], layer_descrip["token_size"], layer_descrip["feature_dim"] * layer_descrip["num_heads"], B_weights=True, flags={"type":"attention", "part":"Q"}, store_immediate=True))
					# encode Q
					layers.append(Layer(layer_descrip["in_tokens"] * layer_descrip["feature_dim"], layer_descrip["num_heads"], layer_descrip["comp_q_heads"], B_weights=True, flags={"type":"attention", "part":"encode Q"}, load_immediate=True))
				else:
					# without encoding
					#print("without encoding", layer_descrip["comp_q_heads"], layer_descrip["num_heads"], layer_descrip["sparsity"])
					#print("Before Q:", layer_descrip["in_tokens"], layer_descrip["token_size"])
					#print("After Q:", layer_descrip["in_tokens"], layer_descrip["feature_dim"], "for each of", layer_descrip["num_heads"], "heads")
					layers.append(Layer(layer_descrip["in_tokens"], layer_descrip["token_size"], layer_descrip["feature_dim"] * layer_descrip["num_heads"], B_weights=True, flags={"type":"attention", "part":"Q"}))
					#layers.append(Layer(1, layer_descrip["token_size"], layer_descrip["feature_dim"] * layer_descrip["num_heads"], B_weights=True, flags={"type":"attention", "part":"Q"}))
				
				# computing K
				if layer_descrip["comp_k_heads"] != -1 and layer_descrip["comp_k_heads"] != layer_descrip["num_heads"]:
					# with encoding
					# compute K
					layers.append(Layer(layer_descrip["in_tokens"], layer_descrip["token_size"], layer_descrip["feature_dim"] * layer_descrip["num_heads"], B_weights=True, flags={"type":"attention", "part":"K"}, store_immediate=True))
					# encode K
					layers.append(Layer(layer_descrip["in_tokens"] * layer_descrip["feature_dim"], layer_descrip["num_heads"], layer_descrip["comp_q_heads"], B_weights=True, flags={"type":"attention", "part":"encode K"}, load_immediate=True))
				else:
					# without encoding
					#print(layer_descrip["in_tokens"], layer_descrip["token_size"], layer_descrip["feature_dim"], layer_descrip["num_heads"])
					layers.append(Layer(layer_descrip["in_tokens"], layer_descrip["token_size"], layer_descrip["feature_dim"] * layer_descrip["num_heads"], B_weights=True, flags={"type":"attention", "part":"K"}))
					#layers.append(Layer(1, layer_descrip["token_size"], layer_descrip["feature_dim"] * layer_descrip["num_heads"], B_weights=True, flags={"type":"attention", "part":"K"}))
				
				# decoding q if necessary
				q_transfer_scale = 1
				if layer_descrip["comp_q_heads"] != -1 and layer_descrip["comp_q_heads"] != layer_descrip["num_heads"]:
					q_transfer_scale = 0.0001
					layers.append(Layer(layer_descrip["in_tokens"] * layer_descrip["feature_dim"], layer_descrip["comp_q_heads"], layer_descrip["num_heads"], B_weights=True, flags={"type":"attention", "part":"decode Q"}, store_immediate=True))
				
				# decoding k if necessary
				k_transfer_scale = 1
				if layer_descrip["comp_k_heads"] != -1 and layer_descrip["comp_k_heads"] != layer_descrip["num_heads"]:
					k_transfer_scale = 0.0001
					layers.append(Layer(layer_descrip["in_tokens"] * layer_descrip["feature_dim"], layer_descrip["comp_k_heads"], layer_descrip["num_heads"], B_weights=True, flags={"type":"attention", "part":"decode K"}, store_immediate=True))
				
				# computing score
				if layer_descrip["comp_q_heads"] != -1 and layer_descrip["comp_q_heads"] != layer_descrip["num_heads"] or layer_descrip["comp_k_heads"] != -1 and layer_descrip["comp_k_heads"] != layer_descrip["num_heads"]:
					# if either Q or K is compressed, then overlap all the heads
					#print(layer_descrip["in_tokens"] * layer_descrip["num_heads"], layer_descrip["feature_dim"], layer_descrip["in_tokens"] * layer_descrip["num_heads"])
					layers.append(Layer(layer_descrip["in_tokens"] * layer_descrip["num_heads"], layer_descrip["feature_dim"], layer_descrip["in_tokens"] * layer_descrip["num_heads"], num_heads=layer_descrip["num_heads"], sparsity=layer_descrip["sparsity"], flags={"type":"attention", "part":"score"}, load_immediate=True))
				else:
					# if neither Q nor K is compressed, then compute each head separately (improves tiling over overlapped heads)
					for i in range(layer_descrip["num_heads"]):
						#print(layer_descrip["in_tokens"], layer_descrip["feature_dim"], layer_descrip["in_tokens"])
						layers.append(Layer(layer_descrip["in_tokens"], layer_descrip["feature_dim"], layer_descrip["in_tokens"], sparsity=layer_descrip["sparsity"], flags={"type":"attention", "part":"score"}))
			
			# V
			for i in range(layer_descrip["batch_size"]):
				# compute V
				layers.append(Layer(layer_descrip["in_tokens"], layer_descrip["token_size"], layer_descrip["feature_dim"] * layer_descrip["num_heads"] * layer_descrip["expand_ratio"], B_weights=True, flags={"type":"attention", "part":"V"}))
				# do V dw conv (convert convolution to matrix mult - more data accesses, but same FLOPS)
				#h = layer_descrip["in_tokens"] ** 0.5
				#w = h
				k = 3
				s = 1
				c_in = 1
				c_out = 1
				#out_width = (w - k + s) // s
				#out_height = (h - k + s) // s
				A_rows = layer_descrip["in_tokens"] #out_width * out_height
				A_cols_B_rows = k * k * c_in
				B_cols = c_out
				for j in range(layer_descrip["feature_dim"] * layer_descrip["num_heads"] * layer_descrip["expand_ratio"]):
					layers.append(Layer(A_rows, A_cols_B_rows, B_cols, init_weights = c_out * k**2 * c_in, flags={"type":"attention", "part":"V_dw_conv"}))
			
			# proj l & w
			for i in range(layer_descrip["batch_size"]):
				layers.append(Layer(layer_descrip["in_tokens"] ** 2, layer_descrip["num_heads"], layer_descrip["num_heads"], B_weights=True, flags={"type":"attention", "part":"proj_l"}))
				#layers.append(Layer(1, layer_descrip["num_heads"], layer_descrip["num_heads"], B_weights=True, flags={"type":"attention", "part":"proj_l"}))
				layers.append(Layer(layer_descrip["in_tokens"] ** 2, layer_descrip["num_heads"], layer_descrip["num_heads"], B_weights=True, flags={"type":"attention", "part":"proj_w"}))
				#layers.append(Layer(1, layer_descrip["num_heads"], layer_descrip["num_heads"], B_weights=True, flags={"type":"attention", "part":"proj_w"}))
			
			# ignoring softmax
			
			# multiply by value vectors
			for i in range(layer_descrip["batch_size"] * layer_descrip["num_heads"]):
				layers.append(Layer(layer_descrip["in_tokens"], layer_descrip["in_tokens"], layer_descrip["feature_dim"] * layer_descrip["expand_ratio"], sparsity=layer_descrip["sparsity"], flags={"type":"attention", "part":"score * V"}))
			
			# out MLP (DeiT and Transformer include this in MHA)
			for i in range(layer_descrip["batch_size"]):
				layers.append(Layer(layer_descrip["in_tokens"], layer_descrip["feature_dim"] * layer_descrip["num_heads"] * layer_descrip["expand_ratio"], layer_descrip["token_size"], B_weights=True, flags={"type":"attention", "part":"MLP"}))
				#layers.append(Layer(1, layer_descrip["feature_dim"] * layer_descrip["num_heads"] * layer_descrip["expand_ratio"], layer_descrip["token_size"], B_weights=True, flags={"type":"attention", "part":"MLP"}))
	
	ls = LayerSet(layers, model.name)
	return ls

def get_AlexNet(batch_size):
	alexNet = Model(batch_size)
	alexNet.add_convolution(227, 227, 3, 96, 11, 4)
	alexNet.add_convolution(31, 31, 48, 256, 5, 1)
	alexNet.add_convolution(15, 15, 256, 384, 3, 1)
	alexNet.add_convolution(15, 15, 192, 384, 3, 1)
	alexNet.add_convolution(15, 15, 192, 256, 3, 1)
	alexNet.add_linear(6*256, 4096)
	alexNet.add_linear(4096, 4096)
	alexNet.add_linear(4096, 1000)
	return alexNet

def get_ViT(batch_size, in_dim, in_channels, patch_size, num_layers, feature_dim, MLP_size, num_heads):
	vit = Model(batch_size)
	
	# compute patch embeddings
	num_patches = math.ceil(in_dim * in_dim / patch_size / patch_size)
	for i in range(num_patches):
		vit.add_linear(patch_size*patch_size*in_channels, feature_dim)
	
	# layers
	for i in range(num_layers):
		vit.add_attention(num_patches, feature_dim, feature_dim, num_heads)
		vit.add_linear(num_patches * feature_dim, MLP_size)
		vit.add_linear(MLP_size, num_patches * feature_dim)
	
	# output
	vit.add_linear(feature_dim, 1000) # assumes ImageNet w/ 1000 possible classification values
	
	return vit

def get_ViT_Large_16(batch_size):
	return get_ViT(batch_size, 224, 3, 16, 12, 768, 3072, 12)

# assumes in_channels = 3
def get_LeViT(batch_size, in_dim, conv_dim, QK_dim, c_1, n_1, count_1, c_2, n_2, count_2, c_3, n_3, count_3, q_1=-1, k_1=-1, q_2=-1, k_2=-1, q_3=-1, k_3=-1, sparsity=0.0):
	levit = Model(batch_size)
	
	# convolution blocks
	w, h, c = levit.add_convolution(in_dim, in_dim, 3, conv_dim, 3, 2)
	w, h, c = levit.add_convolution(w, h, c, c*2, 3, 2)
	w, h, c = levit.add_convolution(w, h, c, c*2, 3, 2)
	w, h, c = levit.add_convolution(w, h, c, c*2, 3, 2)
	#print(w, h, c)
	#w = 14
	#h = 14
	#c = 256
	# stage 1
	#print(c_1, c)
	#assert c_1 == c
	for i in range(count_1):
		levit.add_attention(w * h, c_1, QK_dim, n_1, sparsity=sparsity, expand_ratio=2, comp_q_heads=q_1, comp_k_heads=k_1)
		levit.add_linear(c_1, c_1 * 2, in_rows = w*h)
		levit.add_linear(c_1 * 2, c_1, in_rows = w*h)
	levit.add_attention(w * h, c_1, QK_dim, n_1 * 2) # this is actually supposed to be a "Shrinking Attention Block" but it's less computation heavy than regular attention
	
	w = math.ceil(w / 2)
	h = math.ceil(h / 2)
	
	# stage 2
	levit.add_linear(c_2, c_2 * 2, in_rows = w*h)
	levit.add_linear(c_2 * 2, c_2, in_rows = w*h)
	for i in range(count_2):
		levit.add_attention(w * h, c_2, QK_dim, n_2, sparsity=sparsity, expand_ratio=2, comp_q_heads=q_2, comp_k_heads=k_2)
		levit.add_linear(c_2, c_2 * 2, in_rows = w*h)
		levit.add_linear(c_2 * 2, c_2, in_rows = w*h)
	levit.add_attention(w * h, c_2, QK_dim, n_2 * 2) # this is actually supposed to be a "Shrinking Attention Block" but it's less computation heavy than regular attention
	
	w = math.ceil(w / 2)
	h = math.ceil(h / 2)
	
	# stage 3
	levit.add_linear(c_3, c_3 * 2, in_rows = w*h)
	levit.add_linear(c_3 * 2, c_3, in_rows = w*h)
	for i in range(count_3):
		levit.add_attention(w * h, c_3, QK_dim, n_3, sparsity=sparsity, expand_ratio=2, comp_q_heads=q_3, comp_k_heads=k_3)
		levit.add_linear(c_3, c_3 * 2, in_rows = w*h)
		levit.add_linear(c_3 * 2, c_3, in_rows = w*h)
	#levit.add_attention(w * h, c_3, QK_dim, n_3 * 2) # this is actually supposed to be a "Shrinking Attention Block" but it's less computation heavy than regular attention
	
	# head
	levit.add_linear(c_3, 1000, in_rows = w*h)
	# distillation head
	levit.add_linear(c_3, 1000, in_rows = w*h)
	
	return levit

def get_LeViT_ViTCoD(batch_size, in_dim, conv_dim, QK_dim, c_1, n_1, count_1, c_2, n_2, count_2, c_3, n_3, count_3, q_1=-1, k_1=-1, q_2=-1, k_2=-1, q_3=-1, k_3=-1, sparsity=0.0):
	levit = Model(batch_size)
	
	# convolution blocks
	w, h, c = levit.add_convolution(in_dim, in_dim, 3, conv_dim, 3, 2)
	w, h, c = levit.add_convolution(w, h, c, c*2, 3, 2)
	w, h, c = levit.add_convolution(w, h, c, c*2, 3, 2)
	w, h, c = levit.add_convolution(w, h, c, c*2, 3, 2)
	#print(w, h, c)
	#w = 14
	#h = 14
	#c = 256
	# stage 1
	#print(c_1, c)
	#assert c_1 == c
	for i in range(count_1):
		levit.add_attention(w * h, c_1, QK_dim, n_1, sparsity=sparsity, expand_ratio=2, comp_q_heads=q_1, comp_k_heads=k_1)
		levit.add_linear(c_1, c_1 * 2, in_rows = w*h)
		levit.add_linear(c_1 * 2, c_1, in_rows = w*h)
	#levit.add_attention(w * h, c_1, QK_dim, n_1 * 2) # this is actually supposed to be a "Shrinking Attention Block" but it's less computation heavy than regular attention
	
	w = math.ceil(w / 2)
	h = math.ceil(h / 2)
	
	# stage 2
	levit.add_linear(c_2, c_2 * 2, in_rows = w*h)
	levit.add_linear(c_2 * 2, c_2, in_rows = w*h)
	for i in range(count_2):
		levit.add_attention(w * h, c_2, QK_dim, n_2, sparsity=sparsity, expand_ratio=2, comp_q_heads=q_2, comp_k_heads=k_2)
		levit.add_linear(c_2, c_2 * 2, in_rows = w*h)
		levit.add_linear(c_2 * 2, c_2, in_rows = w*h)
	#levit.add_attention(w * h, c_2, QK_dim, n_2 * 2) # this is actually supposed to be a "Shrinking Attention Block" but it's less computation heavy than regular attention
	
	w = math.ceil(w / 2)
	h = math.ceil(h / 2)
	
	# stage 3
	levit.add_linear(c_3, c_3 * 2, in_rows = w*h)
	levit.add_linear(c_3 * 2, c_3, in_rows = w*h)
	for i in range(count_3):
		levit.add_attention(w * h, c_3, QK_dim, n_3, sparsity=sparsity, expand_ratio=2, comp_q_heads=q_3, comp_k_heads=k_3)
		levit.add_linear(c_3, c_3 * 2, in_rows = w*h)
		levit.add_linear(c_3 * 2, c_3, in_rows = w*h)
	#levit.add_attention(w * h, c_3, QK_dim, n_3 * 2) # this is actually supposed to be a "Shrinking Attention Block" but it's less computation heavy than regular attention
	
	# head
	#levit.add_linear(c_3, 1000, in_rows = w*h)
	# distillation head
	#levit.add_linear(c_3, 1000, in_rows = w*h)
	
	return levit

def get_LeViT_128(batch_size, comp_q_ratio, comp_k_ratio, sparsity, ViTCoD=False):
	if ViTCoD:
		return get_LeViT_ViTCoD(batch_size, 224, 16, 16, 128, 4, 4, 256, 8, 4, 384, 12, 4, int(4 * comp_q_ratio), int(4 * comp_k_ratio), int(8 * comp_q_ratio), int(8 * comp_k_ratio), int(12 * comp_q_ratio), int(12 * comp_k_ratio), sparsity)
	return get_LeViT(batch_size, 224, 16, 16, 128, 4, 4, 256, 8, 4, 384, 12, 4, int(4 * comp_q_ratio), int(4 * comp_k_ratio), int(8 * comp_q_ratio), int(8 * comp_k_ratio), int(12 * comp_q_ratio), int(12 * comp_k_ratio), sparsity)

def get_LeViT_192(batch_size, comp_q_ratio, comp_k_ratio, sparsity, ViTCoD=False):
	if ViTCoD:
		return get_LeViT_ViTCoD(batch_size, 224, 32, 32, 192, 3, 4, 288, 5, 4, 384, 6, 4, int(3 * comp_q_ratio), int(3 * comp_k_ratio), int(5 * comp_q_ratio), int(5 * comp_k_ratio), int(6 * comp_q_ratio), int(6 * comp_k_ratio), sparsity)
	return get_LeViT(batch_size, 224, 32, 32, 192, 3, 4, 288, 5, 4, 384, 6, 4, int(3 * comp_q_ratio), int(3 * comp_k_ratio), int(5 * comp_q_ratio), int(5 * comp_k_ratio), int(6 * comp_q_ratio), int(6 * comp_k_ratio), sparsity)

def get_LeViT_256(batch_size, comp_q_ratio, comp_k_ratio, sparsity, ViTCoD=False):
	if ViTCoD:
		return get_LeViT_ViTCoD(batch_size, 224, 32, 32, 256, 4, 4, 384, 6, 4, 512, 8, 4, int(4 * comp_q_ratio), int(4 * comp_k_ratio), int(6 * comp_q_ratio), int(6 * comp_k_ratio), int(8 * comp_q_ratio), int(8 * comp_k_ratio), sparsity)
	return get_LeViT(batch_size, 224, 32, 32, 256, 4, 4, 384, 6, 4, 512, 8, 4, int(4 * comp_q_ratio), int(4 * comp_k_ratio), int(6 * comp_q_ratio), int(6 * comp_k_ratio), int(8 * comp_q_ratio), int(8 * comp_k_ratio), sparsity)

def get_LeViT_384(batch_size, comp_q_ratio, comp_k_ratio, sparsity, ViTCoD=False):
	if ViTCoD:
		return get_LeViT_ViTCoD(batch_size, 224, 48, 32, 384, 6, 4, 512, 9, 4, 768, 12, 4, int(6 * comp_q_ratio), int(6 * comp_k_ratio), int(9 * comp_q_ratio), int(9 * comp_k_ratio), int(12 * comp_q_ratio), int(12 * comp_k_ratio), sparsity)
	return get_LeViT(batch_size, 224, 48, 32, 384, 6, 4, 512, 9, 4, 768, 12, 4, int(6 * comp_q_ratio), int(6 * comp_k_ratio), int(9 * comp_q_ratio), int(9 * comp_k_ratio), int(12 * comp_q_ratio), int(12 * comp_k_ratio), sparsity)

def get_DeiT_attention(batch_size, in_dim, emb_dim, num_heads, num_layers, comp_q_heads=-1, comp_k_heads=-1, sparsity=0.0):
	deit = Model(batch_size)

	dim_per_head = 64
	patch_size = 16
	
	# patch embedding
	h, w, c = deit.add_convolution(224, 224, 3, emb_dim, 16, 16)
	
	in_tokens = (in_dim // patch_size) ** 2
	in_tokens += 2 # add cls and dist tokens
	for i in range(num_layers):
		deit.add_attention(in_tokens, emb_dim, dim_per_head, num_heads, sparsity=sparsity, comp_q_heads=comp_q_heads, comp_k_heads=comp_k_heads)
		deit.add_linear(emb_dim, emb_dim * 4, in_rows = in_tokens)
		deit.add_linear(emb_dim * 4, emb_dim, in_rows = in_tokens)
	
	# distillation heads
	deit.add_linear(emb_dim, 1000) # reg output
	deit.add_linear(emb_dim, 1000) # dist output
	
	return deit

def get_DeiT_attention_matching_ViTCoD(batch_size, in_dim, emb_dim, num_heads, num_layers, comp_q_heads=-1, comp_k_heads=-1, sparsity=0.0):
	deit = Model(batch_size)

	dim_per_head = 64
	patch_size = 16
	
	# patch embedding
	#h, w, c = deit.add_convolution(224, 224, 3, emb_dim, 16, 16)
	
	in_tokens = (in_dim // patch_size) ** 2
	in_tokens += 1 # add cls token
	for i in range(num_layers):
		deit.add_attention(in_tokens, emb_dim, dim_per_head, num_heads, sparsity=sparsity, comp_q_heads=comp_q_heads, comp_k_heads=comp_k_heads)
		deit.add_linear(emb_dim, emb_dim * 4, in_rows = in_tokens)
		deit.add_linear(emb_dim * 4, emb_dim, in_rows = in_tokens)
	
	# distillation heads
	#deit.add_linear(emb_dim, 1000) # reg output
	#deit.add_linear(emb_dim, 1000) # dist output
	
	return deit

def get_DeiT_Tiny_ViTCoD(batch_size, comp_q_ratio, comp_k_ratio, sparsity):
	return get_DeiT_attention_matching_ViTCoD(batch_size, 224, 192, 3, 12, int(3 * comp_q_ratio), int(3 * comp_k_ratio), sparsity)

def get_DeiT_Tiny(batch_size, comp_q_ratio, comp_k_ratio, sparsity):
	return get_DeiT_attention(batch_size, 224, 192, 3, 12, int(3 * comp_q_ratio), int(3 * comp_k_ratio), sparsity)

def get_DeiT_Small(batch_size, comp_q_ratio, comp_k_ratio, sparsity):
	return get_DeiT_attention(batch_size, 224, 384, 6, 12, int(6 * comp_q_ratio), int(6 * comp_k_ratio), sparsity)

def get_DeiT_Base(batch_size, comp_q_ratio, comp_k_ratio, sparsity):
	return get_DeiT_attention(batch_size, 224, 768, 12, 12, int(12 * comp_q_ratio), int(12 * comp_k_ratio), sparsity)

def get_DeiT_Tiny_Attention(batch_size, comp_ratio, sparsity):
	deit = Model(batch_size)
	
	in_dim = 224
	emb_dim = 192
	num_heads = 3
	
	dim_per_head = 64
	patch_size = 16
	
	in_tokens = (in_dim // patch_size) ** 2
	
	if comp_ratio == 1.0:
		deit.add_attention(in_tokens, emb_dim, dim_per_head, num_heads, sparsity=sparsity)
	else:
		comp_amt = math.ceil(num_heads * comp_ratio)
		deit.add_attention(in_tokens, emb_dim, dim_per_head, num_heads, sparsity=sparsity, comp_amt=comp_amt)
	return deit

def get_test(batch_size=1):
	test = Model(batch_size)
	#test.add_attention(100, 100, 100, 10, sparsity=0.5, comp_amt=5)
	test.add_linear(1000, 1000, 100)
	test.add_linear(1050, 1000, 100)
	return test

#model = get_LeViT(1, 244, 128, 4, 1, 256, 8, 0, 384, 12, 0)

#model = get_DeiT_Tiny(1)
#layer_set = model_to_layer_set(model)
#layer_set.print()
#print(layer_set.get_total_flops(), layer_set.get_total_params())

#model = get_ViT_Large_16(1)
#layer_set = model_to_layer_set(model)

#print("Number of layers:", len(layer_set.layers))
#print("Number of unique layers:", len(layer_set.unique_layers))
#for layer in layer_set.unique_layers:
#	print(layer[1], end=": ")
#	layer[0].print()

# Copied from NASViT
def make_divisible(v, divisor=8, min_value=1):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# Ignores computing shortcut
def get_MBConvLayer(model, h, w, c, out_channel, kernel_size, expand_ratio, stride=1, channels_per_group=1, use_se=False):
	# inverted bottleneck
	if expand_ratio != 1:
		h, w, c = model.add_convolution(h, w, c, make_divisible(c * expand_ratio, 8), 1, 1)
	#print(h, w, c)
	
	# depthwise convolution (h, w, c passes unchnaged to the other side)
	# in channels is c; kernel size is kernel_size; stride is stride, channels_per_group is channels_per_group (aka groups = c // channels_per_group)
	for i in range(c // channels_per_group):
		# quick bug fix 'cause I'm lazy
		if h == 9 and stride == 2:
			h2, w2, c2 = model.add_convolution(h, w, channels_per_group, channels_per_group, kernel_size, stride, padding=kernel_size-stride+1)
		else:
			h2, w2, c2 = model.add_convolution(h, w, channels_per_group, channels_per_group, kernel_size, stride, padding=kernel_size-stride)
	h = h2
	w = w2
	
	# squeeze and excite layer (REDUCTION=4)
	if use_se:
		# reduce
		_, _, c2 = model.add_convolution(1, 1, c, make_divisible(c // 4, 8), 1, 1)
		#print(h, w, c, make_divisible(c // 4, 8))
		# expand
		_, _, c = model.add_convolution(1, 1, c2, c, 1, 1)
	
	# point linear
	h, w, c = model.add_convolution(h, w, c, out_channel, 1, 1)
	#print(h, w, c)
	
	return h, w, c

# TODO: follow nasvit code
# Ignores computing shortcut
def get_SwinTransformerLayer(model, h, w, c, dimension, num_heads, mlp_ratio=1, rescale=1, comp_size_q=1, comp_size_k=1, sparsity=0.0):
	# reshape into tokens
	num_tokens = w * h
	token_dim = c
	
	#print(h, w, c)
	
	# attention
	#token_size = token_dim // num_heads
	
	head_dimension = 8
	num_heads = dimension // head_dimension
	
	#print(num_heads)
	
	# in_tokens, token_size, feature_dim, num_heads, comp_q_heads=-1, comp_k_heads=-1, sparsity=0.0, expand_ratio = 1
	model.add_attention_NASViT(num_tokens, token_dim, head_dimension, num_heads, expand_ratio=4, comp_q_heads = int(num_heads * comp_size_q), comp_k_heads = int(num_heads * comp_size_k), sparsity=sparsity)
	
	# Two MLPs in a row, where the # of hidden neurons is the same as the number of input/output neurons
	model.add_linear(token_dim, token_dim, num_tokens)
	model.add_linear(token_dim, token_dim, num_tokens)
	model.add_linear(token_dim, token_dim, num_tokens)
	model.add_linear(token_dim, token_dim, num_tokens)
	
	return h, w, c

def NASViT_subnet_to_model(subnet_config, batch_size=1, comp_size_q=1, comp_size_k=1):
	model = Model(batch_size)
	
	# constants
	strides = [1, 2, 2, 2, 2, 1, 2]
	num_heads = [8, 8, 16, 16]
	
	# input dimensions
	c = 3
	h = subnet_config["resolution"]
	w = subnet_config["resolution"]
	
	# first do "first conv"
	h, w, c = model.add_convolution(h, w, c, subnet_config["width"][0], 3, 2)
	print(subnet_config["inv_sparsity"])
	# then, for each stage...
	for stage_id in range(7):
		#print(h, w, c)
		#print(subnet_config["depth"][stage_id])
		# for each block in stage...
		for i in range(subnet_config["depth"][stage_id]):
			# generate an MBConv if it's in one of the first three stages, or if it's the first block in a later stage
			if stage_id < 3 or i == 0:
				if i == 0:
					stride = strides[stage_id]
				else:
					stride = 1
				#print("Before MBConv", h, w, c)
				h, w, c = get_MBConvLayer(model, h, w, c, subnet_config["width"][stage_id+1], subnet_config["kernel_size"][stage_id], subnet_config["expansion_ratio"][stage_id], stride=stride, use_se=subnet_config["use_se"][stage_id])
				#print("After MBConv", h, w, c)
				
			# generate SwinTransformers
			else:
				#print("Before swin", h, w, c)
				get_SwinTransformerLayer(model, h, w, c, subnet_config["width"][stage_id+1], num_heads[stage_id-3], mlp_ratio=subnet_config["expansion_ratio"][stage_id], comp_size_q=comp_size_q, comp_size_k=comp_size_k, sparsity=1.0 - subnet_config["inv_sparsity"][stage_id])
				#print("After swin", h, w, c)
		
		#if stage_id == 6: # and i == 0:
		#	return model
	
	# finally, do "last conv"
	#print(h, w, c)
	h, w, c = model.add_convolution(h, w, c, c * 6, 1, 1)
	# adaptive average pooling inbetween here --> it converts hxw to 1x1
	h, w, c = model.add_convolution(1, 1, c * 6, c, 1, 1)
	
	return model

# TODO: replace blocks below with the MBConvLayer and SwinTransformerLayer blocks above
def subnet_to_model(subnet_config, batch_size=1):
	model = Model(batch_size)

	# Working through https://towardsdatascience.com/residual-bottleneck-inverted-residual-linear-bottleneck-mbconv-explained-89d7b7e7c6bc to help debug

	#print("Initial:", 3, subnet_config["resolution"], subnet_config["resolution"])

	# first conv
	h, w, c = model.add_convolution(subnet_config["resolution"], subnet_config["resolution"], 3, subnet_config["width"][0], 3, 2)
	#print("After first conv:", c, w, h)
	block = 0
	# mb conv blocks
	for i in range(subnet_config["depth"][1]):
		#print(h, w, c)
		nh, nw, nc = model.add_convolution(h+1, w+1, c, c * subnet_config["expansion_ratio"][1], 1, 1)
		#print(nh, nw, nc)
		for j in range(nc):
			nh, nw, c2 = model.add_convolution(h, w, 1, 1, subnet_config["kernel_size"][1], 1)
		#print(h, w, c2)
		h, w, c = model.add_convolution(h, w, nc, c, 1, 1)
		#print("After block", block, ":", c, w, h)
		block += 1
		#print(h, w, c)
		#raise Exception()
		#model.add_convolution(h, w, c, subnet_config["width"][1], subnet_config["kernel_size"][1], 1)
	#print("next")
	for i in range(subnet_config["depth"][2]):
		stride = 2 if i == 0 else 1
		nh, nw, nc = model.add_convolution(h, w, c, c * subnet_config["expansion_ratio"][2], 1, 1)
		for j in range(nc):
			nh, nw, c2 = model.add_convolution(h, w, 1, 1, subnet_config["kernel_size"][2], stride)
		if i == 0:
			h, w, c = model.add_convolution(nh+1, nw+1, nc, subnet_config["width"][2], 1, 1)
		else:
			h, w, c = model.add_convolution(h, w, nc, subnet_config["width"][2], 1, 1)
		#print("After block", block, ":", c, w, h)
		block += 1
		#model.add_convolution(h, w, c, subnet_config["width"][2], subnet_config["kernel_size"][2], 2)
	#print("next")
	for i in range(subnet_config["depth"][3]):
		stride = 2 if i == 0 else 1
		nh, nw, nc = model.add_convolution(h, w, c, c * subnet_config["expansion_ratio"][3], 1, 1)
		for j in range(nc):
			nh, nw, c2 = model.add_convolution(h, w, 1, 1, subnet_config["kernel_size"][3], stride)
		if i == 0:
			h, w, c = model.add_convolution(nh, nw, nc, subnet_config["width"][3], 1, 1)
		else:
			h, w, c = model.add_convolution(h, w, nc, subnet_config["width"][3], 1, 1)
		#print("After block", block, ":", c, w, h)
		block += 1
		#model.add_convolution(h, w, c, subnet_config["width"][3], subnet_config["kernel_size"][3], 3)
	
	# transformer blocks
	#print("next")
	nh, nw, nc = model.add_convolution(h, w, c, c * subnet_config["expansion_ratio"][4], 1, 1)
	for j in range(nc):
		nh, nw, c2 = model.add_convolution(h, w, 1, 1, subnet_config["kernel_size"][4], 2)
	h, w, c = model.add_convolution(nh, nw, nc, subnet_config["width"][4], 1, 1)
	#print("After block", block, ":", c, w, h)
	block += 1
	for i in range(subnet_config["depth"][4]-1):
		model.add_attention(h * w, c // 8, subnet_config["width"][4] // 8, 8, sparsity=0, comp_amt=subnet_config["en_de_q"][4], expand_ratio=subnet_config["expansion_ratio"][4]) #, comp_amt_k=subnet_config["en_de_k"][4])
		model.add_linear(subnet_config["width"][4] // 8, subnet_config["width"][4] // 8 * 1, in_rows = h*w)
		model.add_linear(subnet_config["width"][4] // 8 * 1, subnet_config["width"][4] // 8, in_rows = h*w)
		#print("After block", block, ":", c, w, h)
		block += 1
	
	#print("next")
	#h, w, c = model.add_convolution(h, w, c, subnet_config["width"][5], 3, 2)
	nh, nw, nc = model.add_convolution(h, w, c, c * subnet_config["expansion_ratio"][5], 1, 1)
	for j in range(nc):
		nh, nw, c2 = model.add_convolution(h, w, 1, 1, subnet_config["kernel_size"][5], 2)
	h, w, c = model.add_convolution(nh, nw, nc, subnet_config["width"][5], 1, 1)
	#print("After block", block, ":", c, w, h)
	block += 1
	for i in range(subnet_config["depth"][5]-1):
		model.add_attention(h * w, c // 8, subnet_config["width"][5] // 8, 8, sparsity=0, comp_amt=subnet_config["en_de_q"][5], expand_ratio=subnet_config["expansion_ratio"][5]) #, comp_amt_k=subnet_config["en_de_k"][5])
		model.add_linear(subnet_config["width"][5] // 8, subnet_config["width"][5] // 8 * 1, in_rows = h*w)
		model.add_linear(subnet_config["width"][5] // 8 * subnet_config["expansion_ratio"][5], subnet_config["width"][5] // 8, in_rows = h*w)
		#print("After block", block, ":", c, w, h)
		block += 1
	
	#print("next")
	#h, w, c = model.add_convolution(h+2, w+2, c, subnet_config["width"][6], 3, 1)
	nh, nw, nc = model.add_convolution(h, w, c, c * subnet_config["expansion_ratio"][6], 1, 1)
	for j in range(nc):
		nh, nw, c2 = model.add_convolution(h, w, 1, 1, subnet_config["kernel_size"][6], 1)
	h, w, c = model.add_convolution(h, w, nc, subnet_config["width"][6], 1, 1)
	#print("After block", block, ":", c, w, h)
	block += 1
	for i in range(subnet_config["depth"][6]-1):
		model.add_attention(h * w, c // 16, subnet_config["width"][6] // 8, 16, sparsity=0, comp_amt=subnet_config["en_de_q"][6], expand_ratio=subnet_config["expansion_ratio"][6]) #, comp_amt_k=subnet_config["en_de_k"][6])
		model.add_linear(subnet_config["width"][6] // 8, subnet_config["width"][6] // 8 * subnet_config["expansion_ratio"][6], in_rows = h*w)
		model.add_linear(subnet_config["width"][6] // 8 * subnet_config["expansion_ratio"][6], subnet_config["width"][6] // 8, in_rows = h*w)
		#print("After block", block, ":", c, w, h)
		block += 1
	
	#print("next")
	#h, w, c = model.add_convolution(h+1, w+1, c, subnet_config["width"][7], 3, 2)
	nh, nw, nc = model.add_convolution(h, w, c, c * subnet_config["expansion_ratio"][7], 1, 1)
	for j in range(nc):
		nh, nw, c2 = model.add_convolution(h, w, 1, 1, subnet_config["kernel_size"][7], 2)
	h, w, c = model.add_convolution(nh+1, nw+1, nc, subnet_config["width"][7], 1, 1)
	#print("After block", block, ":", c, w, h)
	block += 1
	for i in range(subnet_config["depth"][7]-1):
		model.add_attention(h * w, c // 16, subnet_config["width"][7] // 8, 16, sparsity=0, comp_amt=subnet_config["en_de_q"][7], expand_ratio=subnet_config["expansion_ratio"][7]) #, comp_amt_k=subnet_config["en_de_k"][7])
		model.add_linear(subnet_config["width"][7] // 8, subnet_config["width"][7] // 8 * subnet_config["expansion_ratio"][7], in_rows = h*w)
		model.add_linear(subnet_config["width"][7] // 8 * subnet_config["expansion_ratio"][7], subnet_config["width"][7] // 8, in_rows = h*w)
		#print("After block", block, ":", c, w, h)
		block += 1
	#print(h+1, c
	h, w, c = model.add_convolution(h+1, w+1, c, subnet_config["width"][7], 1, 1)
	#print("next")
	model.add_linear(c, subnet_config["width"][8], in_rows=h*w)
	#print("After last conv:", subnet_config["width"][8])

	return model

def create_nasvit_smallest(batch_size=1, comp_size_q=1, comp_size_k=1, sparsity=0.0):
	config = {'resolution': 192, 'width': [16, 16, 24, 32, 64, 112, 160, 208, 1792], 'depth': [1, 3, 3, 3, 3, 3, 3], 'kernel_size': [3, 3, 3, 3, 3, 3, 3], 'expand_ratio': [1, 4, 4, 4, 4, 6, 6]}
	config["expansion_ratio"] = config["expand_ratio"]
	config["use_se"] = [False, False, True, False, True, True, True]
	return NASViT_subnet_to_model(config, batch_size, comp_size_q, comp_size_k, sparsity)

def create_nasvit_random(batch_size=1):
	config = {'resolution': 288, 'width': [24, 24, 32, 32, 72, 128, 160, 208, 1984], 'depth': [2, 5, 5, 5, 8, 4, 6], 'kernel_size': [5, 3, 3, 3, 3, 3, 3], 'expand_ratio': [1, 4, 6, 4, 6, 6, 6]}
	config["expansion_ratio"] = config["expand_ratio"]
	config["use_se"] = [False, False, True, False, True, True, True]
	return NASViT_subnet_to_model(config, batch_size)

def create_nasvit_supernet(batch_size=1, comp_size_q=1, comp_size_k=1, sparsity=0.0):
	subnet_config = {
		"resolution" : 288,
		"width" : [24, 24, 32, 40, 72, 128, 184, 224, 1984],
		"depth" : [2, 5, 6, 6, 9, 8, 6],
		"kernel_size" : [5, 5, 3, 3, 3, 3, 3],
		"expansion_ratio" : [1, 6, 6, 6, 6, 6, 6],
		"use_se" : [False, False, True, False, True, True, True],
		#"depth" : [0, 2, 5, 6, 6, 9, 8, 6],
		#"kernel_size" : [0, 5, 5, 3, 3, 3, 3, 3],
		#"expansion_ratio" : [0, 1, 6, 6, 6, 6, 6, 6],
		#"en_de_q" : [-1, -1, -1, -1, -1, -1, -1, -1],
		#"en_de_k" : [-1, -1, -1, -1, -1, -1, -1, -1],
		"en_de_q" : [0, 0, 0, 0, 4, 4, 4, 4],
		"en_de_k" : [0, 0, 0, 0, 4, 4, 4, 4],
		}
	return NASViT_subnet_to_model(subnet_config, batch_size, comp_size_q, comp_size_k, sparsity)

def create_nasvit_from_config(config, batch_size=1, comp_size_q=1, comp_size_k=1):
	config["expansion_ratio"] = config["expand_ratio"]
	config["use_se"] = [False, False, True, False, True, True, True]
	return NASViT_subnet_to_model(config, batch_size, comp_size_q, comp_size_k)

def create_nasvit_a1(batch_size=1):
	subnet_config = {
		"resolution" : 192,
		"width" : [16, 16, 24, 32, 64, 112, 160, 216, 1792],
		"depth" : [1, 3, 3, 4, 3, 3, 3],
		"kernel_size" : [3, 3, 3, 3, 3, 3, 3],
		"expansion_ratio" : [3, 4, 4, 1, 1, 1, 1],
		"use_se" : [False, False, True, False, True, True, True],
		#"depth" : [1, 1, 3, 3, 4, 3, 3, 3],
		#"kernel_size" : [3, 3, 3, 3, 3, 3, 3, 3],
		#"expansion_ratio" : [-1, 3, 4, 4, 1, 1, 1, 1],
		"en_de_q" : [-1, -1, -1, -1, -1, -1, -1, -1],
		"en_de_k" : [-1, -1, -1, -1, -1, -1, -1, -1],
		}
	return NASViT_subnet_to_model(subnet_config, batch_size)

def create_nasvit_a2(batch_size=1):
	subnet_config = {
		"resolution" : 224,
		"width" : [16, 16, 24, 32, 64, 112, 160, 208, 1792],
		"depth" : [1, 3, 3, 4, 3, 5, 4],
		"kernel_size" : [3, 3, 3, 3, 3, 3, 3],
		"expansion_ratio" : [3, 4, 6, 1, 1, 1, 1],
		"use_se" : [False, False, True, False, True, True, True],
		"en_de_q" : [-1, -1, -1, -1, -1, -1, -1, -1],
		"en_de_k" : [-1, -1, -1, -1, -1, -1, -1, -1],
		}
	return NASViT_subnet_to_model(subnet_config, batch_size)

def create_nasvit_a3(batch_size=1):
	subnet_config = {
		"resolution" : 256,
		"width" : [16, 16, 24, 32, 64, 112, 160, 216, 1984],
		"depth" : [1, 1, 3, 3, 4, 4, 7, 5],
		"kernel_size" : [3, 3, 3, 3, 3, 3, 3, 3],
		"expansion_ratio" : [-1, 3, 5, 5, 1, 1, 1, 1],
		"en_de_q" : [-1, -1, -1, -1, -1, -1, -1, -1],
		"en_de_k" : [-1, -1, -1, -1, -1, -1, -1, -1],
		}
	return NASViT_subnet_to_model(subnet_config, batch_size)

def create_nasvit_a4(batch_size=1):
	subnet_config = {
		"resolution" : 288,
		"width" : [16, 16, 24, 32, 64, 120, 160, 216, 1984],
		"depth" : [1, 1, 3, 3, 4, 3, 6, 6],
		"kernel_size" : [3, 3, 3, 3, 3, 3, 3, 3],
		"expansion_ratio" : [-1, 3, 4, 6, 1, 1, 1, 1],
		"en_de_q" : [-1, -1, -1, -1, -1, -1, -1, -1],
		"en_de_k" : [-1, -1, -1, -1, -1, -1, -1, -1],
		}
	return NASViT_subnet_to_model(subnet_config, batch_size)

if __name__ == "__main__":

	nasvit = create_nasvit_smallest(1, 0.5, 0.5, 0.9)
	#nasvit = create_nasvit_a1()
	layers = model_to_layer_set(nasvit)
	#for layer in layers.layers:
	#	layer.print()
	print(layers.get_total_flops() / 1000000)
	layers.print()
	#model = get_LeViT_128(1, 0.5, 0.5, 0.9)
	#model = get_DeiT_Tiny(1, 0.34, 0.34, 0.9)
	#model.print_out_sizes()
	#layers = model_to_layer_set(model)
	#print(layers.get_total_params())
	#layers.print()
	#print(layers.get_total_flops_including_extras() / 512)
	#layers.print()
	#flops = layers.layers[0].get_flops() + layers.layers[1].get_flops() + layers.layers[2].get_flops() + layers.layers[3].get_flops()
	#print(flops, flops / 512)
	#print(layers.get_total_flops() / 512)
	# Conv FLOPs
	#print(layers.layers[0].get_flops() + layers.layers[1].get_flops() + layers.layers[2].get_flops() + layers.layers[3].get_flops())
