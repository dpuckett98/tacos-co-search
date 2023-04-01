import math

class Model:

	def __init__(self, batch_size):
		self.layers = []
		self.batch_size = batch_size

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
	def add_convolution(self, in_width, in_height, in_channels, out_channels, filter_dim, step_size):
		# add padding
		while (in_width - filter_dim + step_size) // step_size != (in_width - filter_dim + step_size) / step_size:
			in_width += 1
		while (in_height - filter_dim + step_size) // step_size != (in_height - filter_dim + step_size) / step_size:
			in_height += 1
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
	def add_attention(self, in_tokens, token_size, feature_dim, num_heads, comp_amt=-1, sparsity=0.0, expand_ratio = 1):
		self.layers.append({
			"type" : "attention",
			"batch_size" : self.batch_size,
			"in_tokens" : in_tokens,
			"token_size" : token_size,
			"feature_dim" : feature_dim,
			"num_heads" : num_heads,
			"comp_amt" : comp_amt,
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
				print(idx, ":", layer["in_rows"], layer["out_elements"], "(linear)")
			if layer["type"] == "convolution":
				out_width = (layer["in_width"] - layer["filter_dim"] + layer["step_size"]) // layer["step_size"]
				out_height = (layer["in_height"] - layer["filter_dim"] + layer["step_size"]) // layer["step_size"]
				print(idx, ":", layer["out_channels"], out_width, out_height, "(convolution)")
			if layer["type"] == "attention":
				print(idx, ":", layer["in_tokens"], layer["token_size"], "(attention)")

# encode/decode variables signal whether or not the layer should be encoded after computing or decoded before computing; encode_dim contains info about how the encoding/decoding dimensions
# sparse_map is None if the Layer is dense; otherwise, it is a 2D matrix of bools of size A_rows x B_cols, signaling whether or not the corresponding output element is sparse
class Layer:
	def __init__(self, A_rows, A_cols_B_rows, B_cols, encode=False, decode=False, orig_head_dim=-1, comp_head_dim=-1, sparsity=0.0, A_weights=False, B_weights=False, init_weights=0, flags = {}):
		self.A_rows = A_rows
		self.A_cols_B_rows = A_cols_B_rows
		self.B_cols = B_cols
		self.encode = encode
		self.decode = decode
		self.orig_head_dim = orig_head_dim
		self.comp_head_dim = comp_head_dim
		self.sparsity = sparsity
		self.init_weights = init_weights
		self.A_weights = A_weights
		self.B_weights = B_weights
		
		self.actual_cycles = -1
		self.actual_memory_accesses = -1
		self.flags = flags
	
	def equals(self, other):
		return self.A_rows == other.A_rows and self.A_cols_B_rows == other.A_cols_B_rows and self.B_cols == other.B_cols and self.encode == other.encode and self.decode == other.decode and self.sparsity == other.sparsity
	
	def get_flops(self):
		#if self.encode or self.decode or self.sparse_map != None:
		#	raise Exception("Encode/decode and sparse attention not supported yet")
		return self.A_rows * self.A_cols_B_rows * self.B_cols #* (1 - self.sparsity)
	
	def get_flops_including_extras(self):
		flops = 0
		if self.encode:
			flops += self.A_rows * self.B_cols * self.comp_head_dim
		if self.decode:
			flops += (self.comp_head_dim * self.orig_head_dim) * (self.A_rows // self.orig_head_dim * self.A_cols_B_rows) # number of computations per tile * number of tiles
			flops += (self.comp_head_dim * self.orig_head_dim) * (self.A_cols_B_rows // self.orig_head_dim * self.B_cols) # number of computations per tile * number of tiles
		flops += self.A_rows * self.A_cols_B_rows * self.B_cols * (1 - self.sparsity)
		return flops
	
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
		return self.get_flops() / num_PEs / self.actual_cycles
	
	def print(self):
		print(self.get_string(), end='')
		#print("Flags:" + str(self.flags) + ", Dims:", str(self.A_rows) + ",", str(self.B_cols) + ",", str(self.A_cols_B_rows), "FLOPS:", self.get_flops(), "Params:", self.get_params(), "Enc/Dec:", self.comp_head_dim / self.orig_head_dim, "Sparsity:", self.sparsity)
	
	def get_string(self):
		descrip = "Dims: " + str(self.A_rows) + ", " + str(self.B_cols) + ", " + str(self.A_cols_B_rows) + " FLOPS: " + str(self.get_flops()) + " Params: " + str(self.get_params()) + " Enc/Dec: " + str(self.comp_head_dim / self.orig_head_dim) + " Sparsity: " + str(self.sparsity) + ", Flags:" + str(self.flags) + "\n"
		return descrip
	
	def get_string_no_nl(self):
		descrip = "Dims: " + str(self.A_rows) + ", " + str(self.B_cols) + ", " + str(self.A_cols_B_rows) + " FLOPS: " + str(self.get_flops()) + " Params: " + str(self.get_params()) + " Enc/Dec: " + str(self.comp_head_dim / self.orig_head_dim) + " Sparsity: " + str(self.sparsity) + ", Flags:" + str(self.flags)
		return descrip

class LayerSet:
	
	def __init__(self, layers):
		self.layers = layers
		
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
		ideal_cycles = self.get_total_flops() / num_PEs
		return ideal_cycles / self.get_total_cycles() * 100
	
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
		print("Number of layers:", len(self.layers))
		print("Number of unique layers:", len(self.unique_layers))
		for layer in self.unique_layers:
			print(layer[1], end=": ")
			layer[0].print()
	
	def print_stats(self, num_PEs):
		print("Layer Set stats:")
		total_cycles = self.get_total_cycles()
		print("Total cycles:", total_cycles)
		print("Average utilization: {:.2f}%".format(self.get_utilization(num_PEs)))
		print("Number of layers:", len(self.layers))
		print("Number of unique layers:", len(self.unique_layers))
		for layer in self.unique_layers:
			print("--- Layer ({}x) ---".format(layer[1]))
			layer[0].print()
			print("Cycles:", layer[0].get_actual_cycles())
			print("Utilization: {:.2f}%".format(layer[0].get_utilization(num_PEs) * 100))
			print("% of total cycles: {:.2f}%".format(layer[0].get_actual_cycles() * layer[1] / total_cycles * 100))
	
	def get_string_stats(self, num_PEs, params):
		descrip = "Layer Set stats:\n"
		descrip += "Total cycles: " + str(self.get_total_cycles()) + "\n"
		descrip += "Average utilization: {:.2f}%\n".format(self.get_utilization(num_PEs))
		descrip += "Number of layers: " + str(len(self.layers)) + "\n"
		descrip += "Number of unique layers: " + str(len(self.unique_layers)) + "\n"
		for layer, p in zip(self.unique_layers, params):
			descrip += "--- Layer (" + str(layer[1]) + "x) ---\n"
			descrip += layer[0].get_string()
			descrip += "Cycles: " + str(layer[0].get_actual_cycles()) + "\n"
			descrip += "Utilization: {:.2f}%\n".format(layer[0].get_utilization(num_PEs) * 100)
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
			if layer_descrip["comp_amt"] != -1:
				# Q, and K
				for i in range(layer_descrip["batch_size"]):
					# multiply the same input embedding (in_tokens x token_size) by QKV weights for each head (e.g. 3 x num_heads matrices of size token_size x feature_dim)
					layers.append(Layer(layer_descrip["in_tokens"], layer_descrip["token_size"], layer_descrip["feature_dim"] * layer_descrip["num_heads"] * 2, B_weights=True, encode=True, orig_head_dim=layer_descrip["num_heads"], comp_head_dim=layer_descrip["comp_amt"], flags={"type":"attention", "part":"Q_K"}))
				# V
				for i in range(layer_descrip["batch_size"]):
					layers.append(Layer(layer_descrip["in_tokens"], layer_descrip["token_size"], layer_descrip["feature_dim"] * layer_descrip["num_heads"] * layer_descrip["expand_ratio"], B_weights=True, encode=False, orig_head_dim=layer_descrip["num_heads"], comp_head_dim=layer_descrip["comp_amt"], flags={"type":"attention", "part":"V"}))
					#layers.append(Layer(layer_descrip["in_tokens"] * layer_descrip["num_heads"] * layer_descrip["expand_ratio"]
				
				# compute score
				for i in range(layer_descrip["batch_size"]): # * layer_descrip["num_heads"]):
					layers.append(Layer(layer_descrip["in_tokens"], layer_descrip["feature_dim"], layer_descrip["in_tokens"] * layer_descrip["num_heads"], sparsity=layer_descrip["sparsity"], decode=True, orig_head_dim=layer_descrip["num_heads"], comp_head_dim=layer_descrip["comp_amt"], flags={"type":"attention", "part":"score"}))
			else:
				# Q and K
				for i in range(layer_descrip["batch_size"]):
					# multiply the same input embedding (in_tokens x token_size) by QKV weights for each head (e.g. 3 x num_heads matrices of size token_size x feature_dim)
					layers.append(Layer(layer_descrip["in_tokens"], layer_descrip["token_size"], layer_descrip["feature_dim"] * layer_descrip["num_heads"] * 2, B_weights=True, flags={"type":"attention", "part":"Q_K"}))
				# V
				for i in range(layer_descrip["batch_size"]):
					layers.append(Layer(layer_descrip["in_tokens"], layer_descrip["token_size"], layer_descrip["feature_dim"] * layer_descrip["num_heads"] * layer_descrip["expand_ratio"], B_weights=True, encode=False, flags={"type":"attention", "part":"V"}))
				
				# compute score
				for i in range(layer_descrip["batch_size"] * layer_descrip["num_heads"]):
					layers.append(Layer(layer_descrip["in_tokens"], layer_descrip["feature_dim"], layer_descrip["in_tokens"], sparsity=layer_descrip["sparsity"], flags={"type":"attention", "part":"score"}))
			
			# ignoring softmax
			
			# multiply by value vectors
			for i in range(layer_descrip["batch_size"] * layer_descrip["num_heads"]):
				layers.append(Layer(layer_descrip["in_tokens"], layer_descrip["in_tokens"], layer_descrip["feature_dim"], sparsity=layer_descrip["sparsity"], flags={"type":"attention", "part":"attention"}))
			
			# out MLP (DeiT and Transformer include this in MHA)
			for i in range(layer_descrip["batch_size"]):
				layers.append(Layer(layer_descrip["in_tokens"], layer_descrip["feature_dim"] * layer_descrip["num_heads"], layer_descrip["token_size"], B_weights=True, flags={"type":"attention", "part":"MLP"}))
	
	ls = LayerSet(layers)
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
def get_LeViT(batch_size, in_dim, QK_dim, c_1, n_1, count_1, c_2, n_2, count_2, c_3, n_3, count_3, comp_ratio=1.0, sparsity=0.0):
	levit = Model(batch_size)
	
	# convolution blocks
	w, h, c = levit.add_convolution(in_dim, in_dim, 3, 16, 3, 2)
	w, h, c = levit.add_convolution(w, h, c, 32, 3, 2)
	w, h, c = levit.add_convolution(w, h, c, 64, 3, 2)
	w, h, c = levit.add_convolution(w, h, c, 128, 3, 2)
	
	# stage 1
	#print(c_1, c)
	#assert c_1 == c
	for i in range(count_1):
		if comp_ratio == 1.0:
			levit.add_attention(w * h, c_1, QK_dim, n_1, sparsity=sparsity)
		else:
			comp_amt = math.ceil(n_1 * comp_ratio)
			levit.add_attention(w * h, c_1, QK_dim, n_1, sparsity=sparsity, comp_amt=comp_amt)
		levit.add_linear(c_1, c_1 * 2, in_rows = w*h)
		levit.add_linear(c_1 * 2, c_1, in_rows = w*h)
	levit.add_attention(w * h, c_1, QK_dim, n_1 * 2) # this is actually supposed to be a "Shrinking Attention Block" but it's less computation heavy than regular attention
	
	w = math.ceil(w / 2)
	h = math.ceil(h / 2)
	
	# stage 2
	levit.add_linear(c_2, c_2 * 2, in_rows = w*h)
	levit.add_linear(c_2 * 2, c_2, in_rows = w*h)
	for i in range(count_2):
		if comp_ratio == 1.0:
			levit.add_attention(w * h, c_2, QK_dim, n_2, sparsity=sparsity)
		else:
			comp_amt = math.ceil(n_2 * comp_ratio)
			levit.add_attention(w * h, c_2, QK_dim, n_2, sparsity=sparsity, comp_amt=comp_amt)
		levit.add_linear(c_2, c_2 * 2, in_rows = w*h)
		levit.add_linear(c_2 * 2, c_2, in_rows = w*h)
	levit.add_attention(w * h, c_2, QK_dim, n_2 * 2) # this is actually supposed to be a "Shrinking Attention Block" but it's less computation heavy than regular attention
	
	w = math.ceil(w / 2)
	h = math.ceil(h / 2)
	
	# stage 3
	levit.add_linear(c_3, c_3 * 2, in_rows = w*h)
	levit.add_linear(c_3 * 2, c_3, in_rows = w*h)
	for i in range(count_3):
		if comp_ratio == 1.0:
			levit.add_attention(w * h, c_3, QK_dim, n_3, sparsity=sparsity)
		else:
			comp_amt = math.ceil(n_3 * comp_ratio)
			levit.add_attention(w * h, c_3, QK_dim, n_3, sparsity=sparsity, comp_amt=comp_amt)
		levit.add_linear(c_3, c_3 * 2, in_rows = w*h)
		levit.add_linear(c_3 * 2, c_3, in_rows = w*h)
	levit.add_attention(w * h, c_3, QK_dim, n_3 * 2) # this is actually supposed to be a "Shrinking Attention Block" but it's less computation heavy than regular attention
	
	return levit

def get_LeViT_128(batch_size, comp_ratio, sparsity):
	return get_LeViT(batch_size, 224, 16, 128, 4, 4, 256, 8, 4, 384, 12, 4, comp_ratio, sparsity)

def get_LeViT_192(batch_size, comp_ratio, sparsity):
	return get_LeViT(batch_size, 224, 32, 192, 3, 4, 288, 5, 4, 384, 6, 4, comp_ratio, sparsity)

def get_LeViT_256(batch_size, comp_ratio, sparsity):
	return get_LeViT(batch_size, 224, 32, 256, 4, 4, 384, 6, 4, 512, 8, 4, comp_ratio, sparsity)

def get_LeViT_384(batch_size, comp_ratio, sparsity):
	return get_LeViT(batch_size, 224, 32, 384, 6, 4, 512, 9, 4, 768, 12, 4, comp_ratio, sparsity)

def get_DeiT_attention(batch_size, in_dim, emb_dim, num_heads, num_layers, comp_ratio=1.0, sparsity=0.0):
	deit = Model(batch_size)

	dim_per_head = 64
	patch_size = 16
	
	in_tokens = (in_dim // patch_size) ** 2
	if comp_ratio == 1.0:
		for i in range(num_layers):
			deit.add_attention(in_tokens, emb_dim, dim_per_head, num_heads, sparsity=sparsity)
			deit.add_linear(emb_dim, emb_dim * 4, in_rows = in_tokens)
			deit.add_linear(emb_dim * 4, emb_dim, in_rows = in_tokens)
	else:
		comp_amt = math.ceil(num_heads * comp_ratio)
		#print(comp_amt)
		for i in range(num_layers):
			deit.add_attention(in_tokens, emb_dim, dim_per_head, num_heads, sparsity=sparsity, comp_amt=comp_amt)
			deit.add_linear(emb_dim, emb_dim * 4, in_rows = in_tokens)
			deit.add_linear(emb_dim * 4, emb_dim, in_rows = in_tokens)
	
	return deit

def get_DeiT_Tiny(batch_size, comp_ratio, sparsity):
	return get_DeiT_attention(batch_size, 224, 192, 3, 12, comp_ratio, sparsity)

def get_DeiT_Small(batch_size, comp_ratio, sparsity):
	return get_DeiT_attention(batch_size, 224, 384, 6, 12, comp_ratio, sparsity)

def get_DeiT_Base(batch_size, comp_ratio, sparsity):
	return get_DeiT_attention(batch_size, 224, 768, 12, 12, comp_ratio, sparsity)

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
	test.add_attention(100, 100, 100, 10, sparsity=0.5, comp_amt=5)
	test.add_linear(100, 100, 100)
	#test.add_linear(105, 100, 100)
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

# TODO: follow nasvit code
def get_MBConvLayer(w, h, c):
	pass

# TODO: follow nasvit code
def get_SwinTransformerLayer(w, h, c):
	pass

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

def create_nasvit_supernet(batch_size=1):
	subnet_config = {
		"resolution" : 288,
		"width" : [24, 24, 32, 40, 72, 128, 184, 224, 1984],
		"depth" : [0, 2, 5, 6, 6, 9, 8, 6],
		"kernel_size" : [0, 5, 5, 3, 3, 3, 3, 3],
		"expansion_ratio" : [0, 1, 6, 6, 6, 6, 6, 6],
		#"en_de_q" : [-1, -1, -1, -1, -1, -1, -1, -1],
		#"en_de_k" : [-1, -1, -1, -1, -1, -1, -1, -1],
		"en_de_q" : [0, 0, 0, 0, 4, 4, 4, 4],
		"en_de_k" : [0, 0, 0, 0, 4, 4, 4, 4],
		}
	return subnet_to_model(subnet_config, batch_size)

def create_nasvit_a1(batch_size=1):
	subnet_config = {
		"resolution" : 192,
		"width" : [16, 16, 24, 32, 64, 112, 160, 216, 1792],
		"depth" : [1, 1, 3, 3, 4, 3, 3, 3],
		"kernel_size" : [3, 3, 3, 3, 3, 3, 3, 3],
		"expansion_ratio" : [-1, 3, 4, 4, 1, 1, 1, 1],
		"en_de_q" : [-1, -1, -1, -1, -1, -1, -1, -1],
		"en_de_k" : [-1, -1, -1, -1, -1, -1, -1, -1],
		}
	return subnet_to_model(subnet_config, batch_size)

def create_nasvit_a2(batch_size=1):
	subnet_config = {
		"resolution" : 224,
		"width" : [16, 16, 24, 32, 64, 112, 160, 208, 1792],
		"depth" : [1, 1, 3, 3, 4, 3, 5, 4],
		"kernel_size" : [3, 3, 3, 3, 3, 3, 3, 3],
		"expansion_ratio" : [-1, 3, 4, 6, 1, 1, 1, 1],
		"en_de_q" : [-1, -1, -1, -1, -1, -1, -1, -1],
		"en_de_k" : [-1, -1, -1, -1, -1, -1, -1, -1],
		}
	return subnet_to_model(subnet_config, batch_size)

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
	return subnet_to_model(subnet_config, batch_size)

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
	return subnet_to_model(subnet_config, batch_size)

#nasvit = create_nasvit_supernet()
#layers = model_to_layer_set(nasvit)
#layers.print()
#print(layers.get_total_flops())

#model = get_LeViT_128(1, 1.0, 0.0)
#model.print_out_sizes()
#layers = model_to_layer_set(model)
#layers.print()
#print(layers.get_total_flops_including_extras() / 512)
#print(layers.get_total_flops())