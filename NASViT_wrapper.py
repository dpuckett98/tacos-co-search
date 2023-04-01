from NASViT.main import start, generate_model

class NASViTWrapper:
	# does whatever init the accelerator needs
	def init(self):
		print("Initializing NASViT...")
		config, model, data_loader_train, data_loader_val = start()
		self.config = config
		self.model = model
		self.data_loader_train = data_loader_train
		self.data_loader_val = data_loader_val
	
	# returns a randomly generated model configuration & its accuracy
	def generate_random_model(self):
		subnet_cfg, flops, acc1 = generate_model(self.config, self.model, self.data_loader_train, self.data_loader_val)
		return subnet_cfg, acc1, flops
	
	def full_eval(self, model_config, prev_accuracy, prev_flops):
		return [prev_accuracy, prev_flops]