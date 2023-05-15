#from NASViT.main import start, generate_model
import pickle

def get_curated_results(dense=True):
	if dense:
		file = open("dense_subnets.pickle", "rb")
	else:
		file = open("sparse_subnets.pickle", "rb")

	res = []

	while True:
		try:
			r = pickle.load(file)
			res.append(r)
		except:
			break

	#print(len(res))

	curated_list = []
	last_accuracy = -1
	for r in res:
		if r[2] != last_accuracy:
			last_accuracy = r[2]
			curated_list.append(r)
	#print(len(curated_list))
	#print(curated_list[:5])
	return curated_list

class NASViTWrapper2:
	# does whatever init the accelerator needs
	def init(self, dense=True):
		print("Initializing NASViT results...")
		self.data = get_curated_results(dense)
		print(len(self.data), "total results available")
		self.index = 0
		#config, model, data_loader_train, data_loader_val = start()
		#self.config = config
		#self.model = model
		#self.data_loader_train = data_loader_train
		#self.data_loader_val = data_loader_val
	
	# returns a randomly generated model configuration & its accuracy
	def generate_random_model(self):
		subnet_cfg, flops, acc1 = self.data[self.index]
		self.index += 1
		#subnet_cfg, flops, acc1 = generate_model(self.config, self.model, self.data_loader_train, self.data_loader_val)
		return subnet_cfg, acc1, flops
	
	def full_eval(self, model_config, prev_accuracy, prev_flops):
		return [prev_accuracy, prev_flops]