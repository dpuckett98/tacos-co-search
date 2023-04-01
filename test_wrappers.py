import random

# Test classes

class AccelWrapperTest:
	# does whatever init the accelerator needs
	def init(self):
		print("Accel Wrapper Test init")
	
	# takes the model configuration, and returns [hw_config, params, latency, power] of a randomly-generated accelerator running that model
	def generate_random_accel(self, model_config):
		hw_config = [random.choice([0, 1, 2, 3]) for i in range(5)]
		return [hw_config, None, sum(hw_config), 0]
	
	# does a full evaluation of the given model run on the given accelerator (if this is the same as the latency & power generated earlier, can just pass the inputs directly to the outputs)
	def full_eval(self, model_config, hw_config, param_config, prev_latency, prev_power):
		return [prev_latency, prev_power]

class AlgWrapperTest:
	def init(self):
		print("Alg Wrapper Test init")
	
	# returns a randomly generated model configuration & its accuracy
	def generate_random_model(self):
		return [None, random.choice([80, 81, 82, 83, 84, 85]), 1000]
	
	def full_eval(self, model_config, prev_accuracy, prev_flops):
		return [prev_accuracy, prev_flops]