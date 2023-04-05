import pickle

from test_wrappers import AccelWrapperTest, AlgWrapperTest
from NASViT_wrapper import NASViTWrapper
from fast_arch_wrapper import FastArchWrapper

def run_random_search(accel, alg, calc_score, iterations, latency_boxes):
	accel.init(512, 320000 // 2, 0.5, 77)
	alg.init()
	
	boxes = [[None, None, None, None, 0, 0, 0, 0] for i in range(len(latency_boxes))]
	
	for i in range(iterations):
		print("***"*10)
		print("Starting iteration", i)
		print("***"*10)
	
		model_config, accuracy, flops = alg.generate_random_model()
		#model_config = {'resolution': 288, 'width': [24, 24, 32, 32, 72, 128, 160, 208, 1984], 'depth': [2, 5, 5, 5, 8, 4, 6], 'kernel_size': [5, 3, 3, 3, 3, 3, 3], 'expand_ratio': [1, 4, 6, 4, 6, 6, 6]}
		hw_config, params, latency, power = accel.generate_random_accel(model_config)
		
		score = calc_score(accuracy, latency, power)
		
		print("***"*10)
		print("Finished iteration", i)
		print("Score:", score)
		print("Accuracy:", accuracy)
		print("FLOPs:", flops)
		print("Latency:", latency)
		print("Power:", power)
		
		# find which latency box it's in
		for idx, (mi, ma) in enumerate(latency_boxes):
			if latency >= mi and latency <= ma:
				# if it's the high score, mark it
				if boxes[idx][0] == None or boxes[idx][0] < score:
					boxes[idx] = [score, model_config, hw_config, params, accuracy, flops, latency, power]
					print("Best score in box", idx, ":", mi, "-", ma)
				break
		
		
		print("***"*10)
	
	for idx, (score, model_config, hw_config, params, accuracy, flops, latency, power) in enumerate(boxes):
		if score != None:
			accuracy, flops = alg.full_eval(model_config, accuracy, flops)
			latency, power = accel.full_eval(model_config, hw_config, params, latency, power)
			boxes[idx][4] = accuracy
			boxes[idx][5] = flops
			boxes[idx][6] = latency
			boxes[idx][7] = power
	
	return boxes

def latency_calc_score(accuracy, latency, power):
	return latency

def accuracy_calc_score(accuracy, latency, power):
	return accuracy

if __name__ == "__main__":
	flops_boxes = [[0, 300], [300, 400], [400, 500], [500, 600], [600, 1000], [1000, 100000]] # matching NASViT
	
	latency_boxes = []
	num_PEs = 512
	util = .6
	clock_speed = 500000000
	for (mi, ma) in flops_boxes:
		ideal_min = mi * 1000000 / (num_PEs*util)
		ideal_max = ma * 1000000 / (num_PEs*util)
		latency_boxes.append([ideal_min / clock_speed, ideal_max / clock_speed])
	#print(latency_boxes)
	
	boxes = run_random_search(FastArchWrapper(), NASViTWrapper(), accuracy_calc_score, 50, latency_boxes)

	print(boxes)
	with open("first_res.pickle", 'wb') as out_file:
		pickle.dump(boxes, out_file, pickle.HIGHEST_PROTOCOL)
