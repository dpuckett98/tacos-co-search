# pool_size is the size of the pool
# num_generations is the number of generations searched
# growth_rate is a percentage; growth_rate% are removed each generation, and growth_rate% are mutated each generation
# mutate_rate is a percentage; it's the chance that each element in the config is changed during mutation
# sample_and_eval randomly samples & evaluates an entity
# mutate_and_eval randomly mutates & evaluates an entity (given "mutate_rate" as a parameter)
# eval_fitness evaluates the fitness of an entity
# entity config: [[actual results], [configuration]] (higher fitness score is better)
def run_evolutionary_search(boxes, pool_size, num_generations, growth_rate, mutate_rate, sample_and_eval, mutate_and_eval, eval_fitness):
	# setup
	#growth_amt = int(pool_size * growth_rate)

	# generate initial pool
	print("***"*10)
	print("Generating initial pool of", pool_size*len(boxes), "entities")
	print("***"*10)
	pool_list = [[] for i in range(len(boxes))]
	#running = True
	#while running:
	for i in range(len(boxes) * pool_size):
		entity = sample_and_eval()

		added = False
		print(entity)
		for idx, (mi, ma) in enumerate(boxes):
			print(mi, ma)
			if entity[0][0] >= mi and entity[0][0] <= ma:
				pool_list[idx].append(entity)
				added = True
				break
		if not added:
			print("Failed!!!")
			quit()

		#running = Fale
		#for p in pool_list:
		#	if len(p) < pool_size:
		#		running = True
		#		break
	#pool = [sample_and_eval() for i in range(pool_size)]
	
	# for each generation...
	for g in range(num_generations):
		print("***"*10)
		print("Starting generation", g+1, "out of", num_generations)
		print("***"*10)
		for i in range(len(pool_list)): #enumerate(pool_list):
			if len(pool_list[i]) == 0:
				continue
			growth_amt = int(growth_rate * len(pool_list[i]))
			pool = sorted(pool_list[i], key=eval_fitness, reverse=True) #[r[1] for r in res]
			print("***"*10)
			print("Starting box", i)
			print("Current best entity:", pool[0])
			print("***"*10)
			
			# mutate the top x%
			new_entities = [mutate_and_eval(p, mutate_rate) for p in pool[-growth_amt:]]
			
			# remove the bottom x%
			del pool[:growth_amt]
			
			# add the mutated top x% to the pool
			pool = pool + new_entities

			# replace in pool_list
			pool_list[i] = pool
	
	# sort and return the entire pool
	for i in range(len(pool_list)):
		pool_list[i] = sorted(pool_list[i], key=eval_fitness, reverse=True)
	return pool_list
