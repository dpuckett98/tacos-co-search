# pool_size is the size of the pool
# num_generations is the number of generations searched
# growth_rate is a percentage; growth_rate% are removed each generation, and growth_rate% are mutated each generation
# mutate_rate is a percentage; it's the chance that each element in the config is changed during mutation
# sample_and_eval randomly samples & evaluates an entity
# mutate_and_eval randomly mutates & evaluates an entity (given "mutate_rate" as a parameter)
# eval_fitness evaluates the fitness of an entity
# entity config: [[actual results], [configuration]] (higher fitness score is better)
def run_evolutionary_search(pool_size, num_generations, growth_rate, mutate_rate, sample_and_eval, mutate_and_eval, eval_fitness):
	# setup
	growth_amt = int(pool_size * growth_rate)

	# generate initial pool
	print("***"*10)
	print("Generating initial pool of", pool_size, "entities")
	print("***"*10)
	pool = [sample_and_eval() for i in range(pool_size)]
	#fitness_scores = [eval_fitness(p) for p in pool]
	
	# for each generation...
	for g in range(num_generations):
		print("***"*10)
		print("Starting generation", g+1, "out of", num_generations)
		
		# sort the population by fitness
		#to_sort = [[i, j] for i, j in zip(fitness_scores, pool)]
		#res = sorted(to_sort)
		#fitness_scores = [r[0] for r in res]
		pool = sorted(pool, key=eval_fitness) #[r[1] for r in res]
		print("Current best entity:", pool[0])
		print("***"*10)
		
		# mutate the top x%
		new_entities = [mutate_and_eval(p, mutate_rate) for p in pool[-growth_amt:]]
		
		# evaluate the fitness of the mutated top x%
		new_entities_fitness = [eval_fitness(p) for p in new_entities]
		
		# remove the bottom x%
		del pool[:growth_amt]
		
		# add the mutated top x% to the pool
		pool = pool + new_entities
	
	# sort and return the entire pool
	pool = sorted(pool, key=eval_fitness)
	return pool