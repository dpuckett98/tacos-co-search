'''
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print("Hello from", rank)
'''

import random

def gen_random_set(size):
	res = []
	for i in range(size):
		res.append(random.random())
	return res

pool = gen_random_set(100)

for i in range(500):
	
	pool.sort()
	
	pool = pool + gen_random_set(10)
	
	pool.sort()
	
	del pool[:10]
	print(len(pool))

print(pool)