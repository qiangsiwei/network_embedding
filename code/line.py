# -*- encoding:utf-8 -*-

import fileinput
from collections import defaultdict

class LINE():

	def __init__(self, network_file, embedding_file, dim, order, num_negative, num_samples, num_threads, init_rho):
		self.network_file = network_file
		self.embedding_file = embedding_file
		self.dim = dim
		self.order = order
		self.num_negative = num_negative
		self.num_samples = num_samples
		self.num_threads = num_threads
		self.init_rho = init_rho
		self.sigmoid_bound = 6
		self.sigmoid_table_size = 1e3
		self.negative_table_size = 1e8
		self.neg_sampling_power = 0.75
		self.hash_table = {}
		self.vertex = defaultdict(lambda:{'name':'','degree':0})
		self.edge_source_id = []
		self.edge_target_id = []
		self.edge_weight = []
		self.num_edges = 0
		self.prob = []
		self.alias = []

	def ReadData(self):
		for line in fileinput.input(self.network_file):
			name_v1, name_v2, weight = line.strip().split(); weight = float(weight)
			for name in (name_v1, name_v2):
				self.hash_table[name] = self.hash_table.get(name) or len(self.hash_table)
				self.vertex[self.hash_table[name]]['name'] = name
				self.vertex[self.hash_table[name]]['degree'] += weight
			self.edge_source_id.append(name_v1)
			self.edge_target_id.append(name_v2)
			self.edge_weight.append(weight) 
		fileinput.close()
		self.num_edges = len(self.edge_weight)

	def InitAliasTable(self):
		self.prob, self.alias = [0]*self.num_edges, [0]*self.num_edges
		small_block, large_block = [], []
		total = sum(self.edge_weight)
		norm_prob = [1.*weight*self.num_edges/total for weight in self.edge_weight]
		for k in range(self.num_edges)[::-1]:
			if norm_prob[k]<1: 
				small_block.append(k)
			else: 
				large_block.append(k)
		while small_block and large_block:
			cur_small_block, cur_large_block = small_block.pop(), large_block.pop()
			self.prob[cur_small_block], self.alias[cur_small_block] = norm_prob[cur_small_block], cur_large_block
			norm_prob[cur_large_block] = norm_prob[cur_large_block]+norm_prob[cur_small_block]-1
			if norm_prob[cur_large_block] < 1: 
				small_block.append(cur_large_block)
			else: 
				large_block.append(cur_large_block)
		while large_block: self.prob[large_block.pop()] = 1
		while small_block: self.prob[small_block.pop()] = 1
		print self.prob, self.alias

	def SampleAnEdge(self, rand_value1, rand_value2):
		k = int(num_edges*rand_value1);
		return k if rand_value2<prob[k] else alias[k]

	def InitVector(self):
		pass

	def InitNegTable(self):
		pass

	def InitSigmoidTable(self):
		pass

	def FastSigmoid(self, x):
		pass

	def Update(self, vec_u, vec_v, vec_error, label):
		pass

	def TrainLINEThread():
		pass

	def Output(self):
		pass

	def TrainLINE(self):
		self.ReadData()
		self.InitAliasTable()
		pass


if __name__ == '__main__':
	network_file = 'net_dense.txt'; embedding_file = 'vec_2nd.txt'
	dim = 100; order = 2; num_negative = 10; num_samples = 10**6; num_threads = 20; init_rho = 0.025
	line = LINE(network_file, embedding_file, dim, order, num_negative, num_samples, num_threads, init_rho)
	line.TrainLINE()

