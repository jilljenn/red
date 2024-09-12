#coding:utf-8

import numpy as np

## Gaussian Processes
from sklearn.gaussian_process.kernels import DotProduct, RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor

### ALGORITHMS

class Heuristic(object):
	def __init__(self):
		pass
		
	def fit(self, ratings, item_embeddings):
		raise NotImplemented
		
	def predict(self, user, user_context, k, item_embeddings, item_categories=None):
		raise NotImplemented
		
	def allocation(self, user, user_context, item_embeddings):
		raise NotImplemented
		
	def update(self, user, item, reward, diversity_intra, diversity_inter):
		raise NotImplemented

class GaussianProcess(Heuristic):
	def __init__(self):
		self.gps = None
		self.name = "GaussianProcess"
		
	def fit(self, ratings, item_embeddings):
		#kernel = DotProduct(1.0, (1e-3, 1e3))
		kernel = RBF(1.0, (1e-3, 1e3))
		users = np.unique(ratings[:,0]).astype(int).tolist()
		self.gps = {u : GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2) for u in users}
		for user in users:
			gp = self.gps[user]	
			ratings_user0 = ratings[ratings[:,0].astype(int)==user, :]
			X_user0 = item_embeddings[ratings_user0[:, 1].astype(int)].astype(float)
			y_user0 = ratings_user0[:, -1].astype(float).ravel()
			gp.fit(X_user0, y_user0)
			
	def predict(self, user, user_context, k, item_embeddings, item_categories=None):
		y_pred, y_std = self.gps[user].predict(item_embeddings, return_std=True)
		#for i, (item, pred, std) in enumerate(zip(range(len(item_embeddings)), y_pred, y_std)):
		#	print(i, item, pred.round(5), std.round(5), ((1+pred) * std).round(5))
		scores = np.multiply((1+y_pred), y_std)
		return np.argsort(scores)[(-k):].tolist(), np.sort(scores)[(-k):].tolist()
		
	def allocation(self, user, user_context, item_embeddings):
		y_pred, y_std = self.gps[user].predict(item_embeddings, return_std=True)
		scores = np.multiply((1+y_pred), y_std)
		scores -= np.min(scores)
		scores /= np.sum(scores)
		return scores
		
	def update(self, user, item, reward, diversity_intra, diversity_inter):
		pass # no update
		
### ORACLES
		
class Oracle(object):
	def __init__(self):
		self.generator = None
		self.name = "Oracle"
		
	def fit(self, generator=None):
		assert generator is not None
		self.booking_ui = generator["booking_ui"]
		self.delta_ui = generator["delta_ui"]
		self.delta_ii = generator["delta_ii"]
		self.generator = generator
		
	def predict(self, user, user_context, k, item_embeddings, item_categories=None):
		raise NotImplemented
		
	def reward(self, item_emb, u, context, item_cat):
		b, c = self.booking_ui(item_emb, u, context)
		d = self.delta_ui(item_emb, item_cat, context)
		return [b, c, d]
		
	def diversity(self, item_embeddings, item_categories, batch1, batch2=None):
		d_batch = 0
		if (batch2 is None):
			Ni = len(batch1)
			for ii, i in enumerate(batch1[:-1]):
				for j in batch1[(ii+1):]:
					d_batch += 2*self.delta_ii(item_embeddings[i], item_embeddings[j], item_categories[i], item_categories[j])
			d_batch /= Ni*Ni-Ni
		else:
			d_batch = np.mean([self.delta_ii(item_embeddings[i], item_embeddings[j], item_categories[i], item_categories[j])  for i in batch1 for j in batch2])
			d_batch /= len(batch1)*len(batch2)
		return d_batch
		
	def update(self, user, item, reward, diversity_intra, diversity_inter):
		pass # no update

class TrueRewardPolicy(Oracle):
	def __init__(self):
		super().__init__()
		self.name = "TrueRewardPolicy"
		
	def predict(self, user, user_context, k, item_embeddings, item_categories=None):
		nitems = len(item_embeddings)
		scores = [self.booking_ui(item_embeddings[i], user, user_context)[0] for i in range(nitems)]
		return np.argsort(scores)[(-k):].tolist(), np.sort(scores)[(-k):].tolist()
		
	def allocation(self, user, user_context, item_embeddings, item_categories=None):
		nitems = len(item_embeddings)
		scores = np.array([self.booking_ui(item_embeddings[i], user, user_context)[0] for i in range(nitems)]).astype(float)
		scores -= np.min(scores)
		scores /= np.sum(scores)
		return scores
		
class OraclePolicy(Oracle):
	def __init__(self):
		super().__init__()
		self.name = "OraclePolicy"
		
	def predict(self, user, user_context, k, item_embeddings, item_categories):
		nitems = len(item_embeddings)
		scores = [None]*nitems
		for i in range(nitems):
			b, c = self.booking_ui(item_embeddings[i], user, user_context)
			d = self.delta_ui(item_embeddings[i], item_categories[i], user_context)
			scores[i] = int(c>0)+b*(1+d)
		return np.argsort(scores)[(-k):].tolist(), np.sort(scores)[(-k):].tolist()
		
	def allocation(self, user, user_context, item_embeddings, item_categories):
		nitems = len(item_embeddings)
		scores = [None]*nitems
		for i in range(nitems):
			b, c = self.booking_ui(item_embeddings[i], user, user_context)
			d = self.delta_ui(item_embeddings[i], item_categories[i], user_context)
			scores[i] = float(int(c>0)+b*(1+d))
		scores -= np.min(scores)
		scores /= np.sum(scores)
		return scores
