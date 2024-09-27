#coding:utf-8

import numpy as np
from dppy.finite_dpps import FiniteDPP
from sklearn.preprocessing import OneHotEncoder

from tools import context_int2array

########################
## Algorithms         ##
########################

## Gaussian Processes
from sklearn.gaussian_process.kernels import DotProduct, RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor

class Heuristic(object):
	def __init__(self):
		pass
		
	def fit(self, ratings, item_embeddings, item_categories, Phi):
		raise NotImplemented
		
	def predict(self, user, user_context, k, item_embeddings, item_categories):
		raise NotImplemented
		
	def allocation(self, user, user_context, item_embeddings):
		raise NotImplemented
		
	def update(self, user, item, reward):
		raise NotImplemented

class GaussianProcess(Heuristic):
	def __init__(self):
		self.gps = None
		self.name = "GaussianProcess"
		
	def fit(self, ratings, item_embeddings, item_categories=None, Phi=None):
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
		scores -= np.min(scores)-1
		scores /= np.sum(scores)
		return scores
		
	def update(self, user, item, reward):
		pass # no update
		
## TODO Epsilon greedy

########################
## Oracles            ##
########################
		
class Oracle(Heuristic):
	def __init__(self):
		self.generator = None
		self.name = "Oracle"
		
	def fit(self, reward, item_embeddings, item_categories=None, Phi=None):
		self.reward = reward
		self.L = item_embeddings.dot(item_embeddings.T)
		self.DPP = FiniteDPP('likelihood', **{'L': self.L})
		self.enc = OneHotEncoder()
		self.enc.fit(item_categories.reshape(-1, 1))
		self.ncats = len(np.unique(item_categories))
		self.item_categories = item_categories
		
	def get_items(self, user_context):
		context = (context_int2array(user_context, self.ncats)!=0).astype(int).reshape(-1,1)
		all_cats = self.enc.transform(self.item_categories.reshape(-1,1)).T.toarray()
		return np.argwhere(context.T.dot(all_cats).ravel()==1).ravel().tolist()

	def get_reward(self, i, item_emb, u, context, item_cat):
		c = context_int2array(context, self.ncats)
		return self.reward.get_reward(c, item_emb, i)
		
	def get_diversity(self, batch1, batch2=None):
		if (len(batch1)<2):
			return 1.
		div_batch1 = np.linalg.det(self.L[batch1,:][:,batch1])
		if (batch2 is None):
			return div_batch1
		b = list(set(batch1+batch2))
		return np.linalg.det(self.L[b,:][:,b])-div_batch1
		
	def update(self, user, item, reward):
		pass

class TrueRewardPolicy(Oracle):
	def __init__(self):
		super().__init__()
		self.name = "TrueRewardPolicy"
		
	def predict(self, user, user_context, k, item_embeddings, item_categories=None):
		c = context_int2array(user_context, self.ncats)
		scores = np.array([self.reward.get_oracle(c, item_embeddings[i], i) for i in range(len(item_embeddings))])
		return np.argsort(scores)[(-k):].tolist(), np.sort(scores)[(-k):].tolist()
		
	def allocation(self, user, user_context, item_embeddings):
		c = context_int2array(user_context, self.ncats)
		scores = np.array([self.reward.get_oracle(c, item_embeddings[i], i) for i in range(len(item_embeddings))])
		scores -= np.min(scores)-1
		scores /= np.sum(scores)
		return scores
		
class OraclePolicy(Oracle):
	def __init__(self):
		super().__init__()
		self.name = "OraclePolicy"
		
	def predict(self, user, user_context, k, item_embeddings, item_categories):
		ncats = len(np.unique(item_categories))
		items = self.get_items(user_context)
		c = context_int2array(user_context, self.ncats)
		means = np.array([self.reward.get_oracle(c, item_embeddings[i], i) for i in range(len(item_embeddings))])
		rec = []
		scs = []
		for _ in range(k):
			divs = np.array([self.get_diversity(items+rec, [i]) for i in range(len(item_embeddings))])
			scores = np.multiply(means, divs)
			rec.append(np.argsort(scores)[-1])
			scs.append(np.sort(scores)[-1])
		return rec, scs
		
	def allocation(self, user, user_context, item_embeddings, item_categories=None):
		ncats = len(np.unique(item_categories))
		c = context_int2array(user_context, self.ncats)
		items = self.get_items(user_context)
		scores = np.array([self.reward.get_oracle(c, item_embeddings[i], i)*self.get_diversity(items, [i]) for i in range(len(item_embeddings))])
		scores -= np.min(scores)-1
		scores /= np.sum(scores)
		return scores
