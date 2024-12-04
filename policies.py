#coding:utf-8

import numpy as np
from sklearn.gaussian_process.kernels import DotProduct, RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
#from dppy.finite_dpps import FiniteDPP
#from sklearn.preprocessing import OneHotEncoder

from tools import context_int2array

class Policy(object):
	def __init__(self, info):
		self.name = "Policy"
		
	def fit(self, ratings):
		raise NotImplemented
		
	def predict(context, k, available_items=None):
		raise NotImplemented
		
	def allocation(self, context):
		raise NotImplemented
		
	def update(self, reward, div_intra, div_inter):
		raise NotImplemented

########################
## Algorithms         ##
########################

class GaussianProcess(Policy):
	def __init__(self, info):
		self.gps = None
		self.name = "GaussianProcess"
		self.item_embeddings = info["item_embeddings"]
		#self.kernel = DotProduct(1.0, (1e-3, 1e3))
		self.kernel = RBF(1.0, (1e-3, 1e3))
		self.gp = None
		self.nitems = self.item_embeddings.shape[0]
		
	def fit(self, ratings):
		self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10, alpha=1e-2)
		user_contexts = np.array([context_int2array(uc, self.nitems) for uc in ratings[:,3].flatten().tolist()])
		item_actions = self.item_embeddings.loc[self.item_embeddings.index[ratings[:,1].flatten().tolist()],:]
		X_user0 = np.concatenate((item_actions, user_contexts), axis=1).astype(int).astype(float)
		y_user0 = ratings[:, -1].astype(float).ravel()
		self.gp.fit(X_user0, y_user0)
			
	def predict(self, context, k, available_items=None):
		if (available_items is None):
			items = self.item_embeddings
		else:
			items = available_items
		contexts = np.tile(context.reshape(1,-1), (items.shape[0], 1))
		X_user = np.concatenate((items, contexts), axis=1).astype(int).astype(float)
		y_pred, y_std = self.gp.predict(X_user, return_std=True)
		#for i, (item, pred, std) in enumerate(zip(range(items.shape[0]), y_pred, y_std)):
		#	print(i, item, pred.round(5), std.round(5), ((1+pred) * std).round(5))
		scores = np.multiply((1+y_pred), y_std)
		return np.argsort(scores)[(-k):].tolist()
		
	def allocation(self, context):
		pass
		#y_pred, y_std = self.gps[user].predict(item_embeddings, return_std=True)
		#scores = np.multiply((1+y_pred), y_std)
		#scores -= np.min(scores)-1
		#scores /= np.sum(scores)
		#return scores
		
	def update(self, reward, div_intra, div_inter):
		pass # no update
