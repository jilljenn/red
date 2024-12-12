#coding:utf-8

import numpy as np
from sklearn.gaussian_process.kernels import DotProduct, RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
import warnings
from sklearn.exceptions import ConvergenceWarning

from tools import *

class Policy(object):
	def __init__(self, info):
		'''
		Obtains the optimal selection for the input context
		with respect to the diversity score
		
		---
		Parameters
		context : array of shape (N, 1)
			the feedback observed in the past
		K : int
			the number of actions to select
			
		---
		Returns
		pi : array of shape (K,)
			the optimal diversity-wise selection for context
		'''
		self.name = "Policy"
		
	def fit(self, ratings):
		raise NotImplemented
		
	def predict(context, k):
		raise NotImplemented
		
	def update(self, reward, div_intra, div_inter):
		raise NotImplemented
		
## TODO Epsilon-greedy (exploit-then-commit)
## TODO DPP: https://github.com/jilljenn/red/blob/main/notebooks/Movielens-Kaggle.ipynb
## https://github.com/jilljenn/red/blob/main/notebooks/Gaussian%20Processes%20for%20Collaborative%20Filtering.ipynb
## https://github.com/jilljenn/red/blob/main/notebooks/RED%20Tutorial.ipynb

########################
## Algorithms         ##
########################

from functools import partial
from sklearn.gaussian_process import GaussianProcessRegressor
import scipy.optimize

## https://stackoverflow.com/questions/62376164/how-to-change-max-iter-in-optimize-function-used-by-sklearn-gaussian-process-reg
class GPR(GaussianProcessRegressor):
    def __init__(self, kernel=None, _max_iter=None, n_restarts_optimizer=None, alpha=None, random_state=None, **kwargs):
        super().__init__(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer, alpha=alpha, random_state=random_state, **kwargs)
        self._max_iter = _max_iter

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        def new_optimizer(obj_func, initial_theta, bounds):
            res = scipy.optimize.minimize(
                obj_func,
                initial_theta,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options=dict(maxiter=self._max_iter),
            )
            theta_opt = res.x
            func_min = res.fun
            return theta_opt, func_min
        self.optimizer = new_optimizer
        return super()._constrained_optimization(obj_func, initial_theta, bounds)

## Concatenate the item embedding and the user context
## and learn a single Gaussian Process
class LogisticUCB(Policy):
	def __init__(self, info, random_state=1234):
		self.name = "LogisticUCB"
		self.random_state = random_state
		self.item_embeddings = info["item_embeddings"]
		#self.kernel = DotProduct(1.0, (1e-3, 1e3))
		#self.kernel = DotProduct(1e-3)#, (1e-3, 1e3))
		self.kernel = RBF(1.0, (1e-5, 1e5))
		self.gp = None
		self.nitems = self.item_embeddings.shape[0]
		
	def fit(self, ratings):
		self.gp = GPR(kernel=self.kernel, _max_iter=1000, n_restarts_optimizer=10, alpha=1e-2, random_state=self.random_state)
		user_contexts = np.array([context_int2array(get_context_from_rating(ratings[i]), self.nitems) for i in range(ratings.shape[0])])
		item_actions = self.item_embeddings.loc[self.item_embeddings.index[[get_item_from_rating(ratings[i]) for i in range(ratings.shape[0])]],:]
		X_user0 = np.concatenate((item_actions, user_contexts), axis=1).astype(int).astype(float)
		y_user0 = np.array([get_rating_from_rating(ratings[i]) for i in range(ratings.shape[0])]).astype(float).ravel()
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=ConvergenceWarning)
			self.gp.fit(X_user0, y_user0)
			
	def predict(self, context, k):
		available_items_ids = get_available_actions(context)
		if (available_items_ids.sum()==0):
			#print(f"All items are explored for user {context_array2int(context, 1)}")
			available_items_ids = np.ones(available_items_ids.shape, dtype=int)
		items = self.item_embeddings.values[available_items_ids,:]
		contexts = np.tile(context.reshape(1,-1), (items.shape[0], 1))
		X_user = np.concatenate((items, contexts), axis=1).astype(int).astype(float)
		y_pred, y_std = self.gp.predict(X_user, return_std=True)
		scores = np.multiply((1+y_pred), y_std).flatten()
		all_items = np.arange(available_items_ids.shape[0])
		return all_items[available_items_ids][np.argsort(scores)[(-k):]]
		
	def update(self, reward, div_intra, div_inter):
		pass # no update post fit
		
## Concatenate the item embedding and the user context
## use as feedback reward*diversity with respect to context
## and learn a single Gaussian Process
class LogisticUCBDiversity(LogisticUCB):
	def __init__(self, info, random_state=1234):
		super().__init__(info)
		self.name = "LogisticUCBDiversity"
		self.random_state = random_state
		
	def fit(self, ratings):
		self.gp = GPR(kernel=self.kernel, _max_iter=1000, n_restarts_optimizer=10, alpha=1e-2, random_state=self.random_state)
		user_contexts = np.array([context_int2array(get_context_from_rating(ratings[i]), self.nitems) for i in range(ratings.shape[0])])
		item_actions = self.item_embeddings.loc[self.item_embeddings.index[[get_item_from_rating(ratings[i]) for i in range(ratings.shape[0])]]].values
		X_user0 = np.concatenate((item_actions, user_contexts), axis=1).astype(int).astype(float)
		y_user0 = np.array([get_rating_from_rating(ratings[i]) for i in range(ratings.shape[0])]).astype(float).ravel()
		y_user0div = []
		for i in range(ratings.shape[0]):
			action = item_actions[i].reshape(1,-1)
			context = user_contexts[i]
			available_items_ids = get_available_actions(context)
			if (available_items_ids.sum()==0):
				#print(f"All items are explored for user {context_array2int(context, 1)}")
				available_items_ids = np.ones(available_items_ids.shape, dtype=int)
			context_embeddings = self.item_embeddings.values[~available_items_ids,:]
			if (context_embeddings.shape[0]==0):
				div = 1.0
			else:
				embs = np.concatenate((context_embeddings, action), axis=0)
				div = np.abs(np.linalg.det(self.kernel(embs)))
			y_user0div.append(div)
		y_user0 = np.multiply( y_user0, np.array(y_user0div) )
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=ConvergenceWarning)
			self.gp.fit(X_user0, y_user0)
