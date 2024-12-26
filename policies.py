#coding:utf-8

import numpy as np
import itertools
from dppy.finite_dpps import FiniteDPP

from tools import *

colors = {      "LinUCB": "gray", "LinUCBDiversity": "pink", 
		"CustomGreedy": "green", "CustomBruteForce": "firebrick", "CustomDPP": "steelblue", "CustomSampling": "rebeccapurple", 
		"EpsilonGreedy": "black"
	}

class Policy(object):
	def __init__(self, info, random_state):
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
		seed_everything(int(random_state))
		self.item_embeddings = info["item_embeddings"]
		self.nitems, self.nfeatures = self.item_embeddings.shape
		self.dim = self.nitems+self.nfeatures
		self.random_state = random_state
		self.name = "Policy"
		
	def fit(self, ratings):
		raise NotImplemented
		
	def predict(context, k, only_available=False):
		raise NotImplemented
		
	def update(self, context, rec_items, reward, div_intra, div_inter):
		raise NotImplemented
		
class LogisticPolicy(Policy):
	def __init__(self, info, random_state=1234, max_steps=5, lazy_update_fr=5):
		super().__init__(info, random_state)
		self.name = "Policy"
		self.max_steps = max_steps
		self.lazy_update_fr = lazy_update_fr
		self.clean()
		
	def clean(self):
		self.reg_lambda = self.dim
		self.theta = np.zeros((self.dim,)) # np.random.normal(0, 1, (self.dim,))
		self.hessian = self.reg_lambda * np.eye(self.dim)
		#self.design_matrix = self.reg_lambda * np.eye(self.dim)
		self.design_matrix_inv = (1 / self.reg_lambda) * np.eye(self.dim)
		self.p_visit = None
		self.arms = []
		self.rewards = []
		self.ctr = 0
		
	def f_mix(self, a, c):
		return np.concatenate((a, c), axis=1).astype(float)
		
	def sigmoid(self, x):
		return 1/(1+np.exp(-x))
		
	def fit(self, ratings):
		self.p_visit = 0
		for i in range(ratings.shape[0]):
			context = context_int2array(get_context_from_rating(ratings[i]), self.nitems).reshape(1,-1)
			action = self.item_embeddings.loc[self.item_embeddings.index[get_item_from_rating(ratings[i])],:].values.reshape(1,-1)
			reward = get_rating_from_rating(ratings[i])
			if (reward != 0):
				self.p_visit += 1
				self.update(context, action, np.array([reward]), None, None)
			## ignore if reward = 0
		self.p_visit /= ratings.shape[0]
		
	def update(self, context, rec_items, reward, div_intra, div_inter):
		## Maximum Likelihood Estimator from Faury et al., 2020 https://arxiv.org/pdf/2002.07530
		## As they do https://github.com/louisfaury/logistic_bandit/blob/master/logbexp/algorithms/logistic_ucb_1.py
		## we perform a few steps of a Newton's descent to solve the problem in Equation 8 (in LogisticUCB-1)
		for i in range(rec_items.shape[0]):
			arm = self.f_mix(rec_items[i,:].reshape(1,-1), context.reshape(1, -1))
			if (reward[i]==0):    ##
				return None   ##
			self.arms.append(arm.ravel())
			self.rewards.append((reward[i]+1)/2)
			#self.design_matrix += np.outer(arm, arm)
			arm = arm.reshape(-1,1)
			self.design_matrix_inv += -np.dot(self.design_matrix_inv, np.dot(np.outer(arm, arm), self.design_matrix_inv)) \
				/ (1 + np.dot(arm.T, np.dot(self.design_matrix_inv, arm)))
			self.reg_lambda = self.dim * np.log(2 + len(self.rewards))
			if (self.ctr%self.lazy_update_fr == 0) or (len(self.rewards)<200):
				coeffs = self.sigmoid(np.array(self.arms).dot(self.theta)[:, None])
				#try:
				y = coeffs - np.array(self.rewards)[:, None]
				#except ValueError:
				#	print((np.array(self.arms).shape, self.theta.shape, coeffs.shape, np.array(self.rewards).shape))
				#	raise ValueError
				grad = self.reg_lambda * self.theta + np.sum(y * np.array(self.arms), axis=0)
				self.hessian = np.dot(np.array(self.arms).T,
					coeffs * (1 - coeffs) * np.array(self.arms)) + self.reg_lambda * np.eye(self.dim)
				self.theta -= np.linalg.solve(self.hessian, grad)
			self.ctr += 1
		
	def predict(context, k, only_available=False):
		raise NotImplemented

########################
## Proposition        ##
########################

class Custom(LogisticPolicy):
	def __init__(self, info, random_state=1234, max_steps=5, lazy_update_fr=5, alpha=1.):
		super().__init__(info, random_state=random_state, max_steps=max_steps, lazy_update_fr=lazy_update_fr)
		self.name = "Custom"
		#None: takes the top k of all available scores
		#"BruteForce": iterate over all subsets (for a small number of items N and small K)
		#"greedy": greedily build the subset with the submodular monotone function
		#"DPP": sample with a DPP proportionally to the score
		#"sampling": sample M different K-subsets, return the best one
		self.alpha = alpha
		self.clean()
			
	def predict(self, context, k, only_available=False):
		available_items_ids = np.zeros(context.shape[0], dtype=bool) #get_available_actions(context)
		if (available_items_ids.sum()==0):
			#print(f"All items are explored for user {context_array2int(context, 1)}")
			available_items_ids = np.ones(available_items_ids.shape, dtype=bool)
			if (only_available):
				return None
		items = self.item_embeddings.values[available_items_ids,:]
		all_items = np.arange(available_items_ids.shape[0])
		contexts = np.tile(context.reshape(1,-1), (items.shape[0], 1))
		X_user = self.f_mix(items, contexts)
		qis = X_user.dot(self.theta)            ## only the individual qi's for available items
		qis /= np.max(qis)                      ## rescale values for numerical approximations
		qis[qis<=0] = np.min(qis[qis>0])/2      ## positive scores
		ids_samples = self.sample(qis, k)
		if (len(ids_samples)!=k):
			print((self.name, len(ids_samples), k, ids_samples))
		assert len(ids_samples)==k
		return all_items[available_items_ids][ids_samples]
		
	def sample(self, qis, k):
		return np.argsort(list(qis))[-k:]
		
	def compute_qdd(self, qis, S):
		quality = np.prod(qis[S])**(2*self.alpha)
		Phi = self.item_embeddings.values[S,:]
		diversity = np.linalg.det(Phi.dot(Phi.T))
		score = quality * diversity
		return score
		
	def get_L(self, qis, S=None):
		quality = np.power(qis if (S is None) else qis[S], self.alpha)
		Q = np.diag(quality)
		Phi = self.item_embeddings.values if (S is None) else self.item_embeddings.values[S,:] 
		return [Q, Phi]
		
class CustomBruteForce(Custom):
	def __init__(self, info, random_state=1234, max_steps=5, lazy_update_fr=5, alpha=1.):
		super().__init__(info, random_state=random_state, max_steps=max_steps, lazy_update_fr=lazy_update_fr, alpha=alpha)
		self.name = "CustomBruteForce"
		
	def sample(self, qis, k):
		'''
		Iterate over all k-sized subsets of items and 
		return the k-sized subset S that maximizes 
		Pi_{i in S} q[i]^alpha * det(Phi_{S}Phi_{S}^T)
		where q[i] is the positive quality score for i
		and Phi_{S} is the embedding matrix restricted to indices in S 
		'''
		assert len(scores)<101 and k<3
		all_combs = itertools.combinations(list(range(len(qis))), k)
		max_val, max_comb = -float("inf"), None
		for comb in all_combs:
			score = self.compute_qdd(qis, list(comb))
			if (score > max_val):
				max_comb = list(comb)
		return max_comb
		
class CustomGreedy(Custom):
	def __init__(self, info, random_state=1234, max_steps=5, lazy_update_fr=5, alpha=1.):
		super().__init__(info, random_state=random_state, max_steps=max_steps, lazy_update_fr=lazy_update_fr, alpha=alpha)
		self.name = "CustomGreedy"
		
	def sample(self, qis, k):
		'''
		Recursively adds items to the subset
		until it is of size k
		Added item a is argmax of the score
		Pi_{i in S+[a]} q[i]^alpha * det(Phi_{S+[a]}Phi_{S+[a]}^T)
		where q[i] is the positive quality score for i
		and Phi_{S} is the embedding matrix restricted to indices in S 
		'''
		max_comb = []
		while True:
			all_scores = [self.compute_qdd(qis, max_comb+[j]) if (j not in max_comb) else -float("inf") for j in range(len(qis))]
			if (np.max(all_scores)==-float("inf")):
				break
			candidates = np.argwhere(all_scores == np.max(all_scores)).ravel().tolist()
			if (len(max_comb)+len(candidates)>k):
				max_comb += np.random.choice(candidates, p=None, size=k-len(max_comb)).ravel().tolist()
				break
			else:
				max_comb += candidates
		return max_comb

class CustomSampling(Custom):
	def __init__(self, info, random_state=1234, max_steps=5, lazy_update_fr=5, alpha=1., M=10):
		super().__init__(info, random_state=random_state, max_steps=max_steps, lazy_update_fr=lazy_update_fr, alpha=alpha)
		self.name = "CustomSampling"
		self.M = M
		
	def sample(self, qis, k):
		'''
		Samples M subsets of size k at random
		and returns the subset S that maximizes
		Pi_{i in S+[a]} q[i]^alpha * det(Phi_{S+[a]}Phi_{S+[a]}^T)
		where q[i] is the positive quality score for i
		and Phi_{S} is the embedding matrix restricted to indices in S 
		'''
		test_combs = [np.random.choice(range(len(qis)), p=None, replace=False, size=k).ravel().tolist() for _ in range(self.M)]
		max_val, max_comb = -float("inf"), None
		for comb in test_combs:
			score = self.compute_qdd(qis, list(comb))
			if (score > max_val):
				max_comb = comb
		return max_comb

class CustomDPP(Custom):
	def __init__(self, info, random_state=1234, max_steps=5, lazy_update_fr=5, alpha=1.):
		super().__init__(info, random_state=random_state, max_steps=max_steps, lazy_update_fr=lazy_update_fr, alpha=alpha)
		self.name = "CustomDPP"
		self.DPP = None
		
	def sample(self, qis, k):
		'''
		Samples according to the L-ensemble
		where L = Q.Phi.Phi^T.Q
		
		Samples M subsets of size k at random
		and returns the subset S that maximizes
		Pi_{i in S+[a]} q[i]^alpha * det(Phi_{S+[a]}Phi_{S+[a]}^T)
		where Q is the diagonal matrix of positive quality scores
		and Phi is the item embedding matrix 
		'''
		Q, Phi = self.get_L(qis)
		K = Q.dot(Phi)
		L = K.dot(K.T)
		self.DPP = FiniteDPP('likelihood', **{'L': L})
		self.DPP.sample_exact_k_dpp(size=k)
		lst = self.DPP.list_of_samples[0]
		return lst
		
########################
## Baselines          ##
########################

## Returns random items with probability epsilon, otherwise chooses those with highest estimated expected reward
class EpsilonGreedy(LogisticPolicy):
	def __init__(self, info, random_state=1234, max_steps=5, lazy_update_fr=5, epsilon=0.1):
		super().__init__(info, random_state=random_state, max_steps=max_steps, lazy_update_fr=lazy_update_fr)
		self.name = "EpsilonGreedy"
		self.epsilon = epsilon
		self.clean()
			
	def predict(self, context, k, only_available=False):
		available_items_ids = np.zeros(context.shape[0], dtype=bool) #get_available_actions(context)
		if (available_items_ids.sum()==0):
			#print(f"All items are explored for user {context_array2int(context, 1)}")
			available_items_ids = np.ones(available_items_ids.shape, dtype=bool)
			if (only_available):
				return None
		items = self.item_embeddings.values[available_items_ids,:]
		all_items = np.arange(available_items_ids.shape[0])
		contexts = np.tile(context.reshape(1,-1), (items.shape[0], 1))
		X_user = self.f_mix(items, contexts)
		qis = X_user.dot(self.theta)     ## only the individual qi's for available items
		ids_samples = np.argsort(qis)[-k:]
		eps = np.random.choice([False,True], p=[1-self.epsilon, self.epsilon], size=k)
		ids_samples[eps] = np.random.choice([i for i in range(len(qis)) if (i not in ids_samples)], p=None, size=np.sum(eps))
		return all_items[available_items_ids][ids_samples]
		
from numpy.linalg import slogdet
from scipy.stats import chi2
		
## LogisticUCB-1 from Faury et al, 2020 https://arxiv.org/pdf/2002.07530
class LogisticUCB1(LogisticPolicy):
	def __init__(self, info, max_steps=5, lazy_update_fr=5, random_state=1234):
		super().__init__(info, max_steps=max_steps, lazy_update_fr=lazy_update_fr, random_state=random_state)
		self.name = "LogisticUCB1"
		self.failure_level = 0.05 ##
		self.kappa = 10 ##
		self.param_norm_ub = 1 ##
		self.clean()
		
	def weighted_norm(self, x, A):
		return np.sqrt(np.dot(x, np.dot(A, x)))
			
	def predict(self, context, k, only_available=False):
		## Returns k arms at a time /!\ select iteratively the arm to play
		available_items_ids = np.zeros(context.shape[0], dtype=bool) #get_available_actions(context)
		if (available_items_ids.sum()==0):
			#print(f"All items are explored for user {context_array2int(context, 1)}")
			available_items_ids = np.ones(available_items_ids.shape, dtype=bool)
			if (only_available):
				return None
		items = self.item_embeddings.values[available_items_ids,:]
		all_items = np.arange(available_items_ids.shape[0])
		contexts = np.tile(context.reshape(1,-1), (items.shape[0], 1))
		X_user = self.f_mix(items, contexts)
		arm_set = np.array(X_user)
		## https://github.com/louisfaury/logistic_bandit/blob/master/logbexp/algorithms/logistic_ucb_1.py
		# update bonus bonus
		ucb_bonus = self.update_ucb_bonus()
		# select arm
		all_scores = [self.compute_optimistic_reward(arm_set[i_arm,:], ucb_bonus) for i_arm in range(len(arm_set))]
		arm = np.reshape(arm_set[np.argmax(all_scores),:], (-1,))
		# update design matrix and inverse
		self.design_matrix += np.outer(arm, arm)
		self.design_matrix_inv += -np.dot(self.design_matrix_inv, np.dot(np.outer(arm, arm), self.design_matrix_inv)) \
			/ (1 + np.dot(arm, np.dot(self.design_matrix_inv, arm)))
		return arm

	## https://github.com/louisfaury/logistic_bandit/blob/master/logbexp/algorithms/logistic_ucb_1.py
	def update_ucb_bonus(self):
		"""
		Updates the ucb bonus function (slight refinment from the concentration result of Faury et al. 2020)
		"""
		_, logdet = slogdet(self.hessian)
		gamma_1 = np.sqrt(self.reg_lambda) / 2 + (2 / np.sqrt(self.reg_lambda)) \
			  * (np.log(1 / self.failure_level) + 0.5 * logdet - 0.5 * self.dim * np.log(self.reg_lambda) +
			     np.log(chi2(self.dim).cdf(2 * self.reg_lambda) / chi2(self.dim).cdf(self.reg_lambda)))
		gamma_2 = 1 + np.log(1 / self.failure_level) \
			  + np.log(chi2(self.dim).cdf(2 * self.reg_lambda) / chi2(self.dim).cdf(self.reg_lambda)) \
			  + 0.5 * logdet - 0.5 * self.dim * np.log(self.reg_lambda)
		gamma = np.min([gamma_1, gamma_2])
		res = 0.25 * np.sqrt(self.kappa) * np.min(
		    [np.sqrt(1 + 2 * self.param_norm_ub) * gamma, gamma + gamma ** 2 / np.sqrt(self.reg_lambda)])
		res += np.sqrt(self.reg_lambda) * self.param_norm_ub
		return res

	def compute_optimistic_reward(self, arm, ucb_bonus):
		"""
		Computes UCB for arm.
		"""
		norm = self.weighted_norm(arm, self.design_matrix_inv)
		pred_reward = self.sigmoid(np.sum(self.theta * arm))
		bonus = ucb_bonus * norm
		return pred_reward + bonus
	
	

import scipy.optimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import SGDRegressor

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
class LinUCB(Policy):
	def __init__(self, info, random_state=1234):
		self.name = "LinUCB"
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
			
	def predict(self, context, k, only_available=False):
		available_items_ids = np.zeros(context.shape[0], dtype=bool) #get_available_actions(context)
		if (available_items_ids.sum()==0):
			#print(f"All items are explored for user {context_array2int(context, 1)}")
			available_items_ids = np.ones(available_items_ids.shape, dtype=bool)
			if (only_available):
				return None
		items = self.item_embeddings.values[available_items_ids,:]
		contexts = np.tile(context.reshape(1,-1), (items.shape[0], 1))
		X_user = np.concatenate((items, contexts), axis=1).astype(int).astype(float)
		y_pred, y_std = self.gp.predict(X_user, return_std=True)
		scores = np.multiply((1+y_pred), y_std).flatten()
		all_items = np.arange(available_items_ids.shape[0])
		return all_items[available_items_ids][np.argsort(scores)[(-k):]]
		
	def update(self, context, rec_items, reward, div_intra, div_inter):
		pass # no update post fit
		
## Concatenate the item embedding and the user context
## use as feedback reward*diversity with respect to context
## and learn a single Gaussian Process
class LinUCBDiversity(LinUCB):
	def __init__(self, info, random_state=1234):
		super().__init__(info)
		self.name = "LinUCBDiversity"
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
			available_items_ids = np.zeros(context.shape, dtype=bool) #get_available_actions(context)
			if (available_items_ids.sum()==0):
				#print(f"All items are explored for user {context_array2int(context, 1)}")
				available_items_ids = np.ones(available_items_ids.shape, dtype=bool)
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
