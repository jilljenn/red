#coding:utf-8

import numpy as np
import itertools
from copy import deepcopy
from dppy.finite_dpps import FiniteDPP
from numpy.linalg import slogdet
from scipy.stats import chi2
from sklearn.linear_model import SGDClassifier

from tools import *

colors = {      "CustomGreedy": "green", "CustomBruteForce": "firebrick", "CustomDPP": "steelblue", "CustomSampling": "rebeccapurple", 
		"EpsilonGreedy": "black", "DiversityDPP": "orange", "LogisticUCB1": "cyan", 
		"LinOASM": "blue", "LogisticRegression": "magenta"
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
		
	def clean(self):
		pass
		
	def f_mix(self, a, c):
		return np.concatenate((a, c), axis=1).astype(float)
		
	def fit(self, ratings):
		raise NotImplemented
		
	def predict(context, k, only_available=True):
		raise NotImplemented
		
	def update(self, context, rec_items, reward, div_intra, div_inter):
		raise NotImplemented
		
## Super class implementing the common estimator for theta across policies
class LogisticPolicy_MLEFaury2020(Policy):
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
		
	def predict(context, k, only_available=True):
		raise NotImplemented

## Online Mirror Descent: Zhang, Y. J., & Sugiyama, M. (2024). Online (multinomial) logistic bandit: Improved regret and constant computation cost. Advances in Neural Information Processing Systems, 36.
## Lee, J., & Oh, M. H. (2024). Nearly minimax optimal regret for multinomial logistic bandit. arXiv preprint arXiv:2405.09831.
## Lower computational and storage cost
class LogisticPolicy(LogisticPolicy_MLEFaury2020):
	def __init__(self, info, random_state=1234, max_steps=5, lazy_update_fr=5, reg_eta=0.01):
		super().__init__(info, random_state, max_steps, lazy_update_fr)
		self.reg_eta = reg_eta
		self.name = "Policy"
		self.reg_lambda = self.dim
		self.theta = np.zeros((self.dim,1)) 
		self.design_matrix_inv = (1 / self.reg_lambda) * np.eye(self.dim)

	def sm(self, A, x, y=None):
		'''
		Apply the Sherman-Morrison formula to invert a matrix
		
		---
		Parameters
		A : array of shape (N, N)
			an inverted matrix
		x : array of shape (N, 1)
			a vector
		y : array of shape (N, 1) or None
			a vector (equal to x if None)
			
		---
		Returns
		H : array of shape (N,N)
			the inverse of matrix A+xy^T
		'''
		if (y is None):
			y = x
		matrix_norm = float(1+y.dot(A).dot(x.T))
		return A - A.dot(x.T.dot(y).dot(A))/matrix_norm
		
	def gradsigmoid(self, x):
		return self.sigmoid(x)*(1-self.sigmoid(x))
		
	def update(self, context, rec_items, reward, div_intra, div_inter):
		self.theta = self.theta.reshape(-1,1)
		inv_Hk = deepcopy(self.design_matrix_inv)
		for i in range(rec_items.shape[0]):
			arm = self.f_mix(rec_items[i,:].reshape(1,-1), context.reshape(1, -1))
			self.arms.append(arm.ravel())
			#print(("reward",reward[i]))
			self.rewards.append(reward[i])
			xtheta = float(arm.dot(self.theta.reshape(-1,1)))
			#XX = arm.T.dot(arm) 
			self.design_matrix_inv = self.sm(self.design_matrix_inv, self.sigmoid(xtheta)*arm, arm)
			#self.hessian += self.sigmoid(xtheta)*XX 
			## Online Mirror Descent
			## Orabona, F. (2019). A modern introduction to online learning. arXiv preprint arXiv:1912.13213.
			## Single projected gradient step
			#Htk = Hk + self.reg_eta * self.gradsigmoid(xtheta)*XX
			inv_Htk = self.sm(inv_Hk, self.reg_eta * self.gradsigmoid(xtheta) * arm, arm)
			#print((xtheta, reward[i], arm.shape))
			grad_l = (xtheta - reward[i])*arm.T
			#print((thetak.shape, inv_Htk.dot(grad_l).shape))
			thetakp = self.theta - self.reg_eta*inv_Htk.dot(grad_l)
			self.theta = thetakp.reshape(self.theta.shape)/float(np.linalg.norm(thetakp,2))
			#print((arm.shape, thetak.shape, "1"))
			xtheta = float(arm.dot(self.theta.reshape(-1,1)))
			inv_Hk = self.sm(inv_Hk, self.gradsigmoid(xtheta)*arm, arm)

########################
## Proposition        ##
########################

## Super class implementing the underlying reward model
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
			
	def predict(self, context, k, only_available=True):
		available_items_ids = get_available_actions(context) if (only_available) else np.zeros(context.shape[0], dtype=bool)
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
		if (np.max(qis)!=0):
			qis /= np.max(qis)                      ## rescale values for numerical approximations
		qis[qis<=0] = np.min(qis[qis>0])/2 if (qis>0).any() else 0.1  ## positive scores
		ids_samples = self.sample(qis, k, np.argwhere(available_items_ids==1).ravel())
		assert len(ids_samples)==k
		return all_items[available_items_ids][ids_samples].flatten()
		
	def sample(self, qis, k, available_items_ids_lst):
		return np.argsort(list(qis[available_items_ids]))[-k:]
		
	def compute_qdd(self, qis, S):
		quality = np.prod(qis)**(2*self.alpha) #np.prod(qis[S])**(2*self.alpha)
		Phi = self.item_embeddings.values[S,:]
		diversity = np.linalg.det(Phi.dot(Phi.T))
		score = quality * diversity
		return score
		
	def get_L(self, qis, S=None):
		quality = np.power(qis, self.alpha) #np.power(qis if (S is None) else qis[S], self.alpha)
		Q = np.diag(quality.flatten())
		Phi = self.item_embeddings.values if (S is None) else self.item_embeddings.values[S,:] 
		return [Q, Phi]
	
## Find K maximizers using brute force	
class CustomBruteForce(Custom):
	def __init__(self, info, random_state=1234, max_steps=5, lazy_update_fr=5, alpha=1.):
		super().__init__(info, random_state=random_state, max_steps=max_steps, lazy_update_fr=lazy_update_fr, alpha=alpha)
		self.name = "CustomBruteForce"
		
	def sample(self, qis, k, available_items_ids_lst):
		'''
		Iterate over all k-sized subsets of items and 
		return the k-sized subset S that maximizes 
		Pi_{i in S} q[i]^alpha * det(Phi_{S}Phi_{S}^T)
		where q[i] is the positive quality score for i
		and Phi_{S} is the embedding matrix restricted to indices in S 
		'''
		assert len(qis)<101 and k<3
		all_combs = itertools.combinations(list(range(len(qis))), k)
		max_val, max_comb = -float("inf"), None
		for comb in all_combs:
			score = self.compute_qdd(qis, available_items_ids_lst[list(comb)])
			if (score > max_val):
				max_comb = list(comb)
		return max_comb
	
## Find K maximizers using greedy	
class CustomGreedy(Custom):
	def __init__(self, info, random_state=1234, max_steps=5, lazy_update_fr=5, alpha=1.):
		super().__init__(info, random_state=random_state, max_steps=max_steps, lazy_update_fr=lazy_update_fr, alpha=alpha)
		self.name = "CustomGreedy"
		
	def sample(self, qis, k, available_items_ids_lst):
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
			all_scores = [self.compute_qdd(qis, [available_items_ids_lst[x] for x in max_comb]+[available_items_ids_lst[j]]) if (j not in max_comb) else -float("inf") for j in range(len(qis))]
			if (np.max(all_scores)==-float("inf")):
				break
			candidates = np.argwhere(all_scores == np.max(all_scores)).ravel().tolist()
			if (len(max_comb)+len(candidates)>k):
				max_comb += np.random.choice(candidates, p=None, size=k-len(max_comb)).ravel().tolist()
				break
			else:
				max_comb += candidates
		return max_comb

## Find K maximizers sampling M sets at random
class CustomSampling(Custom):
	def __init__(self, info, random_state=1234, max_steps=5, lazy_update_fr=5, alpha=1., M=10):
		super().__init__(info, random_state=random_state, max_steps=max_steps, lazy_update_fr=lazy_update_fr, alpha=alpha)
		self.name = "CustomSampling"
		self.M = M
		
	def sample(self, qis, k, available_items_ids_lst):
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
			score = self.compute_qdd(qis, available_items_ids_lst[list(comb)].tolist())
			if (score > max_val):
				max_comb = comb
		return max_comb

## Find K maximizers using a DPP
class CustomDPP(Custom):
	def __init__(self, info, random_state=1234, max_steps=5, lazy_update_fr=5, alpha=1.):
		super().__init__(info, random_state=random_state, max_steps=max_steps, lazy_update_fr=lazy_update_fr, alpha=alpha)
		self.name = "CustomDPP"
		self.DPP = None
		
	def sample(self, qis, k, available_items_ids_lst):
		'''
		Samples according to the L-ensemble
		where L = Q.Phi.Phi^T.Q
		
		Samples M subsets of size k at random
		and returns the subset S that maximizes
		Pi_{i in S+[a]} q[i]^(2*alpha) * det(Phi_{S+[a]}Phi_{S+[a]}^T)
		where Q is the diagonal matrix of positive quality scores
		and Phi is the item embedding matrix 
		'''
		Q, Phi = self.get_L(qis, available_items_ids_lst)
		K = Q.dot(Phi)
		L = K.dot(K.T)
		self.DPP = FiniteDPP('likelihood', **{'L': L})
		self.DPP.sample_exact_k_dpp(size=k, random_state=self.random_state)
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
			
	def predict(self, context, k, only_available=True):
		available_items_ids = get_available_actions(context) if (only_available) else np.zeros(context.shape[0], dtype=bool)
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
		if (np.sum(eps)>0):
			ids_samples[eps] = np.random.choice([i for i in range(len(qis)) if (i not in ids_samples)], p=None, size=np.sum(eps))
		return all_items[available_items_ids][ids_samples].flatten()
		
## LogisticUCB-1 from Faury et al, 2020 https://arxiv.org/pdf/2002.07530
class LogisticUCB1(LogisticPolicy_MLEFaury2020):
	def __init__(self, info, max_steps=5, lazy_update_fr=5, random_state=1234):
		super().__init__(info, max_steps=max_steps, lazy_update_fr=lazy_update_fr, random_state=random_state)
		self.name = "LogisticUCB1"
		self.failure_level = 0.05 ##
		self.kappa = 10 ##
		self.param_norm_ub = 1 ##
		self.clean()
		
	def weighted_norm(self, x, A):
		return np.sqrt(np.dot(x, np.dot(A, x)))
			
	def predict(self, context, k, only_available=True):
		## Returns k arms at a time /!\ select iteratively the arm to play
		available_items_ids = get_available_actions(context) if (only_available) else np.zeros(context.shape[0], dtype=bool)
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
		ids_samples = np.argsort(all_scores)[-k:]
		# update design matrix and inverse
		#self.design_matrix += np.outer(arm, arm)
		for i in ids_samples:
			arm = np.reshape(arm_set[i,:], (-1,))
			self.design_matrix_inv += -np.dot(self.design_matrix_inv, np.dot(np.outer(arm, arm), self.design_matrix_inv)) \
				/ (1 + np.dot(arm, np.dot(self.design_matrix_inv, arm)))
		return all_items[available_items_ids][ids_samples].flatten()

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
		
class LinOASM(LogisticUCB1):
	def __init__(self, info, random_state=1234, max_steps=5, lazy_update_fr=5, lbd=1.):
		super().__init__(info, random_state=random_state, max_steps=max_steps, lazy_update_fr=lazy_update_fr)
		self.name = "LinOASM"
		self.lbd = lbd
		self.clean()
		
	def compute_qdd(self, qi, S):
		Phi = self.item_embeddings.values[S,:]
		diversity = np.linalg.det(Phi.dot(Phi.T))
		score = qi * diversity
		return score
		
	def sample(self, qis, k):
		'''
		Recursively adds items to the subset
		until it is of size k
		Added item a is argmax of the score
		q[a]^alpha * det(Phi_{S+[a]}Phi_{S+[a]}^T)
		where q[i] is the positive quality score for i
		and Phi_{S} is the embedding matrix restricted to indices in S 
		'''
		max_comb = []
		while True:
			all_scores = [self.compute_qdd(qis[j], max_comb+[j]) if (j not in max_comb) else -float("inf") for j in range(len(qis))]
			if (np.max(all_scores)==-float("inf")):
				break
			candidates = np.argwhere(all_scores == np.max(all_scores)).ravel().tolist()
			if (len(max_comb)+len(candidates)>k):
				max_comb += np.random.choice(candidates, p=None, size=k-len(max_comb)).ravel().tolist()
				break
			else:
				max_comb += candidates
		return max_comb
			
	def predict(self, context, k, only_available=True):
		available_items_ids = get_available_actions(context) if (only_available) else np.zeros(context.shape[0], dtype=bool)
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
		qis = [self.compute_optimistic_reward(arm_set[i_arm,:], ucb_bonus) for i_arm in range(len(arm_set))]
		ids_samples = self.sample(qis, k)
		return all_items[available_items_ids][ids_samples].flatten()
		
class LogisticRegression(Policy):
	def __init__(self, info, random_state=1234):
		super().__init__(info, random_state=random_state)
		self.name = "LogisticRegression"
		self.clean()
		
	def clean(self):
		self.model = SGDClassifier(loss='log_loss', fit_intercept=False)
		self.theta = None
		
	def fit(self, ratings):
		contexts = np.array([context_int2array(get_context_from_rating(ratings[i]), self.nitems).ravel() for i in range(ratings.shape[0])])
		actions = np.array([self.item_embeddings.loc[self.item_embeddings.index[get_item_from_rating(ratings[i])],:].values.ravel() for i in range(ratings.shape[0])])
		rewards = np.array([get_rating_from_rating(ratings[i]) for i in range(ratings.shape[0])])
		X_user = self.f_mix(actions, contexts)
		Y_user = rewards
		self.model.partial_fit(X_user, Y_user, list(sorted(np.unique(Y_user.astype(int)))))
		self.theta = self.model.coef_
		
	def predict(self, context, k, only_available=True):
		available_items_ids = get_available_actions(context) if (only_available) else np.zeros(context.shape[0], dtype=bool)
		if (available_items_ids.sum()==0):
			#print(f"All items are explored for user {context_array2int(context, 1)}")
			available_items_ids = np.ones(available_items_ids.shape, dtype=bool)
			if (only_available):
				return None
		items = self.item_embeddings.values[available_items_ids,:]
		all_items = np.arange(available_items_ids.shape[0])
		contexts = np.tile(context.reshape(1,-1), (items.shape[0], 1))
		X_user = self.f_mix(items, contexts)
		qis = self.model.predict_proba(X_user)#.ravel()
		qis = qis[:,1].ravel()
		# select arm
		ids_samples = self.sample(qis, k)
		return all_items[available_items_ids][ids_samples].flatten()
		
	def update(self, context, rec_items, reward, div_intra, div_inter):
		contexts = np.tile(context.reshape(1,-1), (rec_items.shape[0], 1))
		X_user = self.f_mix(rec_items, contexts)
		Y_user = reward
		self.model.partial_fit(X_user, Y_user)
		self.theta = self.model.coef_
		
	def compute_qdd(self, qi, S):
		Phi = self.item_embeddings.values[S,:]
		diversity = np.linalg.det(Phi.dot(Phi.T))
		score = qi * diversity
		return score
		
	def sample(self, qis, k):
		'''
		Recursively adds items to the subset
		until it is of size k
		Added item a is argmax of the score
		q[a]^alpha * det(Phi_{S+[a]}Phi_{S+[a]}^T)
		where q[i] is the positive quality score for i
		and Phi_{S} is the embedding matrix restricted to indices in S 
		'''
		max_comb = []
		while True:
			all_scores = [self.compute_qdd(qis[j], max_comb+[j]) if (j not in max_comb) else -float("inf") for j in range(len(qis))]
			if (np.max(all_scores)==-float("inf")):
				break
			candidates = np.argwhere(all_scores == np.max(all_scores)).ravel().tolist()
			if (len(max_comb)+len(candidates)>k):
				max_comb += np.random.choice(candidates, p=None, size=k-len(max_comb)).ravel().tolist()
				break
			else:
				max_comb += candidates
		return max_comb
		
class DiversityDPP(Custom):
	def __init__(self, info, random_state=1234, max_steps=5, lazy_update_fr=5, alpha=1.):
		super().__init__(info, random_state=random_state, max_steps=max_steps, lazy_update_fr=lazy_update_fr, alpha=alpha)
		self.name = "DiversityDPP"
		self.DPP = None
		
	def sample(self, qis, k):
		'''
		Samples according to the L-ensemble
		where L = Phi.Phi^T (only taking into account the features, not the predicted quality)
		
		Samples M subsets of size k at random
		and returns the subset S that maximizes
		det(Phi_{S+[a]}Phi_{S+[a]}^T)
		where Phi is the item embedding matrix 
		'''
		_, Phi = self.get_L(qis)
		L = Phi.dot(Phi.T)
		self.DPP = FiniteDPP('likelihood', **{'L': L})
		self.DPP.sample_exact_k_dpp(size=k, random_state=self.random_state)
		lst = self.DPP.list_of_samples[0]
		return np.array(lst).flatten()
