#coding:utf-8

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from subprocess import Popen
import os
import pandas as pd
from scipy.special import expit
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
import statistics 
import matplotlib.pyplot as plt
import pickle

from tools import *

##############
## NOTATION ##
##############
# N : #items
# K : #recommendations
# d : item embedding dimension
# dp : user embedding dimension
# r : reward model rank
# L : number of ratings
# U : number of users
# C : number of categories

class Reward(object):
	def __init__(self, item_embeddings, add_params=None):
		'''
		Creates the object that describes the reward
		
		---
		Parameters
		item_embeddings : array of shape (N, d)
			the item embeddings
		add_params : dict
			optional additional parameters
		'''
		self.item_embeddings = item_embeddings/np.linalg.norm(item_embeddings)
	
	def get_means(self, context, action_embeddings=None):
		'''
		Obtains the expected reward for N>=K>=1 actions 
		for the input context
		
		---
		Parameters
		context : array of shape (N, 1)
			the feedback observed in the past
		action_embeddings : array of shape (K, d) or None (default)
			the action embeddings. If None, computed for all items
			
		---
		Returns
		means : array of shape (K,)
			the expected rewards for the K actions in context
		'''
		raise NotImplemented
		
	def get_reward(self, context, action_embeddings):
		'''
		Obtains the noisy observation for N>=K>=1 actions 
		for the input context
		
		---
		Parameters
		context : array of shape (N, 1)
			the feedback observed in the past
		action_embeddings : array of shape (K, d)
			the action embeddings
			
		---
		Returns
		rewards : array of shape (K,)
			the noisy observations for the K actions in context
		'''
		raise NotImplemented
		
	def get_diversity(self, action_embeddings_positive, context=None, action_ids=None):
		'''
		Obtains the diversity score for N>=K>=1 actions 
		for the input context
		
		---
		Parameters
		action_embeddings_positive : array of shape (K, d)
			the action embeddings with positive feedback
		context : array of shape (N, 1) or None (default)
			the feedback observed in the past. If None, the 
			diversity score is computed intrabatch (otherwise, interbatch)
		action_ids : array of shape (K,) or None (default)
			the identifiers corresponding to the action embeddings
			
		---
		Returns
		diversity_scores : array of shape (K,)
			the diversity score for the K actions in context
		'''
		raise NotImplemented
		
	def get_oracle_diversity(self, context, K, aggreg_func=None, intra=False):
		'''
		Obtains the optimal selection for the input context
		with respect to the diversity score by greedily adding
		new items to the selection according to their interbatch 
		diversity score
		
		---
		Parameters
		context : array of shape (N, 1) or None
			the feedback observed in the past
		K : int
			the number of actions to select
		aggreg_func : Python function
			the aggregation function for diversity scores
		intra : bool
			computes the intrabatch diversity instead of 
			interbatch diversity if set to True
			
		---
		Returns
		pi : array of shape (K,)
			the optimal diversity-wise selection for context
		'''
		assert aggreg_func is not None
		available_items_ids = np.zeros(context.shape).ravel() #get_available_actions(context)
		if (available_items_ids.sum()==0):
			#print(f"All items are explored for user {context_array2int(context, self.m)}")
			available_items_ids = np.ones(available_items_ids.shape, dtype=int)
		available_items = self.item_embeddings[available_items_ids,:]
		pi = []
		for k in range(K): ## build the oracle greedily as the function is submodular
			vals = np.array([
					aggreg_func( self.get_diversity(available_items[pi+[i],:], context=None if (intra) else context, action_ids=np.array(pi+[i])) ) 
					for i in range(available_items.shape[0]) if (i not in pi)
				])
			vals = vals.flatten().tolist()
			if (len(vals)==0):
				break
			pi += np.argwhere(vals == np.max(vals)).flatten().tolist()
			if (len(pi)>=K):
				pi = np.array(pi)[:K]
				break 
		all_items = np.arange(available_items_ids.shape[0])
		pi = all_items[available_items_ids][np.array(pi)]
		return pi
		
	def get_oracle_reward(self, context, K):
		'''
		Obtains the optimal selection for the input context
		with respect to the reward 
		
		---
		Parameters
		context : array of shape (N, 1)
			the feedback observed in the past
		K : int
			the number of actions to select
			
		---
		Returns
		pi : array of shape (K,)
			the optimal reward-wise selection for context
		'''
		available_items_ids = np.zeros(context.shape).ravel() #get_available_actions(context)
		if (available_items_ids.sum()==0):
			#print(f"All items are explored for user {context_array2int(context, self.m)}")
			available_items_ids = np.ones(available_items_ids.shape, dtype=int)
		available_items = self.item_embeddings[available_items_ids,:]
		vals = self.get_means(context, available_items).flatten().tolist()
		all_items = np.arange(available_items_ids.shape[0])
		pi = all_items[available_items_ids][np.argsort(vals)[(-K):]]
		return pi
		
#####################
## Synthetic       ##
#####################
		
class SyntheticReward(Reward):
	def __init__(self, item_embeddings, add_params=dict(theta=None, item_categories=None, p_visit=0.9)):
		'''
		Logistic model of rewards with Gaussian noise
		
		---
		Parameters
		item_embeddings : array of shape (N, d)
			the item embeddings
		add_params : dict
			contains 
			theta : array of shape (r, 1), e.g., r=d+N if f_mix is the concatenation 
				the coefficients of the linear model
			item_categories : array of shape (N, C)
				the item categories (binary matrix)
			p_visit : float
				the probability of visiting an item
		'''
		assert "theta" in add_params and add_params["theta"] is not None
		assert "item_categories" in add_params and add_params["item_categories"] is not None
		assert "p_visit" in add_params and add_params["p_visit"] >= 0 and add_params["p_visit"] <= 1
		super().__init__(item_embeddings, add_params=add_params)
		self.m = 1
		self.name = "Synthetic"
		self.item_categories = add_params["item_categories"]
		self.theta = add_params["theta"]/np.linalg.norm(add_params["theta"])
		self.p_visit = add_params["p_visit"]
		self.kernel = linear_kernel
		self.LAMBDA = 0.01
		
	def f_mix(self, action, context):
		phi = np.concatenate((action, context), axis=0)
		assert phi.shape[0] == self.theta.shape[0]
		return phi
	
	def get_means(self, context, action_embeddings=None):
		if (action_embeddings is None):
			action_embeddings = self.item_embeddings
		K = action_embeddings.shape[0]
		N = self.item_embeddings.shape[0]
		means = np.zeros(K)
		#available_items_ids = get_available_actions(context)
		#context_categories = self.item_categories[~available_items_ids,:].sum(axis=0)
		for k in range(K):
			xk = self.f_mix(action_embeddings[k].reshape(-1,1), context.reshape(-1,1))
			means[k] = self.theta.T.dot(xk)[0,0]
			#means[k] = context_categories.dot(self.Phi.dot(action_embeddings[k].reshape(-1,1)))
		return means
		
	def get_reward(self, context, action_embeddings):
		K = action_embeddings.shape[0]
		means = self.get_means(context.reshape(1,-1), action_embeddings)
		ps = expit(means)
		reward = np.zeros((K,))
		p_visits = np.random.choice([0,1], p=[1-self.p_visit, self.p_visit], size=(K,))
		for k in range(K):
			p_book = np.random.choice([0,1], p=[1-ps[k], ps[k]])
			reward[k] = p_visits[k] * (-1)**p_book
			assert reward[k] in [-1, 0, 1]
		return reward.astype(int)
		
	def get_diversity(self, action_embeddings_positive=None, context=None, action_ids=None):
		'''
		Obtains the diversity across items by computing
		ridge leverage scores from [1, Def 1] with a linear kernel
		on the item categories
		
		[1] Musco, Cameron, and Christopher Musco. "Recursive sampling for the nystrom method." 
		Advances in neural information processing systems 30 (2017).
		''' 
		if ((action_embeddings_positive is None) or (action_ids is None)):
			action_embeddings_positive = self.item_embeddings
			actions_ids = np.arange(self.item_embeddings.shape[0])
		## from https://github.com/jilljenn/red/blob/main/notebooks/RED%20Tutorial.ipynb
		def compute_leverage_scores(embs):
			kernel = self.kernel(embs)
			return ( np.diagonal(kernel @ np.linalg.inv(kernel + self.LAMBDA * np.identity(len(kernel)))) ).ravel()
		if (len(action_embeddings_positive)==0 or len(action_ids)==0):
			return 0
		action_categories = self.item_categories[action_ids,:]
		scores = compute_leverage_scores(action_categories)
		if (context is None or context.sum()==0):
			return scores
		available_items_ids = np.zeros(context.shape).ravel().astype(bool) #get_available_actions(context)
		context_categories = self.item_categories[~available_items_ids,:]
		#context_embeddings = self.item_embeddings[~available_items_ids,:]
		embs = np.concatenate((action_categories, context_categories), axis=0)
		scores = compute_leverage_scores(embs)
		return scores
	
def synthetic(nusers, nitems, nratings, ncategories, emb_dim=512, emb_dim_user=10, m=1, loc=0, scale=1, p_visit=0.9):
	'''
	Parameters
	----------
	nusers : int
		the number U of users
	nitems : int
		the number N of items
	nratings : int
		the number L of ratings user-item to generate
	ncategories : int
		the number C of (non necessarily distinct) 
		item categories to identify
	emb_dim : int
		the number d of dimensions for item embeddings
	emb_dim_user : int
		the number dp of dimensions for user embeddings
	m : int
		the maximum feedback value in absolute value
	loc : float
		the mean of the data generating Gaussian distribution
	scale : float
		the standard deviation of the data generating Gaussian distribution
	p_visit : float
		the probability of visiting a recommended item
		
	Returns
	-------
	ratings : array of shape (L, 5)
		each row comprises the user identifier, the item identifier, 
		the number of times the user has seen the item,
		the item categories in binary, the user context in 2*m binary integers, 
		the (integer) reward
	info : dict
		contains the following entries
		item_embeddings : array of shape (N, d)
			the item embeddings
		user_embeddings : array of shape (U, dp)
			the user embeddings
		item_categories : array of (N, C)
			the category annotations for each item
		Phi : array of (C, d)
			the centroids for each category of items
	reward : class Reward
		the ground-truth reward for the problem
	'''
	## Generate item embeddings
	item_embeddings = np.random.normal(loc, scale, size=(nitems, emb_dim))
	item_embeddings /= np.linalg.norm(item_embeddings)
	## Generate user embeddings
	user_embeddings = np.random.normal(loc, scale, size=(nusers, emb_dim_user))
	user_embeddings /= np.linalg.norm(user_embeddings)
	## Define item categories
	item_cluster = KMeans(n_clusters=ncategories)
	item_cluster.fit(item_embeddings)
	item_categories = np.zeros((nitems, ncategories))
	for ncat in range(ncategories):
		item_categories[:,ncat] = item_cluster.labels_==ncat
	Phi = item_cluster.cluster_centers_
	Phi /= np.linalg.norm(Phi)
	## Generate model coefficients
	theta = np.random.normal(loc, scale, size=(nitems+emb_dim, 1))
	theta /= np.linalg.norm(theta)
	#M = np.eye(nitems)
	#means = item_embeddings.dot(theta[:item_embeddings.shape[1],:]) + M.dot(theta[item_embeddings.shape[1]:,:])
	#print(means)
	#print(expit(means))
	## Define user contexts
	## However, in the next recommendation, the user sees only once each item
	npulls = np.zeros((nusers, nitems))
	contexts = np.zeros((nusers, nitems))
	## Generate ratings
	ratings = [None]*nratings
	all_pairs = np.array([(u,i) for u in range(nusers) for i in range(nitems)])
	user_item_pairs = all_pairs[np.random.choice(len(all_pairs), size=nratings)]
	reward = SyntheticReward(item_embeddings, add_params=dict(theta=theta, item_categories=item_categories, p_visit=p_visit))
	for nrat, [u, i] in tqdm(enumerate(user_item_pairs.tolist())):
		## Get reward
		rat = 0
		while (rat == 0): ## avoid rewards = 0 as all items here are supposed to be visited here
			rat = int(reward.get_reward(contexts[u], item_embeddings[i].reshape(1,-1)))
		bin_context = context_array2int(contexts[u].flatten(), m)
		bin_cats = "".join(list(map(lambda x : str(int(x)), item_categories[i])))
		ratings[nrat] = [u, i, int(npulls[u,i]), bin_cats, bin_context, rat]
		## Update user context
		contexts[u, i] = (npulls[u,i]*contexts[u, i]+rat)/(npulls[u,i]+1)
		npulls[u,i] += 1
	ratings = np.array(ratings, dtype=object)
	item_embeddings = pd.DataFrame(item_embeddings, index=range(nitems), columns=range(emb_dim))
	user_embeddings = pd.DataFrame(user_embeddings, index=range(nusers), columns=range(emb_dim_user))
	item_categories = pd.DataFrame(item_categories.astype(int), index=range(nitems), columns=range(ncategories))
	Phi = pd.DataFrame(Phi, index=range(ncategories), columns=range(emb_dim))
	return ratings, {"item_embeddings": item_embeddings, "user_embeddings": user_embeddings, "item_categories": item_categories, "Phi": Phi}, reward
	
#####################
## MovieLens       ##
#####################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import Accuracy, R2Score
from sklearn.metrics.pairwise import cosine_similarity

class MovieLensReward(SyntheticReward):
	def __init__(self, item_embeddings, add_params=dict(theta=None, item_categories=None, p_visit=0.9)):
		self.item_categories = add_params["item_categories"]
		super().__init__(item_embeddings, add_params=add_params)

class MLP(nn.Module):
	def __init__(self, item_embeddings, mlp_depth=2, mlp_width=128, last_layer_width=8, dtype=torch.float, Sp=1., S=1.):
		"""
		Parameters
		----------
		mlp_depth : int
		    Number of hidden layers in the MLP.
		mlp_width : int
		    Width of the MLP. If None take mlp_width=n_features. Default: None.
		dtype : _dtype
		    Pytorch dtype for the parameters. Default: torch.float.

		"""
		super().__init__()
		self.item_embeddings = torch.Tensor(item_embeddings.values)
		self.N = self.item_embeddings.shape[0]
		self.nf = self.item_embeddings.shape[1]
		self.layers = nn.Sequential(
		    *[nn.Linear(self.nf, 32, dtype=dtype), nn.ReLU()],
		    *[nn.Linear(32, 64, dtype=dtype), nn.ReLU()],
		    *[nn.Linear(64, mlp_width, dtype=dtype), nn.ReLU()],
		    *[nn.Linear(mlp_width, mlp_width, dtype=dtype), nn.ReLU()]*mlp_depth,
		    *[nn.Linear(mlp_width, 64, dtype=dtype), nn.ReLU()],
		    *[nn.Linear(64, 32, dtype=dtype), nn.ReLU()],
		    *[nn.Linear(32, 16, dtype=dtype), nn.ReLU()],
		    *[nn.Linear(16, last_layer_width, dtype=dtype)]
		)
		self.Sp = Sp
		self.S = S
		self.Theta = torch.nn.parameter.Parameter(data=torch.Tensor(np.random.normal(0,1,size=(self.N+last_layer_width, 1))), requires_grad=True)
		with torch.no_grad():
			self.Theta /= torch.norm(self.Theta)/Sp

	def forward(self, inp):
		## batch_size x N
		contexts = inp[:,self.nf:]
		## batch_size x F
		x = inp[:,:self.nf]
		## batch_size x d
		action_embeddings = self.layers(x)
		with torch.no_grad():
			action_embeddings /= action_embeddings.norm()/self.S
			## d x 1
			self.Theta /= self.Theta.norm()/self.Sp
		## batch_size x (N+d)
		all_actions = torch.cat((contexts, action_embeddings), axis=1)
		## batch_size x (N+d)
		Thetas = self.Theta.repeat(1,x.shape[0]).T
		out = (all_actions * Thetas[None]).sum(dim=-1).squeeze()
		return out
		
def get_optimizer_by_group(model, optim_hyperparams):
    wd = optim_hyperparams.pop('weight_decay')
    group_wd = []
    group_no_wd = []
    for name, param in model.named_parameters():
        if name == 'layers.0.mu':
            group_no_wd.append(param)
        else:
            group_wd.append(param)
    optimizer = optim.Adam(
            [{'params': group_wd, 'weight_decay': wd},
                {'params': group_no_wd, 'weight_decay': 0}],
            **optim_hyperparams
        )
    return optimizer

def learn_from_ratings(ratings_, item_embeddings, emb_dim, nepochs=100, batch_size=1000, test_size=0.2, Sp=1., S=1., lr=0.01, seed=1234):
	'''
	See Appendix F.4 of Papini, Tirinzoni, Restelli, Lazaric and Pirotta (ICML'2021). 
	Linearization: We train a neural network to regress from initial item embeddings to ratings by some of the users
	Neural network: 2 hidden layers of size 256, ReLU activations, linear output layer of size 8
	Feature extraction : last layer of the neural network
	'''
	seed_everything(seed)
	network = MLP(item_embeddings, last_layer_width=emb_dim)
	#ratings_train, ratings_test = ratings_[train_id], ratings_[[i for i in range(ratings_.shape[0]) if (i not in train_id)]]
	#N, nf = item_embeddings.shape
	#X, y = np.zeros((ratings_.shape[0], N+nf)), np.zeros(ratings_.shape[0])
	#for i in tqdm(range(ratings_.shape[0])):
	#	context = context_int2array(get_context_from_rating(ratings_[i]), N).reshape(1,-1)
	#	item = get_item_from_rating(ratings_[i])
	#	x_i = item_embeddings.loc[item_embeddings.index[item]].values.reshape(1,-1)
	#	cx_i = np.concatenate((context, x_i), axis=1)
	#	X[i, :] = cx_i
	#	y_i = get_rating_from_rating(ratings_[i])
	#	y[i] = (y_i+1)/2
	## Split ratings into 80% (training) and 20% (testing)
	train_id = np.random.choice(range(ratings_.shape[0]), size=int((1-test_size)*ratings_.shape[0]))
	test_id = np.array([i for i in range(ratings_.shape[0]) if (i not in train_id)])
	#train_idx = np.zeros(ratings_.shape[0], dtype=bool)
	#test_idx = np.ones(ratings_.shape[0], dtype=bool)
	#train_idx[train_id] = 1
	#test_idx[train_id] = 0
	#X_train, y_train = torch.Tensor(X[train_idx,:]), torch.Tensor(y[train_idx])
	#X_test, y_test = torch.Tensor(X[~train_idx,:]), torch.Tensor(y[~train_idx])
	#ds_train = TensorDataset(X_train, y_train)
	#ds_test = TensorDataset(X_test, y_test)
	ds_train = TensorDataset(torch.Tensor(train_id.reshape(-1,1)), torch.Tensor(train_id))
	#ds_test = TensorDataset(test_id, test_id)
	train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
	def retrieve_instance(ids_list):
		N, nf = item_embeddings.shape
		X, y = np.zeros((len(ids_list), N+nf)), np.zeros(len(ids_list))
		for ii, i in tqdm(enumerate(ids_list), leave=False):
			context = context_int2array(get_context_from_rating(ratings_[i]), N).reshape(1,-1)
			item = get_item_from_rating(ratings_[i])
			x_i = item_embeddings.loc[item_embeddings.index[item]].values.reshape(1,-1)
			cx_i = np.concatenate((context, x_i), axis=1)
			y_i = get_rating_from_rating(ratings_[i])
			X[ii,:] = cx_i
			y[ii] = (y_i+1)/2
		return torch.Tensor(X), torch.Tensor(y)
	X_test, y_test = retrieve_instance(test_id)
	#test_loader = DataLoader(ds_test, batch_size=batch_size)
	## Training
	criterion = nn.MSELoss(reduction="sum") 
	opt = get_optimizer_by_group(network, {'weight_decay': 0.0, 'lr': lr})
	old_loss, max_patience = float("inf"), 3
	patience = max_patience
	all_losses, all_testLoss, all_testR2 = [], [], []
	for epoch in (pbar := tqdm(range(nepochs))):
		train_loss_crit, n_tot = 0, 0
		for bx, by in train_loader:
			## Create data
			X, y = retrieve_instance(by.numpy().ravel().astype(int))
			opt.zero_grad(set_to_none=True)
			loss = criterion(network(X), y) 
			train_loss_crit += loss.item()
			n_tot += by.size(0)
			loss.backward(retain_graph=True)
			with torch.no_grad():
				network.Theta /= network.Theta.norm()/Sp
			opt.step()
		train_loss_crit = train_loss_crit/float(n_tot)
		all_losses.append(train_loss_crit)
		test_loss = criterion(network(X_test), y_test)/X_test.shape[0]
		all_testLoss.append(test_loss.item())
		test_loss /= statistics.pvariance(y_test.numpy()) 
		test_loss = 1 - test_loss
		all_testR2.append(test_loss.item())
		if ((train_loss_crit>old_loss) and (max_patience is not None)):
			patience -= 1
		elif (max_patience is not None):
			patience = max_patience
		pbar.set_description(f"Epoch {epoch+1}/{nepochs} - Train Loss {np.round(all_losses[-1],3)} - Test Loss={np.round(all_testLoss[-1],3)} - Test R2={np.round(all_testR2[-1],3)}")
		if (patience == 0):
			break
		old_loss = train_loss_crit
	## Plot
	plt.plot(range(len(all_losses)), all_losses, "b-", label="train MSE")
	plt.plot(range(len(all_testLoss)), all_testLoss, "g-", label="test MSE")
	#plt.plot(range(len(all_testR2)), all_testR2, "r-", label="test R2")
	#plt.ylim(-1, 1)
	plt.legend()
	plt.show()
	## Return coefficients + new (linear) item embeddings
	Theta = network.Theta.detach().numpy()
	Theta /= np.linalg.norm(Theta)/Sp
	new_item_embeddings = network.layers(torch.Tensor(item_embeddings.values)).detach().numpy()
	new_item_embeddings /= np.linalg.norm(new_item_embeddings)/S
	return Theta, pd.DataFrame(new_item_embeddings, index=item_embeddings.index, columns=range(new_item_embeddings.shape[1]))
		
def movielens(nratings=None, ncategories=None, emb_dim=None,  emb_dim_user=None, p_visit=0.9, savename="movielens_instance.pck"):
	'''
	Parameters
	----------
	nratings : int
		the number of ratings user-item to generate
	ncategories : int
		the number of (non necessarily distinct) item categories to identify
	emb_dim : int
		the number of dimensions for item embeddings
	emb_dim_user : int
		the number of dimensions for user embeddings
	p_visit : float
		the probability of visiting a recommended item
	savename : str
		the file name to which the instance will be stored
		
	Returns
	-------
	ratings : array of shape (L, 5)
		each row comprises the user identifier, the item identifier, 
		the number of times the user has seen the item,
		the item categories in binary, the user context in 2*m binary integers, 
		the (integer) reward
	info : dict
		contains the following entries
		item_embeddings : array of shape (N, d)
			the item embeddings
		user_embeddings : array of shape (U, dp)
			the user embeddings
		item_categories : array of (N, C)
			the category annotations for each item
		Phi : array of (C, d)
			the centroids for each category of items
	reward : class Reward
		the ground-truth reward for the problem
	'''
	nusers=None
	nitems=None
	## Create the MovieLens data set
	if (not os.path.exists("ml-latest-small/")):
		proc = Popen("wget -qO - https://files.grouplens.org/datasets/movielens/ml-latest-small.zip |  bsdtar -xvf -".split(" "))
		proc.wait()
	## 1. Movie feature matrix and item categories
	items = pd.read_csv("ml-latest-small/movies.csv", sep=",", index_col=0)
	all_categories = items["genres"].unique()
	all_genres = list(set([y for x in items["genres"] for y in x.split("|")]))
	item_categories = np.array([[np.argwhere(items.loc[i]["genres"]==all_categories).flatten() for i in items.index]]).T
	if ((ncategories is not None) and abs(ncategories-len(np.unique(item_categories)))<abs(ncategories-len(all_genres))):
		ncategories = len(np.unique(item_categories))
	else:
		ncategories = len(all_genres)
		item_categories = np.array([[ int(g in items.loc[i]["genres"].split("|")) for g in all_genres] for i in items.index])
	### First example: Year + genre
	if (True):
		items["Year"] = [x.split(")")[len(x.split(")"))-2**int(len(x.split(")"))>1)].split("(")[-1].split("–")[-1] if (len(x.split("("))>1) else "0" for x in items["title"]]
		for genre in all_genres:
			items[genre] = [int(genre in x) for x in items["genres"]]
		items = items[["Year"]+all_genres]
		#emb_dim = items.shape[1]
	### Second example (sparsier): bag-of-words
	else:
		from sklearn.feature_extraction.text import TfidfVectorizer
		items = pd.read_csv("ml-latest-small/movies.csv", sep=",", index_col=0)
		corpus = [items.loc[idx]["title"]+" "+" ".join(items.loc[idx]["genres"].split("|")) for idx in items.index]
		min_len = 4
		try:
			for i in range(min_len,20):
				vectorizer = TfidfVectorizer(analyzer="word",stop_words="english",token_pattern=r"(?u)\b"+r"\w"*i+r"+\b")
				items_mat = vectorizer.fit_transform(corpus).toarray().T
				sparsity = items_mat.shape[0]-emb_dim
				if (sparsity < 0):
					min_len = i-1
					break
		except:
			pass
		vectorizer = TfidfVectorizer(analyzer="word",stop_words="english",token_pattern=r"(?u)\b"+r"\w"*min_len+r"+\b")
		items_mat = vectorizer.fit_transform(corpus).toarray().T
		#select = np.argsort((items_mat!=0).mean(axis=1))[-emb_dim:]
		#items_mat = items_mat[select,:]
		#index = vectorizer.get_feature_names_out()[select]
		index = vectorizer.get_feature_names_out()
		items = pd.DataFrame(items_mat, columns=items.index, index=index).T
	#emb_dim = items.shape[1]
	items = items.astype(float)
	items.index = items.index.astype(str)
	items.columns = items.columns.astype(str)
	item_categories = pd.DataFrame(item_categories, index=items.index, columns=range(item_categories.shape[1]))
	## 2. Phi: embeddings of categories
	Phi = np.zeros((ncategories, items.shape[1]))
	for ncat in range(ncategories):
		cent = items.values[item_categories.values[:,ncat]==1,:].mean(axis=0)
		Phi[ncat,:] = cent
	Phi = pd.DataFrame(Phi, index=range(item_categories.shape[1]), columns=range(Phi.shape[1]))
	## 3. User feature matrix 
	users = pd.read_csv("ml-latest-small/tags.csv", sep=",")
	users["count"] = 1
	users = pd.pivot_table(users, columns=["userId"], values=["count"], index=["tag"], aggfunc="sum", fill_value=0)
	#users.reset_index(level=[0,0])
	users = users.astype(float)
	users.index = users.index.astype(str)
	users.columns = users.columns.get_level_values(1).astype(str)
	users = users.T
	emb_dim_user = users.shape[1]
	## 4. Ratings 
	ratings = pd.read_csv("ml-latest-small/ratings.csv", sep=",")
	ratings = pd.pivot_table(ratings, columns=["userId"], values=["rating"], index=["movieId"], aggfunc="mean", fill_value=0)
	ratings = ratings.astype(float)
	ratings.index = ratings.index.astype(str)
	ratings.columns = ratings.columns.get_level_values(1).astype(str)
	col_idx, row_idx = [x for x in list(ratings.columns) if (x in users.index)], [x for x in list(ratings.index) if (x in items.index)]
	users = users.loc[col_idx]
	items = items.loc[row_idx]
	item_categories = item_categories.loc[row_idx]
	ratings = ratings.loc[row_idx][col_idx]
	## Filter
	if (nratings is not None):
		assert nratings is None or nratings<=ratings.shape[0]
		ratings_list = np.array(np.argwhere(ratings.values!=0).tolist()[:nratings])
		idx0, idx1 = np.unique(ratings_list[:,0]), np.unique(ratings_list[:,1])
		ratings_ = ratings.values[idx0, :][:, idx1]
		ratings = pd.DataFrame(ratings_, index=ratings.index[idx0], columns=ratings.columns[idx1])
		#ratings = ratings.iloc[ratings_list[:,0].flatten().tolist()]
		#ratings = ratings[ratings.columns[ratings_list[:,1].flatten().tolist()]]
		items = items.loc[ratings.index]
		users = users.loc[ratings.columns]
		item_categories = item_categories.loc[ratings.index]
		idx = item_categories.sum(axis=0)>0
		item_categories = item_categories[item_categories.columns[idx]]
		Phi = Phi.loc[idx]
	if (nitems is not None) and (nitems<=items.shape[0]):
		item_idx = ratings.sum(axis=1).sort_values(ascending=False).index[:nitems]
		ratings = ratings.loc[item_idx]
		idx = ratings.abs().sum(axis=1)>0
		ratings = ratings.loc[idx]
		users = users.loc[ratings.columns]
		item_categories = item_categories.loc[ratings.index]
		idx = item_categories.sum(axis=0)>0
		item_categories = item_categories[item_categories.columns[idx]]
		Phi = Phi.loc[idx] 
	if (nusers is not None) and (nusers<=users.shape[0]):
		user_idx = ratings.sum(axis=0).sort_values(ascending=False).index[:nusers]
		ratings = ratings[user_idx]
		idx = ratings.sum(axis=0)>0
		ratings = ratings[ratings.columns[idx]]
		items = items.loc[ratings.columns]
		item_categories = item_categories.loc[ratings.index]
		idx = item_categories.sum(axis=0)>0
		item_categories = item_categories[item_categories.columns[idx]]
		Phi = Phi.loc[idx]
	if (nitems is None):
		nitems = items.shape[0] 
	if (nratings is None):
		nratings = int((ratings!=0).sum().sum())
	if (nusers is None):
		nusers = users.shape[0] 
	ncategories = item_categories.shape[1]
	if (False):
		print("Items")
		print(items.shape == (nitems, items.shape[1]))
		print("Users")
		print(users.shape == (nusers, emb_dim_user))
		print("Categories")
		print(item_categories.shape == (nitems, ncategories))
		print("Phi")
		print(Phi.shape == (ncategories, emb_dim))
		print("_________")
	## Define user contexts
	npulls = np.zeros((nusers, nitems))
	contexts = np.zeros((nusers, nitems))
	## Generate ratings 
	ratings_ = [None] * nratings
	ratings_list = np.array(np.argwhere(ratings.values>0).tolist()[:nratings]).tolist()
	threshold = int(np.max(ratings.values)/2)+1
	ratings[(ratings!=0)&(ratings<threshold)] = -1
	ratings[(ratings!=0)&(ratings>=threshold)] = 1
	for nrat, [i, u] in tqdm(enumerate(ratings_list)):
		## Get reward
		rat = ratings.values[i,u]#-threshold
		bin_context = context_array2int(contexts[u].flatten(), threshold)
		item_cat = "".join(list(map(str,item_categories.loc[items.index[i]].values.flatten().tolist())))
		ratings_[nrat] = [u, i, int(npulls[u,i]), item_cat, bin_context, float(rat)]
		## Update user context
		contexts[u, i] = (contexts[u, i]*npulls[u,i]+rat)/(npulls[u,i]+1)
		npulls[u,i] += 1
	ratings_ = ratings_[:nrat]
	ratings_ = np.array(ratings_, dtype=object)
	## 5. Reward 
	theta, item_embeddings = learn_from_ratings(ratings_, items, min(emb_dim,items.shape[1]))
	emb_dim = item_embeddings.shape[1]
	reward = SyntheticReward(item_embeddings, add_params=dict(theta=theta, item_categories=item_categories, p_visit=p_visit))
	info = {"item_embeddings": item_embeddings, "user_embeddings": users, "item_categories": item_categories, "Phi": Phi}
	with open(savename, "wb") as f:
		pickle.dump(dict(ratings=ratings_,info=info,theta=theta,p_visit=p_visit), f)
	return ratings_, info, reward
	
if __name__=="__main__":
	nusers=nitems=10
	nratings=80
	ncategories=2
	emb_dim=512
	emb_dim_user=11
	p_visit=0.9
	print("_"*27)
	if (False):
		print("SYNTHETIC")
		ratings_, info, reward = synthetic(nusers, nitems, nratings, ncategories, emb_dim=emb_dim, emb_dim_user=emb_dim_user, p_visit=p_visit)
		print("Ratings")
		print(ratings_.shape)
		print(ratings_[:5,:])
		item_embeddings, user_embeddings, item_categories, Phi = [info[s] for s in ["item_embeddings", "user_embeddings", "item_categories", "Phi"]]
		print("Items")
		print(item_embeddings.shape == (nitems, emb_dim))
		print("Users")
		print(user_embeddings.shape == (nusers, emb_dim_user))
		print("Categories")
		print(item_categories.shape == (nitems, ncategories))
		print("Phi")
		print(Phi.shape == (ncategories, emb_dim))
		print(get_context_from_rating(ratings_[-1]))
		context = context_int2array(get_context_from_rating(ratings_[-1]), nitems)
		action_emb = item_embeddings.loc[item_embeddings.index[0]].values.reshape(1,-1)
		print(reward.get_reward(context, action_emb))
		print(reward.get_means(context, action_emb))
		print("_"*27)
	if (True): 
		print("MOVIELENS")
		if (not os.path.exists("movielens_instance.pck")):
			ratings_, info, reward = movielens(nratings=None, ncategories=None, emb_dim=8, p_visit=p_visit, savename="movielens_instance.pck")
		else:
			with open("movielens_instance.pck", "rb") as f:
				di = pickle.load(f)
			ratings_, info, theta = [di[n] for n in ["ratings", "info", "theta"]]
			reward = SyntheticReward(info["item_embeddings"], add_params=dict(theta=theta, item_categories=info["item_categories"], p_visit=p_visit))
		print("Ratings")
		print(ratings_.shape)
		print(ratings_[:5,:])
		item_embeddings, user_embeddings, item_categories, Phi = [info[s] for s in ["item_embeddings", "user_embeddings", "item_categories", "Phi"]]
		print(get_context_from_rating(ratings_[-1]))
		nitems = info["item_embeddings"].shape[0]
		context = context_int2array(get_context_from_rating(ratings_[-1]), nitems).reshape(-1,1)
		action_emb = item_embeddings.loc[item_embeddings.index[0]].values.reshape(1,-1)
		print(reward.get_reward(context, action_emb))
		print(reward.get_means(context, action_emb))
		print("_"*27)
