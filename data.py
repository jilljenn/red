#coding:utf-8

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from subprocess import Popen
import os
import pandas as pd
from sklearn.gaussian_process.kernels import DotProduct, RBF, ConstantKernel as C

from tools import *

class Reward(object):
	def __init__(self, Theta, item_embeddings, add_params=None, sigma=0.1, m=1.):
		'''
		Creates the object that describes the synthetic
		environment with Gaussian rewards and ground truth
		
		---
		Parameters
		Theta : array of shape (1, d)
			Low-rank factor of the underlying kernel
		item_embeddings : array of shape (nitems, d)
			item embeddings
		sigma : float
			Variance of the noisy Gaussian observations
		m : int
			Maximum (in absolute value) observed feedback
		'''
		self.m = m
		self.kernel = None
		self.item_embeddings = item_embeddings
	
	def get_means(self, context, action_embeddings):
		'''
		Obtains the expected reward for N>=K>=1 actions based on 
		the current context over the total number of N items
		
		---
		Parameters
		context : array of shape (N, 1)
			The feedback observed for some actions in the past
		action_embeddings : array of shape (K, d)
			Action embeddings
			
		---
		Returns
		means : array of shape (K, 1)
			The expected rewards for all played actions for context
		'''
		raise NotImplemented
		
	def get_reward(self, context, action_embeddings):
		'''
		Obtains the noisy observation for N>=K>=1 actions based on 
		the current context over the total number of N items
		
		---
		Parameters
		context : array of shape (N, 1)
			The feedback observed for some actions in the past
		action_embeddings : array of shape (K, d)
			Action embeddings
			
		---
		Returns
		rewards : array of shape (K, 1)
			The noisy observations for each of the K actions for context
		'''
		raise NotImplemented
		
	def get_diversity(self, action_embeddings_positive, context=None, action_ids=None):
		'''
		Obtains the diversity across items 
		
		---
		Parameters
		action_embeddings_positive : array of shape (K, d)
			Action embeddings (only those who are *visited and booked* = reward > 0)
		context : array of shape (1, N)
			User context (optional)
		action_ids : array of shape (K, 1)
			Action identifiers
			
		---
		Returns
		diversity_scores : array of shape (K, 1)
			The diversity score for each of the K actions
		'''
		raise NotImplemented
		
	def get_oracle_diversity(self, context, K, available_items=None, action_ids=None):
		'''
		Obtains the optimal allocation based on 
		the current context over the total number of N items
		
		---
		Parameters
		context : array of shape (N, 1)
			The feedback observed for some actions in the past
		K : int
			Number of actions to select
		available_items : array of shape (K, d)
			Action embeddings
		action_ids : array of shape (K, 1)
			Action identifiers
			
		---
		Returns
		pi : array of shape (K, 1)
			The optimal allocation for context
		'''
		raise NotImplemented
		
	def get_oracle_reward(self, context, K, action_embeddings=None):
		'''
		Obtains the optimal allocation based on 
		the current context over the total number of N items
		
		---
		Parameters
		context : array of shape (N, 1)
			The feedback observed for some actions in the past
		K : int
			Number of actions to select
			
		---
		Returns
		pi : array of shape (K, 1)
			The optimal allocation for context
		'''
		raise NotImplemented
		
#####################
## Synthetic       ##
#####################
		
class SyntheticReward(Reward):
	def __init__(self, Theta, item_embeddings, add_params=None, sigma=0.1, m=1.):
		'''
		Creates the object that describes the synthetic
		environment with Gaussian rewards and ground truth
		
		---
		Parameters
		Theta : array of shape (1, d)
			Low-rank factor of the underlying kernel
		item_embeddings : array of shape (nitems, d)
			item embeddings
		sigma : float
			Variance of the noisy Gaussian observations
		m : int
			Maximum (in absolute value) observed feedback
		'''
		assert sigma>0
		assert m>0
		self.name = "Synthetic"
		self.Theta = Theta
		self.sigma = sigma
		self.item_embeddings = item_embeddings
		self.S = np.max(np.linalg.norm(item_embeddings, axis=1))
		self.Sp = np.max(np.linalg.norm(Theta, axis=1))
		self.renorm = (self.S**2)*self.Sp
		self.m = m
		self.kernel = DotProduct(1.0, (1e-3, 1e3))
		#self.kernel = RBF(1.0, (1e-3, 1e3))
	
	def get_means(self, context, action_embeddings):
		'''
		Obtains the expected reward for N>=K>=1 actions based on 
		the current context over the total number of N items
		
		---
		Parameters
		context : array of shape (N, 1)
			The feedback observed for some actions in the past
		action_embeddings : array of shape (K, d)
			Action embeddings
			
		---
		Returns
		means : array of shape (K, 1)
			The expected rewards for all played actions for context
		'''
		means = np.zeros((action_embeddings.shape[0], 1))
		for k in range(action_embeddings.shape[0]):
			Xk = np.tile(action_embeddings[k].T, (self.item_embeddings.shape[0], 1))
			dst = 1/self.renorm * np.power(self.item_embeddings-Xk, 2)
			means[k] = float(context.reshape(1,-1).dot(dst).dot(self.Theta.T)[0,0])
		return means
		
	def get_reward(self, context, action_embeddings):
		'''
		Obtains the noisy observation for N>=K>=1 actions based on 
		the current context over the total number of N items
		
		---
		Parameters
		context : array of shape (N, 1)
			The feedback observed for some actions in the past
		action_embeddings : array of shape (K, d)
			Action embeddings
			
		---
		Returns
		rewards : array of shape (K, 1)
			The noisy observations for each of the K actions for context
		'''
		means = self.get_means(context, action_embeddings)
		reward = np.zeros(action_embeddings.shape[0])
		#while (reward == 0).any(): ## avoid this, as all recommended items are assumed to be seen
		reward = np.clip(np.random.normal(means, self.sigma), -self.m, self.m).flatten()
		reward[reward!=0] = (-1)**(reward[reward!=0]<0).astype(int)
		return reward.astype(int)
		
	def get_diversity(self, action_embeddings_positive, context=None, action_ids=None):
		'''
		Obtains the diversity across items 
		
		---
		Parameters
		action_embeddings_positive : array of shape (K, d)
			Action embeddings (only those who are *visited and booked* = reward > 0)
		context : array of shape (1, N)
			User context (optional)
		action_ids : array of shape (K, 1)
			Action identifiers
			
		---
		Returns
		diversity_scores : array of shape (K, 1)
			The diversity score for each of the K actions and those in context (if provided)
		''' 
		if (len(action_embeddings_positive)==0):
			return 0
		div1 = np.linalg.det(self.kernel(action_embeddings_positive))
		if (context is None or context.sum()==0):
			return np.abs(div1)
		available_items_ids = get_available_actions(context)
		context_embeddings = self.item_embeddings[~available_items_ids,:]
		embs = np.concatenate((context_embeddings, action_embeddings_positive), axis=0)
		return np.abs(np.linalg.det(self.kernel(embs)))
		
	def get_oracle_diversity(self, context, k, available_items=None, action_ids=None):
		'''
		Obtains the optimal allocation based on diversity,
		the current context over the total number of N items
		
		---
		Parameters
		context : array of shape (N, 1)
			The feedback observed for some actions in the past
		k : int
			Number of actions to select
		available_items : array of shape (X, 1)
			Item embeddings available for recommendation
		action_ids : array of shape (K, 1)
			Action identifiers
			
		---
		Returns
		pi : array of shape (K, 1)
			The optimal allocation for context
		'''
		item_set = self.item_embeddings if (available_items is None) else available_items
		means = [self.get_diversity(item_set[i].reshape(1,-1), context=context, action_ids=[i]) for i in range(item_set.shape[0])]
		return np.argsort(means)[-k:]
		
	def get_oracle_reward(self, context, k, available_items=None):
		'''
		Obtains the optimal allocation based on rewards,
		the current context over the total number of N items
		
		---
		Parameters
		context : array of shape (N, 1)
			The feedback observed for some actions in the past
		k : int
			Number of actions to select
		available_items : array of shape (X, 1)
			Item embeddings available for recommendation
			
		---
		Returns
		pi : array of shape (K, 1)
			The optimal allocation for context
		'''
		means = self.get_means(context, self.item_embeddings if (available_items is None) else available_items).flatten().tolist()
		return np.argsort(means)[-k:]
	
def synthetic(nusers, nitems, nratings, ncategories, emb_dim=512, emb_dim_user=10, S=1., Sp=1., m=3, sigma=1., loc=0, scale=1):
	'''
	Parameters
	----------
	nusers : int
		number of users
	nitems : int
		number of items
	nratings : int
		number of ratings user-item to generate
	ncategories : int
		number of (non necessarily distinct) item categories to identify
	emb_dim : int
		number of dimensions for item embeddings
	emb_dim_user : int
		number of dimensions for user embeddings
	S : float
		maximum norm of item embeddings
	Sp : float
		maximum norm of Theta parameter
	m : int
		maximum feedback value in absolute value
	sigma : float
		variance of the subgaussian reward noise
		
	Returns
	-------
	ratings : array of shape (nratings, 5)
		each row comprises the user identifier, the item identifier, 
		the number of times the user has seen the item,
		the item categories in binary, the user context in 2m binary integers, 
		the (integer) reward
	item_embeddings : array of shape (nitems, emb_dim)
		item embeddings
	user_embeddings : array of shape (nusers, emb_dim)
		user embeddings
	item_categories : array of (nitems, ncategories)
		category annotations for each item
	Phi : array of (ncategories, emb_dim)
		centroids for each category of items
	reward : class Reward
		encodes the "true" reward for the problem
	'''
	## Generate item embeddings
	item_embeddings = np.random.normal(loc, scale, size=(nitems, emb_dim))
	item_embeddings /= np.linalg.norm(item_embeddings)/S
	## Generate user embeddings
	user_embeddings = np.random.normal(loc, scale, size=(nusers, emb_dim_user))
	user_embeddings /= np.linalg.norm(user_embeddings)/S
	## Define item categories
	item_cluster = KMeans(n_clusters=ncategories)
	item_cluster.fit(item_embeddings)
	item_categories = np.zeros((nitems, ncategories))
	for ncat in range(ncategories):
		item_categories[:,ncat] = item_cluster.labels_==ncat
	Phi = item_cluster.cluster_centers_
	## Define model parameter
	Theta = np.random.normal(loc, scale, size=(1, emb_dim))
	Theta /= np.linalg.norm(Theta)/Sp
	## Define user contexts
	## However, in the next recommendation, the user sees only once each item
	npulls = np.zeros((nusers, nitems))
	contexts = np.zeros((nusers, nitems))
	## Generate ratings
	ratings = [None]*nratings
	all_pairs = np.array([(u,i) for u in range(nusers) for i in range(nitems)])
	user_item_pairs = all_pairs[np.random.choice(len(all_pairs), size=nratings)]
	reward = SyntheticReward(Theta, item_embeddings, sigma=sigma, m=m)
	for nrat, [u, i] in tqdm(enumerate(user_item_pairs.tolist())):
		## Get reward
		rat = 0
		while (rat == 0): ## avoid rewards = 0 as all items here are supposed to be visited
			rat = int(reward.get_reward(contexts[u], item_embeddings[i].reshape(1,-1)))
		bin_context = context_array2int(contexts[u].flatten(), m)
		ratings[nrat] = [u, i, npulls[u,i], "".join(list(map(lambda x : str(int(x)), item_categories[i]))), bin_context, rat]
		## Update user context
		contexts[u, i] = (npulls[u,i]*contexts[u, i]+rat)/(npulls[u,i]+1)
		npulls[u,i] += 1
	ratings = np.array(ratings, dtype=object)
	item_embeddings = pd.DataFrame(item_embeddings, index=range(nitems), columns=range(emb_dim))
	user_embeddings = pd.DataFrame(user_embeddings, index=range(nusers), columns=range(emb_dim_user))
	item_categories = pd.DataFrame(item_categories.astype(int), index=range(nitems), columns=range(ncategories))
	Phi = pd.DataFrame(Phi, index=range(ncategories), columns=range(emb_dim))
	return ratings, {"item_embeddings": item_embeddings, "user_embeddings": user_embeddings, "item_categories": item_categories, "category_embeddings": Phi}, reward
	
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
	def __init__(self, Theta, item_embeddings, add_params=None, sigma=1., m=1.):
		self.item_categories = add_params["item_categories"]
		super(self).__init__(Theta, item_embeddings, sigma=sigma, m=1.)
		
	def get_reward(self, context, action_embeddings):
		## Reward if visited and booked: +1
		##        if non visited: 0
		##        if visited and not booked: -1
		means = self.get_means(context, action_embeddings)
		probas = 1/(1+np.exp(-means))
		reward = [np.random.choice(range(2), p=[1-p, p])*(-1)**np.random.choice(range(2), p=[p, 1-p]) for p in probas.flatten().tolist()]
		return reward
		
	def get_diversity(self, action_embeddings_positive=None, context=None, action_ids=None):
		## Diversity if visited, booked, new (category): +1
		##           otherwise: 0
		action_positive_categories = self.item_categories[action_ids, :]
		div1 = 1-cosine_similarity(action_positive_categories)
		np.fill_diag(div1, 0)
		print(div1)
		div1 = np.mean((np.sum(div1, axis=1)==0).astype(int))
		if (context is None or context.sum()==0):
			return np.abs(div1)
		context_categories = self.item_categories[context, :]
		embs = np.concatenate((context_categories, action_positive_categories), axis=0)
		div2 = 1-cosine_similarity(embs)
		np.fill_diag(div2, 0)
		print(div2)
		Nc = context_categories.shape[0]
		div2 = np.mean((np.sum(div2[Nc:,:Nc], axis=1)==0).astype(int))
		return div2

class MLP(nn.Module):
	def __init__(self, item_embeddings, n_features, mlp_depth=1, mlp_width = 256, last_layer_width = 8, dtype=torch.float, Sp=1., S=1.):
		"""
		Parameters
		----------
		n_features : int
		    Dimension of inputs.
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
		    *[nn.Linear(n_features, mlp_width, dtype=dtype), nn.ReLU()],
		    *[nn.Linear(mlp_width, mlp_width, dtype=dtype), nn.ReLU()]*mlp_depth,
		    *[nn.Linear(mlp_width, last_layer_width, dtype=dtype)]
		)
		self.Sp = Sp
		self.S = S
		self.Theta = torch.nn.parameter.Parameter(data=torch.Tensor(np.random.normal(0,1,size=(last_layer_width, 1))), requires_grad=True)
		with torch.no_grad():
			self.Theta /= torch.norm(self.Theta)/Sp

	def forward(self, inp):
		contexts = inp[:,self.nf:]
		x = inp[:,:self.nf]
		all_actions = self.layers(self.item_embeddings)
		action_embeddings = self.layers(x)
		out = torch.zeros((x.shape[0], 1))
		with torch.no_grad():
			self.Theta /= self.Theta.norm()/self.Sp
		for i in range(x.shape[0]):
			Xk = action_embeddings[i].repeat(self.item_embeddings.size(dim=0), 1)
			dst = 1/(self.S**2*self.Sp) * torch.pow(all_actions-Xk, 2)
			x1 = torch.matmul( contexts[i].reshape((1, -1)) , dst )
			x2 = self.Theta.reshape(-1,1)
			out[i] = torch.matmul( x1, x2 )
		return out.squeeze()
		
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

def learn_from_ratings(ratings_, item_embeddings, emb_dim, nepochs=30, batch_size=200, test_size=0.2, Sp=1., S=1., seed=1234):
	'''
	See Appendix F.4 of Papini, Tirinzoni, Restelli, Lazaric and Pirotta (ICML'2021). 
	Linearization: We train a neural network to regress from initial item embeddings to ratings by some of the users
	Neural network: 2 hidden layers of size 256, ReLU activations, linear output layer of size 8
	Feature extraction: for each item, we consider 
	'''
	seed_everything(seed)
	network = MLP(item_embeddings, n_features=item_embeddings.shape[1], last_layer_width=emb_dim)
	## Split ratings into 80% (training) and 20% (testing)
	train_id = np.random.choice(range(ratings_.shape[0]), size=int((1-test_size)*ratings_.shape[0]))
	train_idx = np.zeros(ratings_.shape[0], dtype=int)
	train_idx[train_id] = 1
	#ratings_train, ratings_test = ratings_[train_id], ratings_[[i for i in range(ratings_.shape[0]) if (i not in train_id)]]
	N, nf = item_embeddings.shape
	X, y = np.zeros((ratings_.shape[0], N+nf)), np.zeros(ratings_.shape[0])
	for i in tqdm(range(ratings_.shape[0])):
		context = context_int2array(get_context_from_rating(ratings_[i]), N).reshape(1,-1)
		item = get_item_from_rating(ratings_[i])
		x = item_embeddings.loc[item_embeddings.index[item]].values.reshape(1,-1)
		reward = get_rating_from_rating(ratings_[i])
		xx = np.concatenate((context, x), axis=1)
		X[i, :] = xx
		y[i] = reward
	X_train, y_train = torch.Tensor(X[train_idx,:]), torch.Tensor(y[train_idx])
	X_test, y_test = torch.Tensor(X[~train_idx,:]), torch.Tensor(y[~train_idx])
	ds_train = TensorDataset(X_train, y_train)
	ds_test = TensorDataset(X_test, y_test)
	train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(ds_test, batch_size=batch_size)
	## Training
	criterion = nn.MSELoss()
	opt = get_optimizer_by_group(network, {'weight_decay': 0.0, 'lr': 0.1})
	for epoch in (pbar := tqdm(range(nepochs))):
		for bx, by in train_loader:
			opt.zero_grad(set_to_none=True)
			preds = network(bx)
			loss = criterion(preds, by.reshape(preds.size()))
			loss.backward(retain_graph=True)
			opt.step()
			with torch.no_grad():
				network.Theta /= torch.norm(network.Theta)/Sp
		y_preds_test = network(X_train)
		y_preds = network(X_test)
		test_loss = R2Score()
		test_loss.update(y_preds_test, y_test)
		test_loss = test_loss.compute()
		train_loss = R2Score()
		train_loss.update(y_preds, y_train)
		train_loss = train_loss.compute()
		pbar.set_description(f"Epoch {epoch+1}/{nepochs} - Train MSE {loss.item()} - Training loss R2={train_loss} - Testing loss R2={test_loss}")
	print(("Final loss", train_loss, test_loss)) 
	## Return coefficients + new (linear) embeddings
	Theta = network.Theta.detach().numpy()
	new_item_embeddings = network.layers(torch.Tensor(item_embeddings.values)).detach().numpy()
	return Theta, pd.DataFrame(new_item_embeddings, index=item_embeddings.index, columns=range(new_item_embeddings.shape[1]))
		
def movielens(nusers=None, nitems=None, nratings=None, ncategories=None, emb_dim=None,  emb_dim_user=None, S=1., Sp=1., sigma=1.):
	'''
	Parameters
	----------
	nusers : int
		number of users
	nitems : int
		number of items
	nratings : int
		number of ratings user-item to generate
	ncategories : int
		number of (non necessarily distinct) item categories to identify
	emb_dim : int
		number of dimensions for item embeddings
	emb_dim_user : int
		number of dimensions for user embeddings
	S : float
		maximum norm of item embeddings
	Sp : float
		maximum norm of Theta parameter
	m : int
		maximum feedback value in absolute value
	sigma : float
		variance of the subgaussian reward noise
		
	Returns
	-------
	ratings : array of shape (nratings, 5)
		each row comprises the user identifier, the item identifier, 
		the item categories in binary, the user context in 2m binary integers, 
		the (integer) reward
	item_embeddings : DataFrame of shape (nitems, emb_dim)
		item embeddings
	user_embeddings : DataFrame of shape (nusers, emb_dim)
		user embeddings
	item_categories : DataFrame of (nitems, ncategories)
		category annotations for each item
	Phi : DataFrame of (ncategories, emb_dim)
		centroids for each category of items
	reward : class Reward
		encodes the "true" reward for the problem
	'''
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
	if (emb_dim is None):
		items["Year"] = [x.split(")")[len(x.split(")"))-2**int(len(x.split(")"))>1)].split("(")[-1].split("–")[-1] if (len(x.split("("))>1) else "0" for x in items["title"]]
		for genre in all_genres:
			items[genre] = [int(genre in x) for x in items["genres"]]
		items = items[["Year"]+all_genres]
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
	ratings = ratings.loc[row_idx][col_idx]
	threshold = int(np.max(ratings.values)/2)+1
	#ratings[(ratings!=0)&(ratings<threshold)] = -1
	#ratings[(ratings!=0)&(ratings>=threshold)] = 1
	## Filter
	if (nratings is not None):
		assert nratings is None or nratings<=ratings.shape[0]
		ratings_list = np.array(np.argwhere(ratings.values>0).tolist()[:nratings])
		ratings = ratings.iloc[ratings_list[:,0].flatten().tolist()]
		ratings = ratings[ratings.columns[ratings_list[:,1].flatten().tolist()]]
		items = items.loc[ratings.index]
		users = users.loc[ratings.columns]
		item_categories = item_categories.loc[ratings.index]
		idx = item_categories.sum(axis=0)>0
		item_categories = item_categories[idx]
		Phi = Phi.loc[idx]
	if (nitems is not None):
		assert nitems is None or nitems<=items.shape[0]
		item_idx = ratings.sum(axis=1).sort_values(ascending=False).index[:nitems]
		ratings = ratings.loc[item_idx]
		idx = ratings.sum(axis=0)>0
		ratings = ratings.loc[idx]
		users = users.loc[ratings.columns]
		item_categories = item_categories.loc[ratings.index]
		idx = item_categories.sum(axis=0)>0
		item_categories = item_categories[idx]
		Phi = Phi.loc[idx] 
	if (nusers is not None):
		assert nusers is None or nusers<=users.shape[0]
		user_idx = ratings.sum(axis=1).sort_values(ascending=False).index[:nusers]
		ratings = ratings.loc[user_idx]
		idx = ratings.sum(axis=0)>0
		ratings = ratings.loc[idx]
		items = items.loc[ratings.columns]
		item_categories = item_categories.loc[ratings.index]
		idx = item_categories.sum(axis=0)>0
		item_categories = item_categories[idx]
		Phi = Phi.loc[idx]
	nitems = items.shape[0] 
	nratings = int(ratings.sum().sum())
	nusers = users.shape[0] 
	ncategories = item_categories.shape[1]
	## Define user contexts
	npulls = np.zeros((nusers, nitems))
	contexts = np.zeros((nusers, nitems))
	## Generate ratings 
	ratings_ = [None] * nratings
	ratings_list = np.array(np.argwhere(ratings.values>0).tolist()[:nratings])
	for nrat, [i, u] in tqdm(enumerate(ratings_list.tolist())):
		## Get reward
		rat = ratings.values[i,u]-threshold
		bin_context = context_array2int(contexts[u].flatten(), threshold)
		item_cat = "".join(list(map(str,item_categories.loc[items.index[i]].values.flatten().tolist())))
		ratings_[nrat] = [u, i, npulls[u,i], item_cat, bin_context, rat]
		## Update user context
		contexts[u, i] = (contexts[u, i]*npulls[u,i]+rat)/(npulls[u,i]+1)
		npulls[u,i] += 1
		#if (nrat > 799): ##
		#	ratings_ = ratings_[:nrat]
		#	break ##
	ratings_ = ratings_[:nrat]
	ratings_ = np.array(ratings_, dtype=object)
	## 5. Reward 
	Theta, item_embeddings = learn_from_ratings(ratings_[:1000,:], items, emb_dim)
	emb_dim = item_embeddings.shape[1]
	reward = SyntheticReward(Theta, item_embeddings, sigma=sigma, m=threshold)
	return ratings_, {"item_embeddings": item_embeddings, "user_embeddings": users, "item_categories": item_categories, "category_embeddings": Phi}, reward
	
if __name__=="__main__":
	nusers=nitems=10
	nratings=80
	ncategories=2
	emb_dim=512
	emb_dim_user=11
	print("_"*27)
	if (False):
		print("SYNTHETIC")
		ratings_, info, reward = synthetic(nusers, nitems, nratings, ncategories, emb_dim=emb_dim, emb_dim_user=emb_dim_user, S=1., Sp=1., m=3, sigma=1., loc=0, scale=1)
		print("Ratings")
		print(ratings_.shape)
		print(ratings_[:5,:])
		item_embeddings, user_embeddings, item_categories, Phi = [info[s] for s in ["item_embeddings", "user_embeddings", "item_categories", "category_embeddings"]]
		print("Items")
		print(item_embeddings.shape == (nitems, emb_dim))
		print("Users")
		print(user_embeddings.shape == (nusers, emb_dim_user))
		print("Categories")
		print(item_categories)
		print("Phi")
		print(Phi)
		print(get_context_from_rating(ratings_[-1]))
		context = context_int2array(get_context_from_rating(ratings_[-1]), nitems)
		action_emb = item_embeddings.loc[item_embeddings.index[0]].values.reshape(1,-1)
		print(reward.get_reward(context, action_emb))
		print(reward.get_means(context, action_emb))
		print("_"*27)
	print("MOVIELENS")
	ratings_, info, reward = movielens(nusers=None, nitems=None, nratings=None, ncategories=None, emb_dim=8) 
	print("Ratings")
	print(ratings_.shape)
	print(ratings_[:5,:])
	item_embeddings, user_embeddings, item_categories, Phi = [info[s] for s in ["item_embeddings", "user_embeddings", "item_categories", "category_embeddings"]]
	print("Items")
	print(item_embeddings.shape == (nitems, emb_dim))
	print("Users")
	print(user_embeddings.shape == (nusers, emb_dim_user))
	print("Categories")
	print(item_categories)
	print("Phi")
	print(Phi)
	print(get_context_from_rating(ratings_[-1]))
	context = context_int2array(get_context_from_rating(ratings_[-1]), nitems)
	action_emb = item_embeddings.loc[item_embeddings.index[0]].values.reshape(1,-1)
	print(reward.get_reward(context, action_emb))
	print(reward.get_means(context, action_emb))
	print("_"*27)
