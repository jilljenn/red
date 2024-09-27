#coding:utf-8

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from tools import context_array2int

class Reward(object):
	def __init__(self):
		pass
		
	def get_reward(self):
		raise NotImplemented
		
	def get_oracle(self):
		raise NotImplemented
		
#####################
## Synthetic       ##
#####################
		
class SyntheticReward(Reward):
	def __init__(self, Theta, Phi, item_categories, sigma=1., S=1., Sp=1., m=1.):
		self.name = "Synthetic"
		self.Theta = Theta
		self.Phi = Phi
		self.sigma = sigma
		self.item_categories = item_categories
		self.nm = 1/(S*Sp)
		self.m = m
		
	def get_reward(self, c, x, k):
		mean = self.get_oracle(c, x, k)
		return np.clip(np.random.normal(mean,self.sigma), -self.m, self.m)
		
	def get_oracle(self, c, x, k):
		Xk = np.tile(x.T, (self.Phi.shape[0], 1))
		dst = self.nm*np.sqrt(np.multiply(self.Phi-Xk, self.Phi-Xk)) ## TODO add sqrt
		mean = c.T.dot(dst).dot(self.Theta[self.item_categories[k], :].T)
		print(mean)
		return mean

def synthetic(nusers, nitems, nratings, ncategories, emb_dim, S=1., Sp=1., m=3, sigma=1., loc=5, scale=10):
	'''
	Parameters
	----------
	nusers : int
		number of users
	nitems : int
		number of items
	nratings : int
		number of ratings user-item to generate
	emb_dim : int
		number of dimensions for item embeddings
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
		the item category, the user context in 2m binary integers, 
		the (integer) reward
	item_embeddings : array of shape (nitems, emb_dim)
		item embeddings
	item_categories : array of (nitems, 1)
		category annotations for each item
	Phi : array of (ncategories, emb_dim)
		centroids for each cluster of items
	reward : class Reward
		encodes the "true" reward for the problem
	'''
	## Generate item embeddings
	item_embeddings = np.random.normal(loc, scale, size=(nitems, emb_dim))
	item_embeddings /= np.linalg.norm(item_embeddings)/S
	## Define item categories
	item_cluster = KMeans(n_clusters=ncategories)
	item_cluster.fit(item_embeddings)
	item_categories = item_cluster.labels_
	Phi = item_cluster.cluster_centers_
	## Define model parameter
	Theta = np.random.normal(loc, scale, size=(ncategories, emb_dim))
	Theta /= np.linalg.norm(Theta)/Sp
	## Define user contexts
	contexts = np.zeros((nusers, ncategories))
	## Generate ratings
	ratings = [None]*nratings
	all_pairs = np.array([(u,i) for u in range(nusers) for i in range(nitems)])
	user_item_pairs = all_pairs[np.random.choice(len(all_pairs), size=nratings)]
	reward = SyntheticReward(Theta, Phi, item_categories, sigma=sigma, S=S, Sp=Sp, m=m)
	for nrat, [u, i] in enumerate(user_item_pairs.tolist()):
		## Get reward
		rat = reward.get_reward(contexts[u], item_embeddings[i], i)
		bin_context = context_array2int(contexts[u].flatten(), m)
		ratings[nrat] = [u, i, item_categories[i], bin_context, int(rat)]
		## Update user context
		contexts[u,item_categories[i]] += int(rat)
	return np.array(ratings), item_embeddings, item_categories, Phi, reward
	
#####################
## MovieLens       ##
#####################

## https://github.com/kuredatan/projet-gml
## https://github.com/jilljenn/red/blob/main/notebooks/Movielens-Kaggle.ipynb
class MovielensReward(object):
	def __init__(self):
		pass
		
	def get_reward(self):
		raise NotImplemented
		
	def get_oracle(self):
		raise NotImplemented
		
def movielens(nusers, nitems, nratings, ncategories, emb_dim, S=1., Sp=1., m=1, sigma=1.):
	'''
	Parameters
	----------
	nusers : int
		number of users
	nitems : int
		number of items
	nratings : int
		number of ratings user-item to generate
	emb_dim : int
		number of dimensions for item embeddings
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
		the item category, the user context in 2m binary integers, 
		the (integer) reward
	item_embeddings : array of shape (nitems, emb_dim)
		item embeddings
	item_categories : array of (nitems, 1)
		category annotations for each item
	Phi : array of (ncategories, emb_dim)
		centroids for each cluster of items
	reward : class Reward
		encodes the "true" reward for the problem
	'''
	raise ValueError("Not implemented yet.")
