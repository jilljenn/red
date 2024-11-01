#coding:utf-8

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from subprocess import Popen
import os
import pandas as pd

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
		dst = self.nm*np.multiply(self.Phi-Xk, self.Phi-Xk)
		mean = c.T.dot(dst).dot(self.Theta[self.item_categories[k], :].T)
		return mean

def synthetic(nusers, nitems, nratings, ncategories, emb_dim, S=1., Sp=1., m=3, sigma=1., loc=0, scale=1):
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
	for nrat, [u, i] in tqdm(enumerate(user_item_pairs.tolist())):
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
		
def movielens(nusers=None, nitems=None, nratings=None, ncategories=None, emb_dim=None, S=1., Sp=1., m=5, sigma=None):
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
	## Create the MovieLens data set
	if (not os.path.exists("ml-latest-small/")):
		proc = Popen("wget -qO - https://files.grouplens.org/datasets/movielens/ml-latest-small.zip |  bsdtar -xvf -".split(" "))
		proc.wait()
	## 1. Movie feature matrix and item categories
	items = pd.read_csv("ml-latest-small/movies.csv", sep=",", index_col=0)
	all_categories = items["genres"].unique()
	all_genres = list(set([y for x in items["genres"] for y in x.split("|")]))
	assert ncategories is None or ncategories<=len(np.unique(item_categories))
	item_categories = np.array([np.argwhere(items.loc[i]["genres"]==all_categories).flatten() for i in items.index]).flatten()
	## TODO
	ncategories = len(np.unique(item_categories))
	assert nitems is None or nitems<=items.shape[0]
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
		select = np.argsort((items_mat!=0).mean(axis=1))[-emb_dim:]
		items = pd.DataFrame(items_mat[select,:], columns=items.index, index=vectorizer.get_feature_names_out()[select]).T
	emb_dim = items.shape[1]
	items = items.astype(float)
	items.index = items.index.astype(str)
	items.columns = items.columns.astype(str)
	## 2. Phi: embeddings of categories
	Phi = np.zeros((ncategories, emb_dim))
	for ncat in range(ncategories):
		Phi[ncat,:] = items.values[item_categories==ncat,:].mean(axis=0)
	## 3. Ratings
	## TODO
	raise ValueError
	### reward
	users = pd.read_csv("ml-latest-small/tags.csv", sep=",")
	users["count"] = 1
	users = pd.pivot_table(users, columns=["userId"], values=["count"], index=["tag"], aggfunc="sum", fill_value=0)
	#users.reset_index(level=[0,0])
	users = users.astype(float)
	users.index = users.index.astype(str)
	users.columns = users.columns.get_level_values(1).astype(str)
	ratings = pd.read_csv("ml-latest-small/ratings.csv", sep=",")
	ratings = pd.pivot_table(ratings, columns=["userId"], values=["rating"], index=["movieId"], aggfunc="mean", fill_value=0)
	ratings = ratings.astype(float)
	ratings.index = ratings.index.astype(str)
	ratings.columns = ratings.columns.get_level_values(1).astype(str)
	col_idx, row_idx = [x for x in list(ratings.columns) if (x in users.columns)], [x for x in list(ratings.columns) if (x in items.columns)]
	users = users[col_idx]
	items = items[row_idx]
	ratings = ratings.loc[row_idx][col_idx]
	threshold = int(np.max(ratings.values)/2)+1
	ratings[(ratings!=0)&(ratings<threshold)] = -1
	ratings[(ratings!=0)&(ratings>=threshold)] = 1
	raise ValueError("Not implemented yet.")
	
if __name__=="__main__":
	movielens(nusers=None, nitems=None, nratings=None, ncategories=None, emb_dim=100, S=1., Sp=1., m=5, sigma=None)
