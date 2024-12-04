#coding:utf-8

from copy import deepcopy
import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt
import yaml

from data import synthetic, movielens
from policies import *
from simulate import simulate
from tools import *

with open('config.yml', 'r') as f:
	params = yaml.safe_load(f)
for param, v in params.items():
	globals()[param] = v
assert k<=emb_dim
assert data_type in ["movielens","synthetic"]
np.random.seed(seed)

## 1. Data generation
if (data_type=="synthetic"):
	ratings, info, reward = synthetic(nusers, nitems, nratings, ncategories, emb_dim=emb_dim, emb_dim_user=emb_dim_user, S=S, Sp=Sp, m=m, sigma=sigma)
elif (data_type=="movielens"):
	ratings, info, reward = movielens(nusers=None, nitems=None, nratings=None, ncategories=None, emb_dim=None, sigma=sigma)
else:
	raise ValueError(f"{data_type} is not implemented.")
	
pretty_ratings = pd.DataFrame(ratings, columns=["user","item","category_id","user_context","reward"], index=range(len(ratings)))
print(pretty_ratings)
print(pretty_ratings["reward"].value_counts())
	
## 2. Fit a policy on previous interactions
policy = eval(policy)(info)
policy.fit(ratings)

user_context = context_int2array(ratings[-1,3], nitems)
rt_ids = policy.predict(user_context, k)
rt = reward.item_embeddings[rt_ids,:]
yt = reward.get_reward(user_context, rt)
# ratings[-1,3] = context_array2int(user_context, reward.m)}
print(f"User with initial context {ratings[-1,3]} recommended items {rt_ids} with scores {np.mean(yt)}")

## 3. Simulate the results from the policy	
user_contexts = np.array([context_int2array(uc, nitems) for uc in ratings[:,3].flatten().tolist()])
trained_policies = [policy]

stime = time()
results = simulate(k, horizon, trained_policies, reward, user_contexts, prob_new_user=prob_new_user, verbose=verbose)
runtime = time()-stime
print("\n\n")
for policy in trained_policies:
	print(f"Policy {policy.name}\tReward={np.sum(results[policy.name][:,0])}\tDiversity (intrabatch)={np.sum(results[policy.name][:,1])}\tDiversity (interbatch)={np.sum(results[policy.name][:,2])}\tTime={runtime} sec.\n\n")
print(f"Reward oracle\tReward={np.sum(results['oracle'][:,0])}\tDiversity (intrabatch)={np.sum(results['oracle'][:,1])}\tDiversity (interbatch)={np.sum(results['oracle'][:,2])}\tTime={runtime} sec.\n\n")
#print(results)

## TODO plots
## TODO check allocation
