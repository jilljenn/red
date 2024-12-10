#coding:utf-8

from copy import deepcopy
import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt
import yaml
from joblib import Parallel, delayed, parallel_backend
from multiprocessing import cpu_count

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
seed_everything(seed)

## 1. Data generation
if (data_type=="synthetic"):
	ratings, info, reward = synthetic(nusers, nitems, nratings, ncategories, emb_dim=emb_dim, emb_dim_user=emb_dim_user, S=S, Sp=Sp, m=m, sigma=sigma)
elif (data_type=="movielens"):
	ratings, info, reward = movielens(nusers=None, nitems=None, nratings=None, ncategories=None, emb_dim=emb_dim) 
else:
	raise ValueError(f"{data_type} is not implemented.")
	
pretty_ratings = pd.DataFrame(ratings, columns=["user","item","#recommended","category_id","user_context","reward"], index=range(len(ratings)))
print(pretty_ratings)
print(pretty_ratings["reward"].value_counts())

def single_run(policies, info, ratings, nitems, k, horizon, reward, prob_new_user, verbose=False, random_seed=13234):
	seed_everything(int(random_seed))
	## 2. Fit a policy on previous interactions
	trained_policies = []
	for policy_name in policies:
		policy = eval(policy_name)(info)
		policy.fit(ratings)

		if (False):
			user_context = context_int2array(get_context_from_rating(ratings[-1]), nitems)
			rt_ids = policy.predict(user_context, k)
			rt = reward.item_embeddings[rt_ids,:]
			yt = reward.get_reward(user_context, rt)
			# ratings[-1,3] = context_array2int(user_context, reward.m)}
			print(f"{policy_name}: User with initial context {get_context_from_rating(ratings[-1])} ({user_context}) recommended items {rt_ids} with scores {np.mean(yt)}")

		## 3. Simulate the results from the policy	
		user_contexts = np.array([context_int2array(get_context_from_rating(ratings[i]), nitems) for i in range(ratings.shape[0])])
		trained_policies += [policy]

	stime = time()
	results = simulate(k, horizon, trained_policies, reward, user_contexts, prob_new_user=prob_new_user, verbose=verbose)
	runtime = time()-stime
	print("\n\n")
	for policy in trained_policies:
		print(f"Policy {policy.name}\n\tReward={np.sum(results[policy.name][:,0])}\tDiversity (intrabatch)={np.sum(results[policy.name][:,1])}\tDiversity (interbatch)={np.sum(results[policy.name][:,2])}\tTime={runtime} sec.\n")
	print(f"Reward oracle\n\tReward={np.sum(results['oracle reward'][:,0])}\tDiversity (intrabatch)={np.sum(results['oracle reward'][:,1])}\tDiversity (interbatch)={np.sum(results['oracle reward'][:,2])}\tTime={runtime} sec.\n")
	print(f"Diversity oracle\n\tReward={np.sum(results['oracle diversity'][:,0])}\tDiversity (intrabatch)={np.sum(results['oracle diversity'][:,1])}\tDiversity (interbatch)={np.sum(results['oracle diversity'][:,2])}\tTime={runtime} sec.\n\n")
	return results
	
assert njobs==1

seeds = np.random.choice(range(int(1e8)), size=niters)
if ((niters==1) or (njobs==1)):
	results_list = [single_run(policies, info, ratings, nitems, k, horizon, reward, prob_new_user, verbose, seeds[iterr]) for iterr in range(niters)]
else:
	with parallel_backend('loky', inner_max_num_threads=njobs):
		results_list = Parallel(n_jobs=njobs, backend='loky')(delayed(single_run)(policies, info, ratings, nitems, k, horizon, reward, prob_new_user, verbose, seeds[iterr]) for iterr in range(niters))
#print(results_list)

## 3. Plots for reward and diversity
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30,10))
policies_names = policies+["oracle reward", "oracle diversity"]
colors = {"LogisticUCB": "b", "LogisticUCBDiversity": "c", "oracle reward": "k", "oracle diversity": "g"}

for policy_name in policies_names:
	for i in range(3):
		values = np.array([np.cumsum(r[policy_name][:,i]) for r in results_list])
		average = values.copy()
		if (len(results_list)>1):
			average = average.mean(axis=0)
		if (len(results_list)>1):
			std = values.std(axis=0)
		else:
			std = np.zeros(average.shape)
		x = np.array(range(horizon))
		axes[i].plot(x.ravel(), average.ravel(), label=policy_name, color=colors[policy_name])
		axes[i].fill_between(x.ravel(), average.ravel() - std.ravel(), average.ravel() + std.ravel(), alpha=0.2, color=colors[policy_name])
		axes[i].set_title({0: "Reward", 1: "Diversity intra-batch", 2: "Diversity inter-batch"}[i])
		axes[i].set_xlabel("Horizon")
		axes[i].set_ylabel("Value")
		plt.legend()
		
plt.savefig("figure1.png", bbox_inches="tight")
plt.close()
