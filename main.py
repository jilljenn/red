#coding:utf-8

from copy import deepcopy
import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt
import yaml
from joblib import Parallel, delayed, parallel_backend
from multiprocessing import cpu_count
from tqdm import trange
import os

from data import synthetic, movielens
from policies import *
from simulate import simulate, simulate_trajectory
from tools import *

with open('config.yml', 'r') as f:
	params = yaml.safe_load(f)
for param, v in params.items():
	globals()[param] = v
#assert k<=emb_dim
assert data_type in ["movielens","synthetic"]
seed_everything(seed)

## 1. Data generation
if (data_type=="synthetic"):
	ratings, info, reward = synthetic(nusers, nitems, nratings, ncategories, emb_dim=emb_dim, emb_dim_user=emb_dim_user, p_visit=p_visit)
elif (data_type=="movielens"):
	if (not os.path.exists("movielens_instance.pck")):
		ratings, info, reward = movielens(nusers=None, nitems=None, nratings=None, ncategories=None, emb_dim=emb_dim, p_visit=p_visit, savename="movielens_instance.pck") 
	else:
		with open("movielens_instance.pck", "rb") as f:
			di = pickle.load(f)
		ratings, info, theta = [di[n] for n in ["ratings", "info", "theta"]]
		reward = SyntheticReward(info["item_embeddings"], add_params=dict(theta=theta, item_categories=info["item_categories"], p_visit=p_visit))
else:
	raise ValueError(f"{data_type} is not implemented.")
	
pretty_ratings = pd.DataFrame(ratings, columns=["user","item","#recommended","category_id","user_context","reward"], index=range(len(ratings)))
print(pretty_ratings)
print(pretty_ratings["reward"].value_counts())

## 2. Fit a policy on previous interactions and simulate on prior contexts
def single_run(policies, info, ratings, nitems, k, horizon, reward, prob_new_user, gamma=gamma, verbose=False, random_seed=13234):
	seed_everything(int(random_seed))
	trained_policies = []
	for policy_name in policies:
		policy = eval(policy_name)(info, random_state=random_seed)
		policy.fit(ratings)

		if (False):
			user_context = context_int2array(get_context_from_rating(ratings[-1]), nitems)
			rt_ids = policy.predict(user_context, k)
			rt = reward.item_embeddings[rt_ids,:]
			yt = reward.get_reward(user_context, rt)
			# ratings[-1,3] = context_array2int(user_context, reward.m)}
			print(f"{policy_name}: User with initial context {get_context_from_rating(ratings[-1])} ({user_context.ravel()}) recommended items {rt_ids} with scores {np.mean(yt)}")	
		trained_policies += [policy]
		
	user_contexts = np.array([context_int2array(get_context_from_rating(ratings[i]), nitems) for i in range(ratings.shape[0])])
	stime = time()
	results = simulate(k, horizon, trained_policies, reward, user_contexts, prob_new_user=prob_new_user, gamma=gamma, verbose=verbose)
	runtime = time()-stime
	#print("\n\n")
	#for policy in trained_policies:
	#	print(f"Policy {policy.name}\n\tRegret Reward={np.sum(results[policy.name][:,0])}\tDiversity (intrabatch)={np.sum(results[policy.name][:,1])}\tDiversity (interbatch)={np.sum(results[policy.name][:,2])}\tTime={runtime} sec.\n")
	#print(f"Reward oracle\n\tRegret Reward={np.sum(results['oracle reward'][:,0])}\tDiversity (intrabatch)={np.sum(results['oracle reward'][:,1])}\tDiversity (interbatch)={np.sum(results['oracle reward'][:,2])}\tTime={runtime} sec.\n")
	#print(f"Diversity oracle\n\tRegret Reward={np.sum(results['oracle diversity'][:,0])}\tDiversity (intrabatch)={np.sum(results['oracle diversity'][:,1])}\tDiversity (interbatch)={np.sum(results['oracle diversity'][:,2])}\tTime={runtime} sec.\n\n")
	return results
	
blank_context = np.zeros((nitems, 1))
def single_trajectory(policy_name, info, ratings, k, horizon, reward, gamma=gamma, context=None, verbose=False, random_seed=13234):
	seed_everything(int(random_seed))
	if (context is None):
		context = blank_context.copy()

	policy = eval(policy_name)(info, random_state=random_seed)
	policy.fit(ratings)
	
	stime = time()
	results = simulate_trajectory(k, horizon, policy, reward, context=context, gamma=gamma, verbose=verbose)
	runtime = time()-stime
	return results
	
assert njobs==1

## 3. User scatter plots of UMAPs of item embeddings according to user feedback (non selected, selected, selected and liked, selected and disliked)
fontsize=15

## Generate a trajectory
item_embs = info["item_embeddings"].values
## Plot UMAP
import umap
dimred_args = dict(n_neighbors=3, min_dist=0.5, metric="euclidean")
with np.errstate(invalid="ignore"): # for NaN or 0 variance matrices
	umap_model = umap.UMAP(**dimred_args)
	embeddings = umap_model.fit_transform(item_embs)

for policy in policies:
	results, contexts = single_trajectory(policy, info, ratings, k, horizon_traj, reward, verbose=verbose)

	item_labels = {t: 0.5*np.ones(item_embs.shape[0]) for t in range(horizon_traj)}
	for t in range(horizon_traj):
		item_lbs = item_labels[t]
		item_lbs[results[t,:k].ravel()] = results[t,k:].ravel()
		item_labels[t] = item_lbs
		
	fig, axes = plt.subplots(nrows=1, ncols=horizon_traj, figsize=(6.5*horizon_traj,6))
	labels = {-1: "selected/disliked", 1: "selected/liked", 0: "selected/not visited", 0.5: "non selected"}
	labels_colors = {-1: "r", 1: "g", 0: "k", 0.5: "b"}
	for t in range(horizon_traj):
		for label in labels:
			embs = embeddings[item_labels[t]==label,:]
			if (embs.shape[0]==0):
				continue
			axes[t].scatter(embs[:,0], embs[:,1], s=200, c=labels_colors[label], marker=".", alpha=0.05 if (label == 0.5) else 0.8, label=labels[label])
		axes[t].set_title(f"Round {t+1}: context {pretty_print_context(contexts[t])[:10]}"+("..." if (len(contexts[t])>10) else ""), fontsize=fontsize)
		if (t==0):
			axes[t].set_ylabel("UMAP C2", fontsize=fontsize)
		axes[t].set_xlabel("UMAP C1", fontsize=fontsize)
		axes[t].set_xticklabels(axes[t].get_xticklabels(), fontsize=fontsize)
		axes[t].set_yticklabels(axes[t].get_yticklabels(), fontsize=fontsize)
		if (t==0):
			axes[t].legend(fontsize=fontsize)
	plt.savefig(f"figure2_{policy}.png", bbox_inches="tight")

## 4. Simulate the results from the policy
seeds = np.random.choice(range(int(1e8)), size=niters)
if ((niters==1) or (njobs==1)):
	results_list = [single_run(policies, info, ratings, nitems, k, horizon, reward, prob_new_user, gamma, verbose, seeds[iterr]) for iterr in trange(niters)]
else:
	with parallel_backend('loky', inner_max_num_threads=njobs):
		results_list = Parallel(n_jobs=njobs, backend='loky')(delayed(single_run)(policies, info, ratings, nitems, k, horizon, reward, prob_new_user, gamma, verbose, seeds[iterr]) for iterr in trange(niters))
#print(results_list)

## 3. Plots for reward and diversity
fontsize=30
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(40,10))
policies_names = policies
colors = {"LinUCB": "gray", "LinUCBDiversity": "pink", "CustomGreedy": "green", "CustomBruteForce": "firebrick", "CustomDPP": "steelblue", "CustomSampling": "rebeccapurple"}

#https://www.mikulskibartosz.name/wilson-score-in-python-example/
def wilson(p, n, z = 1.96): # CI at 95%
	denominator = 1 + z**2/n
	centre_adjusted_probability = p + z*z / (2*n)
	adjusted_standard_deviation = np.sqrt((p*(1 - p) + z*z / (4*n)) / n)

	lower_bound = (centre_adjusted_probability - z*adjusted_standard_deviation) / denominator
	upper_bound = (centre_adjusted_probability + z*adjusted_standard_deviation) / denominator
	return (lower_bound, upper_bound)

handles = []
for policy_name in policies_names:
	for i in range(4):
		values = np.array([np.cumsum(r[policy_name][:,i]) for r in results_list])
		average = values.copy()
		if (len(results_list)>1):
			average = average.mean(axis=0)
		if (len(results_list)>1):
			std = values.std(axis=0)
		else:
			std = np.zeros(average.shape)
		x = np.array(range(horizon))
		h = axes[i].plot(x.ravel(), average.ravel(), label=policy_name, color=colors[policy_name])
		handles.append(h)
		nsim = len(results_list)
		#LB_CI = np.array([float(wilson(m, nsim)[0]) for m in average.flatten().tolist()])
		#UB_CI = np.array([float(wilson(m, nsim)[1]) for m in average.flatten().tolist()])
		LB_CI = average.ravel() - std.ravel()
		if (i != 1):
			LB_CI = np.maximum(LB_CI, 0)
		UB_CI = average.ravel() + std.ravel()
		axes[i].fill_between(x.ravel(), LB_CI, UB_CI, alpha=0.2, color=colors[policy_name])
		axes[i].set_xticklabels(axes[i].get_xticklabels(), fontsize=fontsize)
		axes[i].set_yticklabels(axes[i].get_yticklabels(), fontsize=fontsize)
		axes[i].set_title({0: "Reward regret", 1: "Reward aggregated", 2: "Diversity intra-batch regret", 3: "Diversity inter-batch regret"}[i], fontsize=fontsize)
		axes[i].set_xlabel("Horizon", fontsize=fontsize)
		axes[i].set_ylabel("", fontsize=fontsize)
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles, labels, fontsize=fontsize)
		
plt.savefig("figure1.png", bbox_inches="tight")
plt.close()

