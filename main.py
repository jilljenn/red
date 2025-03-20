#coding:utf-8

import pandas as pd
import numpy as np
import pickle
import yaml
from joblib import Parallel, delayed, parallel_backend
from multiprocessing import cpu_count
from tqdm import trange
import json
import os

from data import synthetic, movielens, SyntheticReward
from simulate import single_run, single_trajectory
from plots import plot_umap, plot_regret
from tools import *

with open('config.yml', 'r') as f:
	params = yaml.safe_load(f)
for param, v in params.items():
	globals()[param] = v
with open(f"{params['exp_name']}_parameters.json", "w") as f:
	json.dump(params, f, ensure_ascii=False)
assert k<=emb_dim
assert data_type in ["movielens","synthetic"]
seed_everything(seed)

## 1. Data generation
if (data_type=="synthetic"):
	ratings, info, reward = synthetic(nusers, nitems, nratings, ncategories, emb_dim=emb_dim, emb_dim_user=emb_dim_user, p_visit=p_visit)
elif (data_type=="movielens"):
	if (not os.path.exists(f"{exp_name}_movielens_instance.pck")):
		ratings, info, reward = movielens(nratings=nratings, ncategories=ncategories, emb_dim=emb_dim,  emb_dim_user=emb_dim_user, p_visit=p_visit, savename=f"{exp_name}_movielens_instance.pck") 
	else:
		with open(f"{exp_name}_movielens_instance.pck", "rb") as f:
			di = pickle.load(f)
		ratings, info, theta = [di[n] for n in ["ratings", "info", "theta"]]
		reward = SyntheticReward(info["item_embeddings"].values, add_params=dict(theta=theta, item_categories=info["item_categories"].values, p_visit=p_visit))
else:
	raise ValueError(f"{data_type} is not implemented.")
nitems = info["item_embeddings"].shape[0]
	
pretty_ratings = pd.DataFrame(ratings, columns=["user","item","#recommended","category_id","user_context","reward"], index=range(len(ratings)))
print(pretty_ratings)
print(pretty_ratings["reward"].value_counts())

## 2. Fit a policy on previous interactions and simulate on prior contexts
assert njobs==1
results_traj = {}
for policy in policies:
	## Generate a trajectory
	results, contexts = single_trajectory(policy, info, ratings, k, horizon_traj, reward, gamma=gamma, verbose=verbose, only_available=only_available)
	results_traj.update({policy: (results, contexts)})

## 3. User scatter plots of UMAPs of item embeddings according to user feedback (non selected, selected, selected and liked, selected and disliked)
plot_umap(info["item_embeddings"].values, results_traj, k, fig_title=f"{exp_name}_figure2")

## 4. Simulate the results from the policy
seeds = np.random.choice(range(int(1e8)), size=niters)
if ((niters==1) or (njobs==1)):
	results_list = [single_run(policies, info, ratings, nitems, k, horizon, reward, prob_new_user, gamma, verbose, seeds[iterr], savefname=f"{exp_name}_seed={seeds[iterr]}_intermediary_results.pck") for iterr in trange(niters)]
else:
	with parallel_backend('loky', inner_max_num_threads=njobs):
		results_list = Parallel(n_jobs=njobs, backend='loky')(delayed(single_run)(policies, info, ratings, nitems, k, horizon, reward, prob_new_user, gamma, verbose, seeds[iterr], savefname=f"{exp_name}_seed={seeds[iterr]}_intermediary_results.pck") for iterr in trange(niters))

## 5. Plots for reward and diversity
plot_regret(results_list, policies, horizon, fig_title=f"{exp_name}_figure1")

