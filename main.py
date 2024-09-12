#coding:utf-8

from data import generate_data
from policies import *
from tools import simulate

from copy import deepcopy
import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt

params=dict(
	nusers=250, ## number of users
	nitems=1000, ## number of items
	nratings=10000, ## number of ratings
	ncategories=14, ## number of item categories
	threshold=0.5, ## threshold on the diversity
	emb_dim=100, ## item embedding size
	k=3, ## number of items to sample at each time
	seed=1234, ## random seed
	horizon=10000, ## recommendation rounds
	ndec=3, ## number of decimals to print out
	ntests=10000, ## number of samples to compare distributions
	with_visited=1, ## B=1 => C=1, and C=0 => B=0 use or not visited variables
	booking_proba=lambda p : 0.9, ## P(B | C=1) or introduce a dependency in the probability of visiting p
	gamma=0.9 ## discounting factor for the time
)
for param, v in params.items():
	globals()[param] = v
	
assert k<=emb_dim
np.random.seed(seed)

## Data generation
ratings, item_embeddings, item_categories, contexts, generator = generate_data(nusers, nitems, nratings, ncategories, emb_dim, threshold=threshold, with_visited=with_visited)
pretty_ratings = pd.DataFrame(ratings, columns=["user","item","context","event_name","category_id","delta","visited","booking","reward"], index=range(len(ratings)))
print(pretty_ratings)
print(f"#bookings={ratings[:,-2].astype(int).sum()}\t#visits={ratings[:,-3].astype(int).sum()}")

## TODO: what if new users?
users = np.unique(ratings[:,0].astype(int)).tolist() 
contexts = np.zeros((nusers, ncategories))

## Fit a policy on previous interactions
#policy = GaussianProcess()
policy = OnlineLearner()
policy.fit(ratings, item_embeddings)
#recs, scores = policy.predict(int(ratings[0,0]), k, item_embeddings)
#print((recs, scores))

## Simulate the results from the policy
stime = time()
cum_reward, cum_diversity_intra, cum_diversity_inter, rewards_policy, diversity_intra_policy, diversity_inter_policy, w_policy, w_oracle = simulate(k, policy, generator, users, horizon, item_embeddings, item_categories, contexts, gamma=gamma, compute_allocation=True)
print(f"Policy {policy.name}\tB={np.round(cum_reward, ndec)}\tD (intrabatch)={np.round(cum_diversity_intra,ndec)}\tD (interbatch)={np.round(cum_diversity_inter,ndec)}\tTime={int(time()-stime)} sec.")

## Compare to oracle policies (with access to the true distributions)
for oracle in [TrueRewardPolicy(), OraclePolicy()]:
	oracle.fit(generator)
	stime = time()
	cum_reward, cum_diversity_intra, cum_diversity_inter, rewards, diversity_intra, diversity_inter, _, _ = simulate(k, oracle, generator, users, horizon, item_embeddings, item_categories, contexts, gamma=gamma)
	print(f"Oracle {oracle.name}\tB={np.round(cum_reward, ndec)}\tD (intrabatch)={np.round(cum_diversity_intra,ndec)}\tD (interbatch)={np.round(cum_diversity_inter,ndec)}\tTime={int(time()-stime)} sec.")
	if (oracle.name=="TrueRewardPolicy"):
		rewards_true, diversity_intra_true, diversity_inter_true = deepcopy(rewards), deepcopy(diversity_intra), deepcopy(diversity_inter)
	else:
		rewards_oracle, diversity_intra_oracle, diversity_inter_oracle = deepcopy(rewards), deepcopy(diversity_intra), deepcopy(diversity_inter)

## Plot the results
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(25,8))
colors = {policy.name: "blue", "TrueRewardPolicy": "magenta", "OraclePolicy": "green"}
axes[0].plot(range(horizon+1), rewards_policy, color=colors[policy.name], label=policy.name)
axes[0].plot(range(horizon+1), rewards_true, color=colors["TrueRewardPolicy"], label="TrueRewardPolicy")
axes[0].plot(range(horizon+1), rewards_oracle, color=colors["OraclePolicy"], label="OraclePolicy")

axes[1].plot(range(horizon+1), diversity_inter_policy, color=colors[policy.name])
axes[1].plot(range(horizon+1), diversity_inter_true, color=colors["TrueRewardPolicy"])
axes[1].plot(range(horizon+1), diversity_inter_oracle, color=colors["OraclePolicy"])

axes[2].plot(range(horizon+1), diversity_intra_policy, color=colors[policy.name])
axes[2].plot(range(horizon+1), diversity_intra_true, color=colors["TrueRewardPolicy"])
axes[2].plot(range(horizon+1), diversity_intra_oracle, color=colors["OraclePolicy"])

r_policy = np.random.choice(nitems, p=w_policy, size=ntests)
r_oracle = np.random.choice(nitems, p=w_oracle, size=ntests)
axes[3].hist(r_policy, bins=50, color=colors[policy.name], alpha=0.2)
axes[3].hist(r_oracle, bins=50, color=colors["TrueRewardPolicy"], alpha=0.2)

axes[0].set_xlabel("Horizon")
axes[1].set_xlabel("Horizon")
axes[2].set_xlabel("Horizon")
axes[3].set_xlabel("Items")

axes[0].set_ylabel("B")
axes[1].set_ylabel("D(inter)")
axes[2].set_ylabel("D(intra)")
axes[3].set_ylabel("Counts")

axes[0].legend()
plt.savefig(f"results_{policy.name}.png", bbox_inches="tight")
plt.close()
