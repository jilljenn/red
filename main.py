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

with open('config.yml', 'r') as f:
	params = yaml.safe_load(f)
for param, v in params.items():
	globals()[param] = v
assert k<=emb_dim
assert data_type in ["movielens","synthetic"]
np.random.seed(seed)

## Data generation
ratings, item_embeddings, item_categories, Phi, reward = eval(data_type)(nusers, nitems, nratings, ncategories, emb_dim, S=S, Sp=Sp, m=m, sigma=sigma)
pretty_ratings = pd.DataFrame(ratings, columns=["user","item","category_id","user_context","reward"], index=range(len(ratings)))
print(pretty_ratings)
print(pretty_ratings["reward"].value_counts())

## Fit a policy on previous interactions
policy = eval(policy)()
policy.fit(ratings, item_embeddings, item_categories, Phi)

recs, scores = policy.predict(int(ratings[-1,0]), ratings[-1,3], k, item_embeddings, item_categories)
print(f"User {int(ratings[-1,0])} with initial context {ratings[-1,3]} recommended items {recs} with scores {np.array(scores).round(3)}")

## Simulate the results from the policy
stime = time()
cum_reward, cum_diversity_intra, cum_diversity_inter, rewards_policy, diversity_intra_policy, diversity_inter_policy, w_policy, w_oracle = simulate(k, policy, reward, horizon, nusers, ncategories, m, item_embeddings, item_categories, gamma=gamma, compute_allocation=True)
print(f"Policy {policy.name}\tReward={np.round(cum_reward, ndec)}\tDiversity (intrabatch)={np.round(cum_diversity_intra,ndec)}\tDiversity (interbatch)={np.round(cum_diversity_inter,ndec)}\tTime={int(time()-stime)} sec.\n\n")

## Compare to oracle policies (with access to the true distributions)
for oracle in [TrueRewardPolicy(), OraclePolicy()]:
	oracle.fit(reward, item_embeddings, item_categories)
	stime = time()
	cum_reward, cum_diversity_intra, cum_diversity_inter, rewards, diversity_intra, diversity_inter, _, _ = simulate(k, oracle, reward, horizon, nusers, ncategories, m, item_embeddings, item_categories, gamma=gamma, compute_allocation=False)
	print(f"Policy {oracle.name}\tReward={np.round(cum_reward, ndec)}\tDiversity (intrabatch)={np.round(cum_diversity_intra,ndec)}\tDiversity (interbatch)={np.round(cum_diversity_inter,ndec)}\tTime={int(time()-stime)} sec.\n\n")
	if (oracle.name=="TrueRewardPolicy"):
		rewards_true, diversity_intra_true, diversity_inter_true = deepcopy(rewards), deepcopy(diversity_intra), deepcopy(diversity_inter)
	else:
		rewards_oracle, diversity_intra_oracle, diversity_inter_oracle = deepcopy(rewards), deepcopy(diversity_intra), deepcopy(diversity_inter)
		
## TODO
## check the oracle policy (recommend np.int64(99))
## check computation of the diversity measure with DPP

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

axes[0].set_ylabel("Reward")
axes[1].set_ylabel("Diversity(inter)")
axes[2].set_ylabel("Diversity(intra)")
axes[3].set_ylabel("Counts")

axes[0].legend()
plt.savefig(f"results_{policy.name}.png", bbox_inches="tight")
plt.close()
