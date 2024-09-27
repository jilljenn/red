#coding: utf-8

import numpy as np
from tqdm import tqdm
from tools import context_array2int, context_int2array

from policies import *

def simulate(k, policy, reward, horizon, nusers, ncategories, m, item_embeddings, item_categories, gamma, compute_allocation=False, verbose=False):
	oracle = Oracle()
	oracle.fit(reward, item_embeddings, item_categories)
	user_order = np.random.choice(range(nusers), size=horizon)
	user_contexts = np.array([",".join(["0"]*(2*m))]*nusers)
	cum_reward, cum_diversity_intra, cum_diversity_inter = 0, 0, 0
	rewards, diversity_intra, diversity_inter = [[0]+([None]*horizon) for _ in range(3)]
	for t in tqdm(range(1,horizon+1)):
		user = user_order[t-1]
		recs, _ = policy.predict(user, user_contexts[user], k, item_embeddings, item_categories)
		## Diversity across batches
		items = oracle.get_items(user_contexts[user])
		dd = oracle.get_diversity(items, recs)
		for item_k in range(k):
			y = oracle.get_reward(item_k, item_embeddings[recs[item_k]], user, user_contexts[user], item_categories[recs[item_k]])
			d = oracle.get_diversity([r for r in recs if (r != recs[item_k])], [recs[item_k]])
			cum_reward += y*gamma**t
			cum_diversity_inter += d*gamma**t
			rewards[t] = y+rewards[t-1]
			diversity_intra[t] = d+diversity_intra[t-1]
			policy.update(user, recs[item_k], y)
		cum_diversity_intra += dd*gamma**t 
		diversity_inter[t] = dd+diversity_inter[t-1]
		if (verbose or (t%int(horizon//10)==0)):
			print(f"At t={t}, {policy.name} recommends items {recs} to user {user} with context {user_contexts[user]} (r={np.round(y,3)}, dinter={np.round(d,3)}, dintra={np.round(dd,3)})")
		c = context_int2array(user_contexts[user], oracle.ncats)
		c[item_categories[recs]] = 1
		user_contexts[user] = context_array2int(c, m) 
	## Final average allocation
	w_policy = np.zeros(len(item_embeddings))
	w_oracle = np.zeros(len(item_embeddings))
	if (compute_allocation):
		oracle = OraclePolicy()
		oracle.fit(reward, item_embeddings, item_categories)
		for user in range(nusers):
			w_policy += policy.allocation(user, user_contexts[user], item_embeddings)
			w_oracle += oracle.allocation(user, user_contexts[user], item_embeddings)
		w_policy /= nusers
		w_oracle /= nusers
	return cum_reward, cum_diversity_intra, cum_diversity_inter, rewards, diversity_intra, diversity_inter, w_policy, w_oracle
