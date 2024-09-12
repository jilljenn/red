#coding:utf-8

from policies import Oracle, TrueRewardPolicy

import numpy as np

def simulate(k, policy, generator, users, horizon, item_embeddings, item_categories, contexts, gamma=1.0, verbose=False, compute_allocation=False):
	oracle = Oracle()
	oracle.fit(generator)
	user_order = np.random.choice(users, size=horizon)
	cum_reward, cum_diversity_intra, cum_diversity_inter = 0, 0, 0
	rewards, diversity_intra, diversity_inter = [[0]+([None]*horizon) for _ in range(3)]
	for t in range(1,horizon+1):
		user = user_order[t-1]
		recs, _ = policy.predict(user, contexts[user], k, item_embeddings, item_categories)
		dd = oracle.diversity(item_embeddings, item_categories, recs)
		for item_k in range(k):
			b, c, d = oracle.reward(item_embeddings[recs[item_k]], user, contexts[user], item_categories[recs[item_k]])
			cum_reward += b*gamma**t
			cum_diversity_inter += d*gamma**t
			rewards[t] = b+rewards[t-1]
			diversity_intra[t] = d+diversity_intra[t-1]
			policy.update(user, item_k, b, d, dd)
		cum_diversity_intra += dd*gamma**t 
		diversity_inter[t] = dd+diversity_inter[t-1]
		if (verbose or (t%int(horizon//10)==0)):
			print(f"At t={t}, {policy.name} recommends items {recs} to user {user} (b={np.round(b,3)}, c={np.round(c,3)}, dinter={np.round(d,3)}, dintra={np.round(dd,3)})")
		contexts[user,item_categories[recs]] = 1
	## Final allocation
	w_policy = np.zeros(len(item_embeddings))
	w_oracle = np.zeros(len(item_embeddings))
	if (compute_allocation):
		for user in users:
			w_policy += policy.allocation(user, contexts[user], item_embeddings)
			oracle = TrueRewardPolicy()
			oracle.fit(generator)
			w_oracle += oracle.allocation(user, contexts[user], item_embeddings)
		w_policy /= len(users)
		w_oracle /= len(users)
	return cum_reward, cum_diversity_intra, cum_diversity_inter, rewards, diversity_intra, diversity_inter, w_policy, w_oracle

## TODO other metrics? RMSE, NDCG?
