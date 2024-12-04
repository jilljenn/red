#coding: utf-8

import numpy as np
from tqdm import tqdm
from tools import context_array2int, context_int2array, get_available_actions

def simulate(k, horizon, trained_policies, reward, user_contexts, prob_new_user=0.1, gamma=1., verbose=False):
	'''
	Simulate some rounds of recommendations
	
	-- 
	Parameters
	k : int
		Number of recommended items
	horizon : int
		Number of recommendation rounds
	trained_policies : list of Policy class
		Policies trained on prior ratings
	reward : Reward class
		Environment
	user_contexts : array of shape (n_ratings, L)
		Prior user contexts (integer format)
	prob_new_user : float
		Probability of introducing a new user (blank context) during a round
	gamma : float
		Discount factor
	verbose : bool
		Printing
		
	Returns
	results : dictionary of arrays of shape (horizon, 3)
		For each policy, reward values, diversity inside and across batches of k recommended items across rounds
	'''
	results = {policy.name: np.zeros((horizon, 3)) for policy in trained_policies}
	results.update({"oracle": np.zeros((horizon, 3))})
	for t in tqdm(range(1,horizon+1)):
		draw = np.random.choice([0,1], p=[1-prob_new_user, prob_new_user])
		if (draw):
			context = np.zeros((1, user_contexts.shape[1]))
		else:
			i_user = np.random.choice(range(user_contexts.shape[0]))
			context = user_contexts[i_user]
		available_items_ids = get_available_actions(context)
		if (available_items_ids.sum()==0):
			print(f"No item available for user {context_array2int(context, reward.m)}")
			continue
		available_items = reward.item_embeddings[available_items_ids,:] ## do not recommend again an item
		context_embeddings = reward.item_embeddings[~available_items_ids,:]
		for policy in trained_policies:
			rt_ids = policy.predict(context, k, available_items=available_items)
			rt = available_items[rt_ids,:]
			yt = reward.get_reward(context, rt)
			div_inter = reward.get_diversity(rt, context_embeddings=context_embeddings)
			div_intra = reward.get_diversity(rt)
			res = results[policy.name]
			res[t-1,:] = [np.mean(yt)*gamma**t, div_intra, div_inter]
			results.update({policy.name: res})
			if (verbose or (t%int(horizon//10)==0)):
				print(f"At t={t}, {policy.name} recommends items {rt_ids} to user {context_array2int(context, reward.m)} (r={np.mean(yt)}, dintra={div_intra}, dinter={div_inter})")
		rt_ids = reward.get_oracle(context, k, available_items=available_items)
		rt = available_items[rt_ids,:]
		means = reward.get_means(context, rt)
		div_inter = reward.get_diversity(rt, context_embeddings=context_embeddings)
		div_intra = reward.get_diversity(rt)
		res = results["oracle"]
		if (verbose or (t%int(horizon//10)==0)):
			print(f"At t={t}, Reward Oracle recommends items {rt_ids} to user {context_array2int(context, reward.m)} (r={np.mean(means)}, dintra={div_intra}, dinter={div_inter})")
		res[t-1,:] = [np.mean(means), div_intra, div_inter]
		results.update({"oracle": res})
	return results
	
if __name__=="__main__":
	pass
