#coding: utf-8

import numpy as np
from tqdm import tqdm
from tools import context_array2int, context_int2array, get_available_actions

def simulate(k, horizon, trained_policies, reward, user_contexts, prob_new_user=0.1, gamma=1., verbose=False, aggreg="sum"):
	'''
	Simulate some rounds of recommendations
	
	Parameters
	--
	k : int
		the number of recommended items
	horizon : int
		the number of recommendation rounds
	trained_policies : list of Policy class
		the policies trained on prior ratings
	reward : Reward class
		the environment
	user_contexts : array of shape (n_ratings, L)
		the prior user contexts (array format)
	prob_new_user : float
		the probability of introducing a new user (blank context) during a round
	gamma : float
		the time discount factor
	verbose : bool
		whether to print info
	aggreg : str
		the batch aggregation function
		
	Returns
	--
	results : dictionary of arrays of shape (horizon, 4)
		for each policy, reward values, reward sum, diversity inside and across batches of k recommended items across rounds
	'''
	results = {policy.name: np.zeros((horizon, 4)) for policy in trained_policies}
	#results.update({"oracle reward": np.zeros((horizon, 3)), "oracle diversity": np.zeros((horizon, 3))})
	
	if (aggreg=="sum"):
		aggreg_func = np.sum
	elif (aggreg=="mean"):
		aggreg_func = np.mean
	elif (aggreg=="max"):
		aggreg_funct = np.max
	else:
		raise ValueError(f"Aggregation method {aggreg} not implemented")
	
	for t in tqdm(range(1,horizon+1), leave=False):
	
		## draw a user context or create a new user
		draw = np.random.choice([0,1], p=[1-prob_new_user, prob_new_user])
		if (draw):
			context = np.zeros((1, user_contexts.shape[1]))
		else:
			i_user = np.random.choice(range(user_contexts.shape[0]))
			context = user_contexts[i_user].reshape(1, -1)
		
		rt_ids = reward.get_oracle_reward(context, k)
		rt = reward.item_embeddings[rt_ids,:]
		means = reward.get_means(context, rt)
		div_inter = reward.get_diversity(rt[means.flatten()>0,:], context=context, action_ids=rt_ids[means.flatten()>0])
		div_intra = reward.get_diversity(rt[means.flatten()>0,:], action_ids=rt_ids[means.flatten()>0])
		r, d, dd = [np.round(x,3) for x in [aggreg_func(means), aggreg_func(div_intra), aggreg_func(div_inter)]]
		if (verbose):# or (t%int(horizon//10)==0)):
			print(f"At t={t}, Reward Oracle recommends items {rt_ids} to user {context.ravel()} (r={r}, dintra={d}, dinter={dd})")
		#res = results["oracle reward"]
		#res[t-1,:] = [aggreg_func(means), div_intra, div_inter]
		#results.update({"oracle reward": res})
		best_reward = aggreg_func(means)
		
		rt_ids = reward.get_oracle_diversity(context, k, aggreg_func)
		rt = reward.item_embeddings[rt_ids,:]
		means = reward.get_means(context, rt)
		#div_inter = reward.get_diversity(rt[means.flatten()>0,:], context=context, action_ids=rt_ids[means.flatten()>0]) ## true
		#div_intra = reward.get_diversity(rt[means.flatten()>0,:], action_ids=rt_ids[means.flatten()>0])                  ## true
		div_inter = reward.get_diversity(rt, context=context, action_ids=rt_ids) ## to get positive regret
		div_intra = reward.get_diversity(rt, action_ids=rt_ids)                  ## to get positive regret
		best_diversity_intra = aggreg_func(div_intra)
		best_diversity_inter = aggreg_func(div_inter)
		r, d, dd = [np.round(x,3) for x in [aggreg_func(means), aggreg_func(div_intra), aggreg_func(div_inter)]]
		#res = results["oracle diversity"]
		#res[t-1,:] = [aggreg_func(means), div_intra, div_inter]
		#results.update({"oracle diversity": res})
		if (verbose):# or (t%int(horizon//10)==0)):
			print(f"At t={t}, Diversity Oracle recommends items {rt_ids} to user {context.ravel()} (r={r}, dintra={d}, dinter={dd})")
		
		for policy in trained_policies:
			rt_ids = policy.predict(context, k)
			rt = reward.item_embeddings[rt_ids,:]
			yt = reward.get_reward(context, rt)
			div_inter = reward.get_diversity(rt[yt.flatten()>0,:], context=context, action_ids=rt_ids[yt.flatten()>0])
			div_intra = reward.get_diversity(rt[yt.flatten()>0,:], action_ids=rt_ids[yt.flatten()>0])
			res = results[policy.name]
			#res[t-1,:] = [aggreg_func(yt)*gamma**t, div_intra, div_inter]
			rrt = aggreg_func(yt)
			dia = aggreg_func(div_intra)
			die = aggreg_func(div_inter)
			policy.update(rrt, dia, die)
			reg_reward = aggreg_func(reward.get_means(context, rt))
			res[t-1,:] = [(best_reward - reg_reward)*gamma**t, rrt, best_diversity_intra - dia, best_diversity_inter - die]
			results.update({policy.name: res})
			if (verbose):# or (t%int(horizon//10)==0)):
				print(f"At t={t}, {policy.name} recommends items {rt_ids} to user {context.ravel()} (r={np.round(rrt,3)} (reward={np.round(reg_reward,3)}), dintra={np.round(dia,3)}, dinter={np.round(die,3)})")
				
		if (verbose):
			print("")
		
	return results
	
if __name__=="__main__":
	pass
