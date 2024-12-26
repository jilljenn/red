#coding: utf-8

import numpy as np
from tqdm import tqdm
from time import time
import pickle

from policies import *
from tools import *

def simulate(k, horizon, trained_policies, reward, user_contexts, prob_new_user=0.1, gamma=1., verbose=False, aggreg="sum", savefname="intermediary_simulate.pck"):
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
		for each policy, reward regret, aggregated feedback, intrabatch diversity regret, interbatch diversity regret at each time t
	'''
	if (os.path.exists(savefname)):
		with open(savefname, "rb") as f:
			di = pickle.load(f)
		results = di["results"]
		start_t = di["t"]
	else:
		results = {policy.name: np.zeros((horizon, 4)) for policy in trained_policies}
		start_t = 0
	#results.update({"oracle reward": np.zeros((horizon, 3)), "oracle diversity": np.zeros((horizon, 3))})
	
	aggreg_func = choose_aggregation(aggreg)
	
	for t in tqdm(range(start_t+1,horizon+1), leave=False):
	
		## draw a user context or create a new user
		draw = np.random.choice([0,1], p=[1-prob_new_user, prob_new_user])
		if (draw):
			context = np.zeros((user_contexts.shape[1], 1))
		else:
			i_user = np.random.choice(range(user_contexts.shape[0]))
			context = user_contexts[i_user].reshape(-1, 1)
		
		rt_ids = reward.get_oracle_reward(context, k)
		rt = reward.item_embeddings[rt_ids,:]
		means = reward.get_means(context, rt)
		#div_inter = reward.get_diversity(rt[means.flatten()>0,:], context=context, action_ids=rt_ids[means.flatten()>0])
		#div_intra = reward.get_diversity(rt[means.flatten()>0,:], action_ids=rt_ids[means.flatten()>0])
		div_inter = reward.get_diversity(rt, context=context, action_ids=rt_ids)
		div_intra = reward.get_diversity(rt, action_ids=rt_ids)
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
		div_inter = reward.get_diversity(rt, context=context, action_ids=rt_ids) ## to get positive regret
		best_diversity_inter = aggreg_func(div_inter)
		#res = results["oracle diversity"]
		#res[t-1,:] = [aggreg_func(means), div_intra, div_inter]
		#results.update({"oracle diversity": res})
		if (verbose):# or (t%int(horizon//10)==0)):
			r, d, dd = [np.round(x,3) for x in [aggreg_func(means), aggreg_func(div_intra), aggreg_func(div_inter)]]
			print(f"At t={t}, InterDiversity Oracle recommends items {rt_ids} to user {context.ravel()} (r={r}, dintra={d}, dinter={dd})")
		
		rt_ids = reward.get_oracle_diversity(context, k, aggreg_func, intra=True)
		rt = reward.item_embeddings[rt_ids,:]
		means = reward.get_means(context, rt)	
		#div_intra = reward.get_diversity(rt[means.flatten()>0,:], action_ids=rt_ids[means.flatten()>0])                  ## true
		div_intra = reward.get_diversity(rt, action_ids=rt_ids)                  ## to get positive regret
		best_diversity_intra = aggreg_func(div_intra)
		if (verbose):# or (t%int(horizon//10)==0)):
			r, d, dd = [np.round(x,3) for x in [aggreg_func(means), aggreg_func(div_intra), aggreg_func(div_inter)]]
			print(f"At t={t}, IntraDiversity Oracle recommends items {rt_ids} to user {context.ravel()} (r={r}, dintra={d}, dinter={dd})")
		
		for policy in trained_policies:
			rt_ids = policy.predict(context, k)
			rt = reward.item_embeddings[rt_ids,:]
			yt = reward.get_reward(context, rt)
			#div_inter = reward.get_diversity(rt[yt.flatten()>0,:], context=context, action_ids=rt_ids[yt.flatten()>0])
			#div_intra = reward.get_diversity(rt[yt.flatten()>0,:], action_ids=rt_ids[yt.flatten()>0])
			div_inter = reward.get_diversity(rt, context=context, action_ids=rt_ids)
			div_intra = reward.get_diversity(rt, action_ids=rt_ids)
			res = results[policy.name]
			#res[t-1,:] = [aggreg_func(yt)*gamma**t, div_intra, div_inter]
			rrt = aggreg_func(yt)
			dia = aggreg_func(div_intra)
			die = aggreg_func(div_inter)
			policy.update(context, rt, yt, div_intra, div_inter)
			reg_reward = aggreg_func(reward.get_means(context, rt))
			res[t-1,:] = [(best_reward - reg_reward)*gamma**t, rrt, best_diversity_intra - dia, best_diversity_inter - die]
			results.update({policy.name: res})
			print(f"{policy.name}: ||theta*-theta||_2={float(np.linalg.norm(reward.theta - policy.theta, 2))}") ## TODO
			if (verbose):# or (t%int(horizon//10)==0)):
				print(f"At t={t}, {policy.name} recommends items {rt_ids} to user {context.ravel()} (r={np.round(rrt,3)} (reward={np.round(reg_reward,3)}), dintra={np.round(dia,3)}, dinter={np.round(die,3)})")
				
		if (verbose):
			print("")
			
		with open(savefname, "wb") as f:
			pickle.dump({"results": results, "t": t}, f)
		
	return results
	
def simulate_trajectory(k, horizon, policy, reward, context, gamma=1., verbose=False, aggreg="sum"):
	'''
	Simulate a trajectory from an initial user context
	
	Parameters
	--
	k : int
		the number of recommended items
	horizon : int
		the number of recommendation rounds
	policy : Policy class
		the policy trained on prior ratings
	reward : Reward class
		the environment
	context : array of shape (n_ratings, 1)
		the prior user context (array format)
	gamma : float
		the time discount factor
	verbose : bool
		whether to print info
		
	Returns
	--
	results : arrays of shape (horizon, 2*K)
		the K recommended item identifiers, the K feedback values at time t
	contexts : list of arrays of shape (N, 1)
		the list of contexts at each round
	'''
	results = np.zeros((horizon, 2*k), dtype=int) 
	contexts = [context.copy()]
	aggreg_func = choose_aggregation(aggreg)
	
	for t in tqdm(range(1,horizon+1), leave=False):
		rt_ids = policy.predict(context, k, only_available=False)#True)
		if (rt_ids is None):
			results = results[:(t-1),:]
			return results
		results[t-1,:k] = rt_ids 
		rt = reward.item_embeddings[rt_ids,:]
		yt = reward.get_reward(context, rt)
		results[t-1,k:] = yt
		rrt = aggreg_func(yt)
		
		## update context
		cc = context.ravel().copy()
		context[rt_ids,:] += yt.reshape(-1,1)
		contexts.append(context.copy())
		if (verbose):# or (t%int(horizon//10)==0)):
			print(f"At t={t}, {policy.name} recommends items {rt_ids} to user {pretty_print_context(cc)} -> {pretty_print_context(context.ravel())} (aggregated reward={np.round(rrt,3)})")

	return results, contexts

def single_run(policies, info, ratings, nitems, k, horizon, reward, prob_new_user, gamma=1., verbose=False, random_seed=13234, savefname="intermediary_simulate.pck"):
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
			print(f"{policy_name}: User with initial context {get_context_from_rating(ratings[-1])} ({user_context.ravel()}) recommended items {rt_ids} with scores {np.mean(yt)}")	
		trained_policies += [policy]
		
	user_contexts = np.array([context_int2array(get_context_from_rating(ratings[i]), nitems) for i in range(ratings.shape[0])])
	stime = time()
	results = simulate(k, horizon, trained_policies, reward, user_contexts, prob_new_user=prob_new_user, gamma=gamma, verbose=verbose, savefname=savefname)
	runtime = time()-stime
	return results

def single_trajectory(policy_name, info, ratings, k, horizon, reward, gamma=1., context=None, verbose=False, random_seed=13234):
	seed_everything(int(random_seed))
	nitems = info["item_embeddings"].shape[0]
	if (context is None):
		blank_context = np.zeros((nitems, 1))
		context = blank_context.copy()

	policy = eval(policy_name)(info, random_state=random_seed)
	policy.fit(ratings)
	
	stime = time()
	results, contexts = simulate_trajectory(k, horizon, policy, reward, context=context, gamma=gamma, verbose=verbose)
	runtime = time()-stime
	return results, contexts
	
if __name__=="__main__":
	pass
