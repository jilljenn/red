#coding:utf-8

import matplotlib.pyplot as plt
import umap
import numpy as np
import warnings

from tools import *
from policies import colors

def plot_umap(item_embs, results_traj, k, fontsize=15, n_neighbors=3, fig_title="figure2"):
	dimred_args = dict(n_neighbors=n_neighbors, min_dist=0.5, metric="euclidean", random_state=42)
	with np.errstate(invalid="ignore"): # for NaN or 0 variance matrices
		umap_model = umap.UMAP(**dimred_args)
		with warnings.catch_warnings():
			warnings.simplefilter('ignore', FutureWarning)
			embeddings = umap_model.fit_transform(item_embs)

	for policy_name in results_traj:
		results, contexts = results_traj[policy_name]
		horizon_traj = results.shape[0]

		item_labels = {t: 0.5*np.ones(item_embs.shape[0]) for t in range(horizon_traj)}
		for t in range(horizon_traj):
			item_lbs = item_labels[t]
			X = np.isnan(results[t,:k].ravel())
			if (X.any()):
				kk = min(k,np.min(np.argwhere(X)))
			else:
				kk = k
			item_lbs[results[t,:kk].ravel().astype(int)] = results[t,k:(k+kk)].ravel()
			for tau in range(t):
				X = np.isnan(results[tau,:k].ravel())
				if (X.any()):
					kk = min(k,np.min(np.argwhere(X)))
				else:
					kk = k
				item_lbs[results[tau,:kk].ravel().astype(int)] = 0.26 * np.sign(item_lbs[results[tau,:kk].ravel().astype(int)]-0.5) + 0.25
			assert np.sum(np.vectorize(lambda x : x in [1,-1,0,-0.01,0.51])(item_lbs))==k
			item_labels[t] = item_lbs
			
		fig, axes = plt.subplots(nrows=1, ncols=horizon_traj, figsize=(6.5*horizon_traj,6))
		labels = {-1: "selected/disliked", 1: "selected/liked", 0: "selected/not visited", 0.5: "non selected", 0.25 :"seen", -0.01: "seen/disliked", 0.51: "seen/liked"}
		labels_colors = {-1: "r", 1: "g", 0: "k", 0.5: "b", 0.25: "y", -0.01: "m", 0.51: "c"}
		for t in range(horizon_traj):
			nlab = 0
			for label in list(sorted(list(labels.keys()))):
				embs = embeddings[item_labels[t]==label,:]
				if (embs.shape[0]==0):
					continue
				nlab += 1
				axes[t].scatter(embs[:,0], embs[:,1], s=200, c=labels_colors[label], marker=".", alpha=0.05 if (label == 0.5) else 0.8, label=labels[label]+(f" {embs.shape[0]}" if (label!=0.5) else f" k={k}"))
			axes[t].set_title(f"Round {t+1}: context {pretty_print_context(contexts[t])[:18]}"+("..." if (len(contexts[t])>10) else ""), fontsize=fontsize)
			if (t==0):
				axes[t].set_ylabel("UMAP C2", fontsize=fontsize)
			axes[t].set_xlabel("UMAP C1", fontsize=fontsize)
			axes[t].set_yticks(axes[t].get_yticks())
			axes[t].set_xticks(axes[t].get_xticks())
			axes[t].set_xticklabels(axes[t].get_xticklabels(), fontsize=fontsize)
			axes[t].set_yticklabels(axes[t].get_yticklabels(), fontsize=fontsize)
			#if (t==0):
			#	axes[t].legend(fontsize=fontsize)
			axes[t].legend(fontsize=fontsize, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False)#, ncol=nlab)
		plt.savefig(f"{fig_title}_{policy_name}.png", bbox_inches="tight")
		plt.close()
		
def plot_regret(results_list, policies_names, horizon, fontsize=30, figsize=(40,10), fig_title="figure1"):
	fig, axes = plt.subplots(nrows=1, ncols=4, figsize=figsize)

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
			UB_CI = average.ravel() + std.ravel()
			if (i != 1):
				LB_CI = np.maximum(LB_CI, 0)
			axes[i].fill_between(x.ravel(), LB_CI, UB_CI, alpha=0.2, color=colors[policy_name])
			axes[i].set_xlim((0, horizon))
			if (i!=1):
				axes[i].set_ylim(bottom=0)
			axes[i].set_yticks(axes[i].get_yticks())
			axes[i].set_xticks(axes[i].get_xticks())
			axes[i].set_xticklabels(axes[i].get_xticklabels(), fontsize=fontsize)
			axes[i].set_yticklabels(axes[i].get_yticklabels(), fontsize=fontsize)
			axes[i].set_title({0: "Reward regret", 1: "Reward aggregated", 2: "Diversity intra-batch regret", 3: "Diversity inter-batch regret"}[i], fontsize=fontsize)
			axes[i].set_xlabel("Horizon", fontsize=fontsize)
			axes[i].set_ylabel("", fontsize=fontsize)
	handles, labels = axes[0].get_legend_handles_labels()
	axes[0].legend(handles, labels, fontsize=fontsize, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False)
			
	plt.savefig(f"{fig_title}.png", bbox_inches="tight")
	plt.close()
