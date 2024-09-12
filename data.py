#coding:utf-8

import numpy as np
from sklearn.cluster import KMeans

## MovieLens todo?

## users have an embedding too
## diversity = <item1_emb,item2_emb> > tau
## matching between item and user = <user_emb, item_emb> > tau // theta_user ~ user_emb in LinUCB
def generate_data(nusers, nitems, nratings, ncategories, emb_dim, threshold=0.5, with_visited=-1, booking_proba=lambda p : 0.3):
	item_embeddings = np.random.normal(0, 1, size=(nitems, emb_dim)).reshape((nitems, emb_dim))
	item_embeddings /= np.linalg.norm(item_embeddings)
	item_categories = KMeans(n_clusters=ncategories).fit_predict(item_embeddings)
	user_embeddings = np.random.normal(0, 1, size=(nusers, emb_dim)).reshape((nusers, emb_dim))
	user_embeddings /= np.linalg.norm(user_embeddings)
	ratings = [None]*nratings
	all_pairs = np.array([(u,i) for u in range(nusers) for i in range(nitems)])
	user_item_pairs = all_pairs[np.random.choice(len(all_pairs), size=nratings)].tolist()
	contexts = np.zeros((nusers, ncategories))
	def booking_ui(item_emb, u, context, user_embeddings=user_embeddings, threshold=threshold):
		p = np.clip(user_embeddings[u].T.dot(item_emb)*2000, 0, 1)
		visited = int(np.random.binomial(1, p, size=1)) if (with_visited>0) else -1
		if (with_visited):
			booking = int(np.random.binomial(1, booking_proba(p), size=1))*visited
		else:
			booking = int(np.random.binomial(1, p, size=1))
		return booking, visited
	def delta_ui(item_emb, item_category, context, threshold=threshold):
		cat_i = np.zeros(ncategories)
		cat_i[item_category] = 1
		delta = int((context.dot(cat_i.T)<threshold) and (context[item_category]==0))
		return delta
	def delta_ii(item1_emb, item2_emb, item1_cat, item2_cat):
		return int(item1_emb.T.dot(item2_emb)<threshold and (item1_cat!=item2_cat))
	for nrat, [u, i] in enumerate(user_item_pairs):
		cat_i = np.zeros(ncategories)
		cat_i[item_categories[i]] = 1
		d_ui = delta_ui(item_embeddings[i], item_categories[i], contexts[u])
		b, c = booking_ui(item_embeddings[i], u, contexts[u])
		bin_context = int("".join(map(str, contexts[u].flatten().astype(int).tolist())),2)
		ratings[nrat] = [u, i, bin_context, f"user={u}_item={i}", item_categories[i], d_ui, c, b, int(c>0)+b*(1+d_ui)]
		contexts[u,item_categories[i]] = 1 
	return np.array(ratings), item_embeddings, item_categories, contexts, {"user_embeddings": user_embeddings, "threshold": threshold, "booking_ui": booking_ui, "delta_ui": delta_ui, "delta_ii": delta_ii}
