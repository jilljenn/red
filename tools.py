#coding:utf-8

import numpy as np
import torch
import os
import random

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)

def context_array2int(arr, m=1):
	'''
	Converts an array in {-m,-m+1,...,-1,0,1,...,m}^L in a string with 2m << L binary digits
	
	Parameters
	----------
	arr : array of shape (L,1)
		contains integer values in [-m,m]
	m : int
		maximum value in arr
	
	Returns
	-------
	intg : str
		contains 2m binary digits separated by commas
	'''
	bins = [0]*(2*m)
	arr_bin = np.zeros((len(arr), 2*m))
	for i in range(-m, m+1, 1):
		if (i==0):
			continue
		idx_i = i+m-int(i>0)
		arr_bin[arr==i,idx_i] = 1
		bins[idx_i] = int("".join(map(str, arr_bin[:,idx_i].flatten().astype(int).tolist())),2)
	return ",".join(map(str,bins))

def context_int2array(intg, L):
	'''
	Converts a string with 2m binary digits into an array in {-m,-m+1,...,-1,0,1,...,m}^L
	
	Parameters
	----------
	intg : str
		contains 2m binary digits separated by commas
	L : int
		length of array
	
	Returns
	-------
	arr : array of shape (L,1)
		contains integer values in [-m,m]
	'''
	m = len(intg.split(","))//2
	arr_bin = list(map(lambda x : list(map(int,list(str(bin(int(x)))[2:]))),intg.split(",")))
	arr_bin = np.array([[0]*(L-len(x))+x for x in arr_bin]).T
	values = np.array([i for i in range(-m, m+1, 1) if (i != 0)])
	arr = np.array([0 if (arr_bin[i].sum()==0) else values[np.argmax(arr_bin[i])] for i in range(len(arr_bin))])
	return arr
	
def test_int2array(m, L):
	arr = np.random.choice(range(m+1), size=L)
	intg = context_array2int(arr, m=m)
	arr2 = context_int2array(intg, L)
	return (arr==arr2).all()
