import numpy as np

try:
	import torch
except ModuleNotFoundError:
	print('Torch backend unavailable (missing dependencies)')

from scipy.sparse import csr_matrix




MARGIN = 1e-10

def round_prob(C):
	'''
	Infers the most likely clusters from an array of soft cluster
	assignments and returns them as a list of lists.


	Arguments
	---------
	C : array of shape (n_nodes, n_clusters)
		Soft cluster assignments.

	Returns
	-------
	clusters : list
		List of lists representing the inferred clusters.
		The i-th list contains the indices of the nodes belonging to
		the i-th cluster.

	'''

	B = np.argmax(C, 1)
	return [np.where(B==i)[0].tolist() for i in range(C.shape[1])]

def safediv(X, Y):
	'''
	Safe division (avoids division by zero).


	Arguments
	---------
	X : scalar, array or tensor
		Dividend.
	Y : scalar, array or tensor
		Divisor.

	Returns
	-------
	ratio : scalar, array or tensor
		Result of the division.

	'''

	return X/(Y + MARGIN)

def safelog(X, backend='numpy'):
	'''
	Safe logarithm (avoids zero values inside the log).


	Arguments
	---------
	X : scalar, array or tensor
		Argument of the logarithm.
	backend : str, default='numpy'
		Backend to use.
		Should be 'numpy' if X is an array and 'torch' if X is a
		tensor.

	Returns
	-------
	log : scalar, array or tensor
		Logarithm of X.

	'''

	if backend == 'numpy':
		return np.log(np.fmax(X, MARGIN))
	else:
		return torch.log(torch.fmax(X, X.new_full((1,), MARGIN)))

def sparse_block_matrix(C):
	'''
	Transforms a list of clusters into a compressed sparse row matrix
	representing the cluster assignments.


	Arguments
	---------
	C : list
		List of lists representing the clusters.
		The i-th list contains the indices of the nodes belonging to
		the i-th cluster.

	Returns
	-------
	matrix : sparse matrix of shape (n_nodes, n_clusters)
		Binary cluster assignment matrix.

	'''

	n_nodes = sum(len(c) for c in C)
	idx = np.array([
		(i, j) for j, c in enumerate(C) for i in c
	])
	Cm = csr_matrix(
		(
			np.ones(n_nodes),
			(idx[:, 0], idx[:, 1])
		),
		shape=(n_nodes, len(C))
	)
	return Cm