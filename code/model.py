import time

import numpy as np

try:
	import torch
	from torch_sparse import transpose, spmm
except ModuleNotFoundError:
	print('Torch backend unavailable (missing dependencies)')

from scipy.sparse import csr_matrix
from scipy.stats import entropy

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state

from joblib import Parallel, delayed

from utils import safediv, safelog, round_prob, sparse_block_matrix




class MultilayerPoissonLBM(BaseEstimator):
	'''
	Estimator for the multilayer Poisson Latent Block Model with
	NumPy backend.
	The inference procedure is adapted from the variational EM
	algorithm introduced in [1].


	Parameters
	----------
	H : int, default=3
		Number of top clusters.
	K : int, default=3
		Number of bottom clusters.
	epsilon : float, default=1e-7
		Stopping criterion for the inference procedure.
		The procedure keeps going as long as the relative variation of
		the fuzzy log-likelihood after each iteration is greater than
		epsilon.
	max_iter : int, default=100
		Maximum number of iterations in the inference procedure.
	runs : int, default=20
		Number of distinct runs of the inference procedure.
		The best model across all runs is selected.
	verbose : int, default=0
		Level of verbosity.
		If verbose == 0, no message is displayed.
		If verbose >= 1, a message is displayed at the start and at
		the end of each run of the inference procedure.
		If verbose > 1, a message is displayed after each iteration of
		the inference procedure.
	random_state : int, default=None
		Seed for the random number generator.

	Attributes
	----------
	rnd : object
		Random number generator.
	I : int
		Number of top nodes.
		This attribute is set when fitting the model.
		It is inferred from the data.
	J : int
		Number of bottom nodes.
		This attribute is set when fitting the model.
		It is inferred from the data.
	L : int
		Number of edge types.
		This attribute is set when fitting the model.
		It is inferred from the data.
	top : LabelEncoder
		Label encoder for the top nodes.
		This attribute is set when fitting the model.
	bottom : LabelEncoder
		Label encoder for the bottom nodes.
		This attribute is set when fitting the model.
	types : LabelEncoder
		Label encoder for the edge types.
		This attribute is set when fitting the model.
	G : list
		List of sparse row matrices representing the layers of the
		multiplex graph.
		This attribute is set when fitting the model.
		It is inferred from the data.
	U : array of shape (self.I, self.H)
		Array containing the fuzzy cluster assignments of the top
		nodes.
		This attribute is set when fitting the model.
	V : array of shape (self.J, self.K)
		Array containing the fuzzy cluster assignments of the bottom
		nodes.
		This attribute is set when fitting the model.
	mu : array of shape (self.I,)
		Array containing the marginal rates of the top nodes.
		This attribute is set when fitting the model.
	nu : array of shape (self.J,)
		Array containing the marginal rates of the bottom nodes.
		This attribute is set when fitting the model.
	theta : array of shape (self.L, self.H, self.K)
		Array containing the rate of each (edge type, top cluster,
		bottom cluster) triple.
		This attribute is set when fitting the model.
	pi : array of shape (self.H,)
		Array containing the probability of a top node being assigned
		to each top cluster.
		This attribute is set when fitting the model.
	rho : array of shape (self.K,)
		Array containing the probability of a bottom node being
		assigned to each bottom cluster.
		This attribute is set when fitting the model.

	References
	----------
	[1] Govaert, Gerard and Nadif, Mohamed. Latent block model for
		contingency table. In Commun. Stat. Theory Methods 39(3), 2010.

	'''

	def __init__(
		self,
		H=3,
		K=3,
		epsilon=1e-7,
		max_iter=100,
		runs=20,
		verbose=0,
		random_state=None
		):

		self.H = H
		self.K = K
		self.epsilon = epsilon
		self.max_iter = max_iter
		self.runs = runs
		self.verbose = verbose
		self.random_state = random_state


	def fit(self, X, y=None):
		'''
		Fit the estimator.


		Arguments
		---------
		X : array of shape (n_edges, 3)
			Array of typed edges.
			Each row contains three values: top node, bottom node, and
			edge type.
			These values can be strings or integers.
			They are encoded using a LabelEncoder.
		y : not used, included for consistency with the scikit-learn
			API.

		Returns
		-------
		self : object
			Fitted estimator.

		'''

		self._make_graph(X)
		self.rnd = check_random_state(self.random_state)
		params = [self._fit_params() for i in range(self.runs)]
		params.sort(key=lambda x: x[-1])
		self.U, self.V, self.theta, self.mu, self.nu, _ = params[-1]
		self.pi = self.U.mean(0)
		self.rho = self.V.mean(0)
		return self


	def get_block_params(self):
		'''
		Returns top and bottom clusters inferred from the current soft
		assignments, as well as the current rate matrices.


		Returns
		-------
		U_dict : list
			List of dictionaries corresponding to the top clusters.
			The keys of each dictionary are the indices of the nodes
			within it, and the corresponding values are their
			probabilities of belonging to this cluster.
		V_dict : list
			List of dictionaries corresponding to the bottom clusters.
			The keys of each dictionary are the indices of the nodes
			within it, and the corresponding values are their
			probabilities of belonging to this cluster.
		theta : array of shape (self.L, self.H, self.K)
			Array containing the rate of each (edge_type, top cluster,
			bottom cluster) triple.

		'''

		U, V = round_prob(self.U), round_prob(self.V)
		theta = self.theta
		Up = [self.U[u, i] for i, u in enumerate(U)]
		Vp = [self.V[v, j] for j, v in enumerate(V)]
		U_dict = [dict(zip(u, up)) for u, up in zip(U, Up)]
		V_dict = [dict(zip(v, vp)) for v, vp in zip(V, Vp)]
		return U_dict, V_dict, theta


	def get_results(self):
		'''
		Returns top and bottom clusters inferred from the current soft
		assignments, as well as the current rate matrices, as
		dictionaries containing the original names of the nodes and
		edge types.


		Returns
		-------
		top : list
			List of dictionaries corresponding to the top clusters.
			The keys of each dictionary are the names of the nodes
			within it, and the corresponding values are their
			probabilities of belonging to this cluster.
		bottom : list
			List of dictionaries corresponding to the bottom clusters.
			The keys of each dictionary are the names of the nodes
			within it, and the corresponding values are their
			probabilities of belonging to this cluster.
		thetas : dict
			Dictionary containing the rate matrices for each edge type.
			The keys are the names of the edge types, and the
			corresponding values are the rate matrices in list-of-lists
			format.

		'''

		U, V, T = self.get_block_params()
		top = []
		to_del = ([], [])
		for i, u in enumerate(U):
			keys = sorted(list(u.keys()))
			if len(keys) > 0:
				top.append(dict(zip(
					self.top.inverse_transform(keys),
					[u[k] for k in keys]
				)))
			else:
				to_del[0].append(i)
		bottom = []
		for j, v in enumerate(V):
			keys = sorted(list(v.keys()))
			if len(keys) > 0:
				bottom.append(dict(zip(
					self.bottom.inverse_transform(keys),
					[v[k] for k in keys]
				)))
			else:
				to_del[1].append(j)
		T = np.delete(
			np.delete(
				T, to_del[0], axis=1
			),
			to_del[1],
			axis=2
		)
		thetas = dict([
			(t, T[i, :, :].tolist())
			for i, t in zip(
				self.types.transform(self.types.classes_),
				self.types.classes_
			)
		])
		return top, bottom, thetas


	def icl(self):
		'''
		Returns the logarithm of the integrated completed likelihood
		(ICL) of the current model on the dataset stored in self.G.
		See [1] for a definition of the ICL.


		Returns
		-------
		icl : float
			Integrated completed log-likelihood (the higher, the
			better).

		References
		----------
		[1] Biernacki, Christophe, et al. Assessing a mixture model for
			clustering with the integrated completed likelihood.
			In IEEE Trans. Pattern Anal. Mach. Intell. 22(7), 2000.

		'''

		res = 2*self._log_likelihood(fuzzy=False)
		res -= (self.H - 1) * np.log(self.I)
		res -= (self.K - 1) * np.log(self.J)
		res -= (self.L * self.H * self.K) * np.log(
			self.L * self.I * self.J
		)
		return .5*res


	def _build_graph(self, X):
		'''
		Builds a sparse matrix-based representation of the input data
		and stores it into self.G.


		Arguments
		---------
		X : array of shape (n_edges, 3)
			Array of typed edges.
			Each row contains three values: top node, bottom node, and
			edge type.
			These values can be strings or integers.

		'''

		self.G = []
		for l in range(self.L):
			t = self.types.inverse_transform([l])[0]
			Xt = X[X[:, 2] == t]
			top = self.top.transform(Xt[:, 0])
			bottom = self.bottom.transform(Xt[:, 1])
			g = csr_matrix(
				(
					np.ones(Xt.shape[0]),
					(top, bottom)
				),
				shape=(self.I, self.J)
			)
			self.G.append(g)


	def _copy_params(self):
		'''
		Returns a copy of the current parameters (soft cluster
		assignments, rate matrices, and marginal rates of the nodes).


		Returns
		-------
		U : array of shape (self.I, self.H)
			Copy of self.U.
		V : array of shape (self.J, self.K)
			Copy of self.V.
		theta : array of shape (self.L, self.H, self.K)
			Copy of self.theta.
		mu : array of shape (self.I,)
			Copy of self.mu.
		nu : array of shape (self.J,)
			Copy of self.nu.

		'''

		res = (
			np.array(self.U),
			np.array(self.V),
			np.array(self.theta),
			np.array(self.mu),
			np.array(self.nu)
		)
		return res


	def _estimate_clusters(self, top):
		'''
		E-step of the inference procedure.
		Estimates the soft cluster assignments of the top or bottom
		nodes given the current parameters and the data stored in
		self.G.
		The result is returned as an array of shape
		(n_nodes, n_clusters).


		Arguments
		---------
		top : bool
			If true, soft cluster assignments are evaluated for the
			top nodes.
			Otherwise, the assignments of the bottom nodes are
			evaluated.

		Returns
		-------
		assignments : array of shape (n_nodes, n_clusters)
			Soft cluster assignments of the top nodes if top is true,
			and of the bottom nodes otherwise.

		'''

		if top:
			Gp = np.stack([g.dot(self.V) for g in self.G])
			X1 = (
				Gp[:, :, np.newaxis, :] * safelog(
					self.theta[:, np.newaxis, :, :]
				)
			).sum((0, 3))
		else:
			Gp = np.stack([g.T.dot(self.U).T for g in self.G])
			X1 = (
				Gp[:, :, :, np.newaxis] * safelog(
					self.theta[:, :, np.newaxis, :]
				)
			).sum((0, 1))
		theta = self.theta
		if not top:
			theta = theta.transpose(0, 2, 1)
		Y = self.V if top else self.U
		m = self.mu if top else self.nu
		n = self.nu if top else self.mu
		X2 = -m[:, np.newaxis] * (
			np.matmul(
				theta,
				np.matmul(
					Y.T,
					n[:, np.newaxis]
				)[np.newaxis, :, :]
			)[:, np.newaxis, :, 0]
		).sum(0)
		p = self.pi if top else self.rho
		X = X1 + X2 + safelog(p, backend='numpy')
		X = np.exp(X - np.amax(X, 1)[:, np.newaxis])
		return X/X.sum(1)[:, np.newaxis]


	def _estimate_theta(self):
		'''
		M-step of the inference procedure.
		Estimates the rate matrices given the soft cluster assignments
		and the data stored in self.G.
		The result is returned as an array of shape
		(n_layers, n_top_clusters, n_bottom_clusters).


		Returns
		-------
		theta : array of shape (self.L, self.H, self.K)
			Stacked rate matrices (one per edge type).

		'''

		M = np.stack([self.U.T.dot(g.dot(self.V)) for g in self.G])
		P = np.tile(
			np.matmul(
				self.mu.dot(self.U)[:, np.newaxis],
				self.nu.dot(self.V)[np.newaxis, :]
			),
			(self.L, 1, 1)
		)
		return safediv(M, P)


	def _fit_params(self):
		'''
		Runs the inference procedure on the data stored in self.G and
		returns the obtained parameters.


		Returns
		-------
		U : array of shape (self.I, self.H)
			Inferred soft cluster assignments of the top nodes.
		V : array of shape (self.J, self.K)
			Inferred soft cluster assignments of the bottom nodes.
		theta : array of shape (self.L, self.H, self.K)
			Inferred rate matrices.
		mu : array of shape (self.I,)
			Inferred marginal rates of the top nodes.
		nu : array of shape (self.J,)
			Inferred marginal rates of the bottom nodes.
		score : float
			Complete data log-likelihood of the inferred parameters.

		'''

		start_time = time.time()
		self._initialize()
		old_score = self._log_likelihood()
		diff = 10
		n_iter = 0
		if self.verbose > 0:
			print('[*] Starting inference (H=%d, K=%d)' % (self.H, self.K))
		while diff > self.epsilon and n_iter < self.max_iter:
			self.U = self._estimate_clusters(True)
			self.pi = self.U.mean(0)

			self.V = self._estimate_clusters(False)
			self.rho = self.V.mean(0)

			self.theta = self._estimate_theta()

			score = self._log_likelihood()
			diff = np.abs(1-safediv(score, old_score))
			old_score = score
			n_iter += 1
			if self.verbose > 1:
				print(
					'\tIteration %d; Log-likelihood: %f' % (n_iter, score)
				)

		end_time = time.time()
		if self.verbose > 0:
			diff = end_time - start_time
			minutes = int(diff)//60
			seconds = diff % 60
			print((
				'[*] Reached convergence after %d iterations '
				'(%d min %d sec); '
				'Log-likelihood: %f\n'
				) % (
					n_iter, minutes, seconds, score
				)
			)

		score = self._log_likelihood(fuzzy=False)
		res = self._copy_params()
		return (*res, score)


	def _initialize(self):
		'''
		Initializes the parameters of the model.

		'''

		U = self.rnd.uniform(size=(self.I, self.H))
		self.U = U/U.sum(1)[:, np.newaxis]
		V = self.rnd.uniform(size=(self.J, self.K))
		self.V = V/V.sum(1)[:, np.newaxis]
		tot = np.sqrt(sum(g.sum() for g in self.G))
		self.mu = sum(
			np.array(g.sum(1))[:, 0]
			for g in self.G
		)/tot
		self.nu = sum(
			np.array(g.sum(0))[0, :]
			for g in self.G
		)/tot
		self.theta = self.rnd.uniform(size=(self.L, self.H, self.K))
		self.pi = self.rnd.uniform(size=(self.H,))
		self.pi /= self.pi.sum()
		self.rho = self.rnd.uniform(size=(self.K,))
		self.rho /= self.rho.sum()


	def _log_likelihood(self, fuzzy=True):
		'''
		Computes the (exact or fuzzy) complete data log-likelihood of
		the current model for the dataset stored in self.G.


		Arguments
		---------
		fuzzy : bool
			If true, the fuzzy criterion introduced in [1] is computed.
			Otherwise, the exact complete data log-likelihood (with
			hard cluster assignments) is returned.

		Returns
		-------
		score : float
			Exact or fuzzy log-likelihood of the current model.

		References
		----------
		[1] Govaert, Gerard and Nadif, Mohamed. Latent block model for
			contingency table. In Commun. Stat. Theory Methods 39(3), 2010.

		'''

		if not fuzzy:
			U = sparse_block_matrix(round_prob(self.U)).toarray()
			V = sparse_block_matrix(round_prob(self.V)).toarray()
			res = 0
		else:
			U, V = self.U, self.V
			res = entropy(U, axis=1).sum() + entropy(V, axis=1).sum()
		res += U.dot(safelog(self.pi)).sum()
		res += V.dot(safelog(self.rho)).sum()
		res += sum(
			(
				U.T.dot(
					g.dot(V)) * safelog(
					self.theta[i, :, :]
				)
				- U.T.dot(self.mu)[:, np.newaxis].dot(
					V.T.dot(self.nu)[np.newaxis, :]
				) * self.theta[i, :, :]
			).sum()
			for i, g in enumerate(self.G)
		)
		if not fuzzy:
			res += sum(
				g.T.dot(safelog(self.mu)).sum()
				+ g.dot(safelog(self.nu)).sum()
				for g in self.G
			)

		return res


	def _make_graph(self, X):
		'''
		Builds the label encoders for top nodes, bottom nodes and edge
		types, then builds the multiplex graph representing the input
		dataset and stores it into self.G.


		Arguments
		---------
		X : array of shape (n_edges, 3)
			Array of typed edges.
			Each row contains three values: top node, bottom node, and
			edge type.
			These values can be strings or integers.

		'''

		encoders = [LabelEncoder() for i in range(3)]
		for i, e in enumerate(encoders):
			e.fit(X[:, i])
		self.top, self.bottom, self.types = encoders
		self.I = len(self.top.classes_)
		self.J = len(self.bottom.classes_)
		self.L = len(self.types.classes_)
		self._build_graph(X)


class TorchMultilayerPoissonLBM(MultilayerPoissonLBM):
	'''
	Estimator for the multilayer Poisson Latent Block Model with
	PyTorch backend.
	The inference procedure is adapted from the variational EM
	algorithm introduced in [1].


	Parameters
	----------
	H : int, default=3
		Number of top clusters.
	K : int, default=3
		Number of bottom clusters.
	epsilon : float, default=1e-7
		Stopping criterion for the inference procedure.
		The procedure keeps going as long as the relative variation of
		the fuzzy log-likelihood after each iteration is greater than
		epsilon.
	max_iter : int, default=100
		Maximum number of iterations in the inference procedure.
	runs : int, default=20
		Number of distinct runs of the inference procedure.
		The best model across all runs is selected.
	verbose : int, default=0
		Level of verbosity.
		If verbose == 0, no message is displayed.
		If verbose >= 1, a message is displayed at the start and at
		the end of each run of the inference procedure.
		If verbose > 1, a message is displayed after each iteration of
		the inference procedure.
	device : str, default='cuda'
		Identifier of the device used by PyTorch.
	random_state : int, default=None
		Seed for the random number generator.

	Attributes
	----------
	rnd : object
		Random number generator.
	I : int
		Number of top nodes.
		This attribute is set when fitting the model.
		It is inferred from the data.
	J : int
		Number of bottom nodes.
		This attribute is set when fitting the model.
		It is inferred from the data.
	L : int
		Number of edge types.
		This attribute is set when fitting the model.
		It is inferred from the data.
	top : LabelEncoder
		Label encoder for the top nodes.
		This attribute is set when fitting the model.
	bottom : LabelEncoder
		Label encoder for the bottom nodes.
		This attribute is set when fitting the model.
	types : LabelEncoder
		Label encoder for the edge types.
		This attribute is set when fitting the model.
	G : list
		List of sparse tensors representing the layers of the
		multiplex graph.
		This attribute is set when fitting the model.
		It is inferred from the data.
	U : tensor of shape (self.I, self.H)
		Tensor containing the fuzzy cluster assignments of the top
		nodes.
		This attribute is set when fitting the model.
	V : tensor of shape (self.J, self.K)
		Tensor containing the fuzzy cluster assignments of the bottom
		nodes.
		This attribute is set when fitting the model.
	mu : tensor of shape (self.I,)
		Tensor containing the marginal rates of the top nodes.
		This attribute is set when fitting the model.
	nu : tensor of shape (self.J,)
		Tensor containing the marginal rates of the bottom nodes.
		This attribute is set when fitting the model.
	theta : tensor of shape (self.L, self.H, self.K)
		Tensor containing the rate of each (edge type, top cluster,
		bottom cluster) triple.
		This attribute is set when fitting the model.
	pi : tensor of shape (self.H,)
		Tensor containing the probability of a top node being assigned
		to each top cluster.
		This attribute is set when fitting the model.
	rho : tensor of shape (self.K,)
		Tensor containing the probability of a bottom node being
		assigned to each bottom cluster.
		This attribute is set when fitting the model.

	References
	----------
	[1] Govaert, Gerard and Nadif, Mohamed. Latent block model for
		contingency table. In Commun. Stat. Theory Methods 39(3), 2010.

	'''

	def __init__(
		self,
		H=3,
		K=3,
		epsilon=1e-7,
		max_iter=100,
		runs=20,
		verbose=0,
		device='cuda',
		random_state=None
		):

		super(TorchMultilayerPoissonLBM, self).__init__(
			H=H,
			K=K,
			epsilon=epsilon,
			max_iter=max_iter,
			runs=runs,
			verbose=verbose,
			random_state=random_state
		)
		self.device = torch.device(device)


	def get_block_params(self):
		'''
		Returns top and bottom clusters inferred from the current soft
		assignments, as well as the current rate matrices.


		Returns
		-------
		U_dict : list
			List of dictionaries corresponding to the top clusters.
			The keys of each dictionary are the indices of the nodes
			within it, and the corresponding values are their
			probabilities of belonging to this cluster.
		V_dict : list
			List of dictionaries corresponding to the bottom clusters.
			The keys of each dictionary are the indices of the nodes
			within it, and the corresponding values are their
			probabilities of belonging to this cluster.
		theta : array of shape (self.L, self.H, self.K)
			Array containing the rate of each (edge_type, top cluster,
			bottom cluster) triple.

		'''

		U = round_prob(self.U.cpu().numpy())
		V = round_prob(self.V.cpu().numpy())
		theta = self.theta.cpu().numpy()
		Up = [
			self.U[u, i].cpu().numpy().astype(float)
			for i, u in enumerate(U)
		]
		Vp = [
			self.V[v, j].cpu().numpy().astype(float)
			for j, v in enumerate(V)
		]
		U_dict = [dict(zip(u, up)) for u, up in zip(U, Up)]
		V_dict = [dict(zip(v, vp)) for v, vp in zip(V, Vp)]
		return U_dict, V_dict, theta


	def _build_graph(self, X):
		'''
		Builds a sparse tensor-based representation of the input data
		and stores it into self.G.


		Arguments
		---------
		X : array of shape (n_edges, 3)
			Array of typed edges.
			Each row contains three values: top node, bottom node, and
			edge type.
			These values can be strings or integers.

		'''

		self.G = []
		for l in range(self.L):
			t = self.types.inverse_transform([l])[0]
			Xt = X[X[:, 2] == t]
			top = self.top.transform(Xt[:, 0])
			bottom = self.bottom.transform(Xt[:, 1])
			index = torch.from_numpy(
				np.stack([top, bottom], axis=1)
			).T.to(self.device)
			values = torch.ones(Xt.shape[0]).to(self.device)
			g = (index, values)
			self.G.append(g)


	def _copy_params(self):
		'''
		Returns a copy of the current parameters (soft cluster
		assignments, rate matrices, and marginal rates of the nodes).


		Returns
		-------
		U : tensor of shape (self.I, self.H)
			Copy of self.U.
		V : tensor of shape (self.J, self.K)
			Copy of self.V.
		theta : tensor of shape (self.L, self.H, self.K)
			Copy of self.theta.
		mu : tensor of shape (self.I,)
			Copy of self.mu.
		nu : tensor of shape (self.J,)
			Copy of self.nu.

		'''

		res = (
			self.U.clone(),
			self.V.clone(),
			self.theta.clone(),
			self.mu.clone(),
			self.nu.clone()
		)
		return res


	def _estimate_clusters(self, top):
		'''
		E-step of the inference procedure.
		Estimates the soft cluster assignments of the top or bottom
		nodes given the current parameters and the data stored in
		self.G.
		The result is returned as a tensor of shape
		(n_nodes, n_clusters).


		Arguments
		---------
		top : bool
			If true, soft cluster assignments are evaluated for the
			top nodes.
			Otherwise, the assignments of the bottom nodes are
			evaluated.

		Returns
		-------
		assignments : tensor of shape (n_nodes, n_clusters)
			Soft cluster assignments of the top nodes if top is true,
			and of the bottom nodes otherwise.

		'''

		if top:
			X1 = sum(
				(
					spmm(
						g[0],
						g[1],
						self.I,
						self.J,
						self.V
					).unsqueeze(1)
					* safelog(
						self.theta[l, :, :],
						backend='torch'
					).unsqueeze(0)
				).sum(2)
				for l, g in enumerate(self.G)
			)
		else:
			X1 = sum(
				(
					spmm(
						*transpose(g[0], g[1], self.I, self.J),
						self.J,
						self.I,
						self.U
					).T.unsqueeze(2)
					* safelog(
						self.theta[l, :, :],
						backend='torch'
					).unsqueeze(1)
				).sum(0)
				for l, g in enumerate(self.G)
			)
		theta = self.theta
		if not top:
			theta = theta.transpose(2, 1)
		Y = self.V if top else self.U
		m = self.mu if top else self.nu
		n = self.nu if top else self.mu
		X2 = -m.unsqueeze(1) * (
			torch.matmul(
				theta,
				torch.matmul(
					Y.T,
					n.unsqueeze(1)
				).unsqueeze(0)
			).unsqueeze(1).squeeze(3)
		).sum(0)
		p = self.pi if top else self.rho
		X = X1 + X2 + safelog(p, backend='torch')
		X = torch.exp(X - torch.amax(X, 1).unsqueeze(1))
		return X/X.sum(1).unsqueeze(1)


	def _estimate_theta(self):
		'''
		M-step of the inference procedure.
		Estimates the rate matrices given the soft cluster assignments
		and the data stored in self.G.
		The result is returned as a tensor of shape
		(n_layers, n_top_clusters, n_bottom_clusters).


		Returns
		-------
		theta : tensor of shape (self.L, self.H, self.K)
			Stacked rate matrices (one per edge type).

		'''

		M = torch.stack([
			self.U.T.matmul(
				spmm(g[0], g[1], self.I, self.J, self.V)
			)
			for g in self.G
		])
		P = torch.matmul(
			torch.matmul(self.mu, self.U).unsqueeze(1),
			torch.matmul(self.nu, self.V).unsqueeze(0)
		).repeat((self.L, 1, 1))
		return safediv(M, P)


	def _initialize(self):
		'''
		Initializes the parameters of the model.

		'''

		U = self.rnd.uniform(size=(self.I, self.H))
		self.U = U/U.sum(1)[:, np.newaxis]
		V = self.rnd.uniform(size=(self.J, self.K))
		self.V = V/V.sum(1)[:, np.newaxis]
		tot = torch.sqrt(sum(g[1].sum() for g in self.G))
		self.mu = sum(
			spmm(
				g[0], g[1], self.I, self.J,
				g[0].new_ones(self.J).unsqueeze(1)
			).squeeze(1)
			for g in self.G
		)/tot
		self.nu = sum(
			spmm(
				*transpose(g[0], g[1], self.I, self.J),
				self.J, self.I,
				g[0].new_ones(self.I).unsqueeze(1)
			).squeeze(1)
			for g in self.G
		)/tot
		self.theta = self.rnd.uniform(size=(self.L, self.H, self.K))
		self.pi = self.rnd.uniform(size=(self.H,))
		self.pi /= self.pi.sum()
		self.rho = self.rnd.uniform(size=(self.K,))
		self.rho /= self.rho.sum()

		def make_params(params):
			res = [
				torch.from_numpy(
					p.astype(np.float32)
				).to(self.device)
				for p in params
			]
			return res
		self.U, self.V, self.theta, self.pi, self.rho = make_params([
			self.U, self.V, self.theta,
			self.pi, self.rho
		])


	def _log_likelihood(self, fuzzy=True):
		'''
		Computes the (exact or fuzzy) complete data log-likelihood of
		the current model for the dataset stored in self.G.


		Arguments
		---------
		fuzzy : bool
			If true, the fuzzy criterion introduced in [1] is computed.
			Otherwise, the exact complete data log-likelihood (with
			hard cluster assignments) is returned.

		Returns
		-------
		score : float
			Exact or fuzzy log-likelihood of the current model.

		References
		----------
		[1] Govaert, Gerard and Nadif, Mohamed. Latent block model for
			contingency table. In Commun. Stat. Theory Methods 39(3), 2010.

		'''

		if not fuzzy:
			U = torch.from_numpy(
				sparse_block_matrix(
					round_prob(self.U.cpu())
				).toarray().astype(np.float32)
			).to(self.device)
			V = torch.from_numpy(
				sparse_block_matrix(
					round_prob(self.V.cpu())
				).toarray().astype(np.float32)
			).to(self.device)
			res = 0
		else:
			U, V = self.U, self.V
			res = entropy(U.cpu().numpy(), axis=1).sum()
			res += entropy(V.cpu().numpy(), axis=1).sum()
		Gp = [
			spmm(g[0], g[1], self.I, self.J, V)
			for g in self.G
		]
		res += sum(
			(
				U.T.matmul(g) * safelog(
					self.theta[i, :, :],
					backend='torch'
				)
				- U.T.matmul(self.mu).unsqueeze(1).matmul(
					V.T.matmul(self.nu).unsqueeze(0)
				) * self.theta[i, :, :]
			).sum()
			for i, g in enumerate(Gp)
		)
		if not fuzzy:
			for g in self.G:
				res += spmm(
					g[0], g[1], self.I, self.J,
					safelog(
						self.nu,
						backend='torch'
					).unsqueeze(1)
				).sum()
				index, value = transpose(g[0], g[1], self.I, self.J)
				res += spmm(
					index, value, self.J, self.I,
					safelog(
						self.mu,
						backend='torch'
					).unsqueeze(1)
				).sum()
		return res.data.item()


def fit_mlplbm(
	X,
	H=(3, 4, 5, 6),
	K=(3, 4, 5, 6),
	epsilon=1e-7,
	max_iter=100,
	runs=20,
	n_jobs=1,
	verbose=0,
	backend='numpy',
	device=['cuda'],
	random_state=None
	):

	def fit_model(params, device):
		if backend == 'numpy':
			return [
				MultilayerPoissonLBM(
					h,
					k,
					epsilon=epsilon,
					max_iter=max_iter,
					runs=runs,
					verbose=verbose,
					random_state=random_state
				).fit(X)
				for h, k in params
			]
		else:
			return [
				TorchMultilayerPoissonLBM(
					h,
					k,
					epsilon=epsilon,
					max_iter=max_iter,
					runs=runs,
					verbose=verbose,
					device=device,
					random_state=random_state
				).fit(X)
				for h, k in params
			]

	params = [(h, k) for h in H for k in K]
	chunks = [[] for i in range(n_jobs)]
	for i, p in enumerate(params):
		chunks[i%n_jobs].append(p)

	res = Parallel(n_jobs=n_jobs)(
		delayed(fit_model)(chunks[i], device[i%len(device)])
		for i in range(n_jobs)
	)
	models = [m for r in res for m in r]
	crit = [m.icl() for m in models]
	best = [m for i, m in enumerate(models) if crit[i] == max(crit)]
	best.sort(key=lambda m: m.H + m.K)

	return best[0]
