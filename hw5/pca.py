from pylab import *
from scipy.sparse.linalg import svds

class PCA(object):
	'''
		PCA modified from ECE661.
	'''
	def __init__(self, K):
		super(PCA, self).__init__()
		self.K = K
		self.WK = None

	def fit(self, X):
		'''
			Fit the model. X is num_samples x num_features.
		'''
		# Follow the notation of Avi's tutorial
		X = X.T
		m = mean(X, axis=1)
		_, _, Ut = svds(dot(X.T, X), k=self.K)
		W = dot(X, Ut.T)
		# Preserve the first K eigenvectors
		# Each column is an eigenvector
		self.WK = W[:,:self.K]
		# Return the eigenvectors
		return self.WK.T

	def transform(self, X):
		'''
			Transform the data. X is num_samples x num_features.
		'''
		X = X.T
		m = mean(X, axis=1)
		# Y is num_features x num_samples
		Y = dot(self.WK.T, X - m.reshape(-1,1))
		return Y.T

	def fit_transform(self, X):
		'''
			Project samples onto the subspace spanned by the
			first K eigenvectors of the covariance matrix.
			Each sample is a row vector (same with sklearn).
			Return num_samples x K.
		'''
		self.fit(X)
		return self.transform(X)
