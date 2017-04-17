from pylab import *
from sklearn.neighbors import NearestNeighbors

class PCA(object):
	def __init__(self, K):
		super(PCA, self).__init__()
		self.K = K

	def fit_transform(self, X):
		'''
			Project samples onto the subspace spanned by the
			first K eigenvectors of the covariance matrix.
			Each sample is a row vector (same with sklearn).
			Return num_samples x K.
		'''
		# Follow the notation of Avi's tutorial
		X = X.T
		m = mean(X, axis=1)
		_, _, Ut = svd(dot(X.T, X))
		W = dot(X, Ut.T)
		# Preserve the first K eigenvectors
		WK = W[:,:self.K]
		# Project all samples to the K-D subspace
		Y = dot( WK.T, X - m.reshape(-1,1) )
		return Y.T
