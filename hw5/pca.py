from pylab import *
from scipy.sparse.linalg import svds
from sklearn.decomposition import PCA as skPCA

class PCA(object):
	'''
		PCA modified from ECE661.
	'''
	def __init__(self, n_components):
		super(PCA, self).__init__()
		self.K = n_components
		self.WK = None

	def fit(self, X):
		'''
			Fit the model. 
			X is n_samples x n_features.
		'''
		n_samples, n_features = X.shape
		# Follow the notation of Avi's tutorial
		# Center the data
		X = X - mean(X, axis=0)
		# X /= std(X, axis=0)
		# Covariance matrix
		C = dot(X.T, X) / (n_samples - 1.)
		_, _, Ut = svd(C)
		# Preserve the first K eigenvectors
		# Each row is an eigenvector
		self.WK = Ut[:self.K]
		return self.WK

	def transform(self, X):
		'''
			Transform the data. 
			X is n_samples x n_features.
		'''
		X = X - mean(X, axis=0)
		# Y is n_features x n_samples
		return dot(X, self.WK.T)

	def fit_transform(self, X):
		'''
			Project samples onto the subspace spanned by the
			first K eigenvectors of the covariance matrix.
			Each sample is a row vector (same with sklearn).
			Return n_samples x K.
		'''
		self.fit(X)
		return self.transform(X)

def main():
	set_printoptions(precision=3, suppress=True)
	X = randn(5, 3)
	pca = PCA(n_components=2)
	print pca.fit_transform(X)
	print pca.fit(X)
	skpca = skPCA(n_components=2)
	print skpca.fit_transform(X)
	print skpca.components_

if __name__ == '__main__':
	main()