from pylab import *
from sklearn.neighbors import NearestNeighbors

class KMeans(object):
	"""KMeans clustering"""
	def __init__(self, n_clusters=10, max_iter=50):
		super(KMeans, self).__init__()
		self.n_clusters = n_clusters
		self.max_iter = max_iter

	def fit(self, X):
		'''
			Compute k-means clustering.
			X is n_samples x n_features.
		'''
		n_samples, n_features = X.shape
		# Randomly select K samples as initial centers
		perm = permutation(n_samples)[:self.n_clusters]
		C = X[perm]
		# Iterate 50 times
		for i in xrange(50):
			# Assign points to nearest cluster centers
			NN = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(C)
			dist, ind = NN.kneighbors(X)
			ind = ind.flatten()
			# Update the cluster centers
			for k in range(self.n_clusters):
				C[k] = mean(X[ind == k], axis=0)
		return ind

if __name__ == '__main__':
	X = randn(1000,2)
	n_clusters = 10
	kmeans = KMeans(n_clusters)
	ind = kmeans.fit(X)
	colors = rand(n_clusters, 3)[ind, :]
	scatter(X[:, 0], X[:, 1], c=colors, alpha=0.9, s=30)
	show()