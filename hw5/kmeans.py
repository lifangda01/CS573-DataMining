from pylab import *
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, mutual_info_score

class KMeans(object):
	"""KMeans clustering"""
	def __init__(self, n_clusters=10, max_iter=50):
		super(KMeans, self).__init__()
		self.n_clusters = n_clusters
		self.max_iter = max_iter
		self.WC_SSD = -1.
		self.SC = -1.
		self.NMI = -1.

	def fit(self, X, y=None):
		'''
			Compute k-means clustering.
			X is n_samples x n_features.
			y is n_samples.
		'''
		n_samples, n_features = X.shape
		# Randomly select K samples as initial centers
		perm = permutation(n_samples)[:self.n_clusters]
		C = X[perm]
		# Nearest neighbor function
		nn = lambda x: argmin(norm(C - x, axis=1))
		# Iterate 50 times
		for i in xrange(50):
			# Assign points to nearest cluster centers
			# NN = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(C)
			# dist, ind = NN.kneighbors(X)
			# ind = ind.flatten()
			ind = apply_along_axis(nn, 1, X)
			# Update the cluster centers
			for k in range(self.n_clusters):
				C[k] = mean(X[ind == k], axis=0)
		# Compute the WC_SSD
		self.WC_SSD = sum((X - C[ind])**2)
		# Compute the SC
		def sc(x):
			'''
				Compute SC for a single sample.
			'''
			dist = norm(X - x[:-1], axis=1)
			# x[-1] is encoded as the ind of x
			wc = mean(dist[ind == x[-1]])
			bc = mean(dist[ind != x[-1]])
			return (bc - wc) / max(bc, wc)
		# Encode ind along with X
		X_enc = hstack((X, ind.reshape(-1, 1)))
		self.SC = mean(apply_along_axis(sc, 1, X_enc))
		# Compute the NMI
		if isinstance(y, ndarray):
			pc = unique(y, return_counts=True)[1]
			pc = pc * 1.0 / sum(pc)
			logpc = log(pc)
			n_classes = size(pc)
			pg = unique(ind, return_counts=True)[1]
			pg = pg * 1.0 / sum(pg)
			logpg = log(pg)
			# Compute the contingency table
			# pcg is n_clusters x n_classes
			pcg = bincount(self.n_clusters * ind + y, 
				minlength=self.n_clusters * n_classes).reshape(n_clusters, n_classes)
			pcg = pcg * 1.0 / sum(pcg, axis=0)
			# nomin = sum(pcg * log(pcg)) - sum(dot(pcg, logpc)) - sum(dot(logpg, pcg))
			nomin = sum(pcg * log(pcg / pc / pg[:, None]))
			denom = - dot(pc, logpc) - dot(pg, logpg)
			print pcg
			print nomin, denom
			self.NMI = nomin / denom
			print "sklearn NMI, NMI =", mutual_info_score(y, ind), nomin
		return ind

	def get_evals(self):
		'''
			Return WC_SSD, SC, NMI.
		'''
		return self.WC_SSD, self.SC, self.NMI

if __name__ == '__main__':
	n_clusters = 10
	X = randn(1000,2)
	y = randint(n_clusters, size=1000)
	kmeans = KMeans(n_clusters)
	ind = kmeans.fit(X, y)
	print "WC_SSD, SC, NMI =", kmeans.get_evals()
	print "sklearn SC =", silhouette_score(X, ind)
	colors = rand(n_clusters, 3)[ind, :]
	scatter(X[:, 0], X[:, 1], c=colors, alpha=0.9, s=30)
	show()