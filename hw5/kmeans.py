from pylab import *

class KMeans(object):
	"""KMeans"""
	def __init__(self, n_clusters=10, max_iter=50):
		super(KMeans, self).__init__()
		self.n_clusters = n_clusters
		self.max_iter = max_iter

	def fit(self, X):
		pass
