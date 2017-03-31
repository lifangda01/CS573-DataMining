from pylab import *
from preprocess import training_preprocess_from_csv, testing_preprocess_from_csv

_gini = lambda y: 1 - sum(( 1.0*unique(y, return_counts=True)[1] / size(y) )**2)

class _Node(object):
	"""Container for a node in DecisionTree"""
	def __init__(self, feature_index=None, label=None):
		super(_Node, self).__init__()
		self.left = None
		self.right = None
		self.label = label
		self.feature_index = feature_index

class DecisionTree(object):
	"""DecisionTree model"""
	def __init__(self, depth_limit=10, sample_limit=10):
		super(DecisionTree, self).__init__()
		self.depth_limit = depth_limit
		self.sample_limit = sample_limit
		self.root = None
		self.feature_words = None

	def train_from_csv(self, csv_file_name, num_words=1000):
		'''
			Train the DT model from a CSV file.
		'''
		self.feature_words, X, y = training_preprocess_from_csv(csv_file_name, num_words)
		self.train(X, y)

	def train(self, X, y):
		'''
			Train the DT model from data matrix and target vector.
		'''
		# Recursively grow the tree
		self.root = self._find_best_split(X, y, None, 1)

	def test_from_csv(self, csv_file_name):
		'''
			Test the DT model on a CSV file and return 0/1 loss.
		'''
		X, y = testing_preprocess_from_csv(csv_file_name, self.feature_words)
		return self.test(X, y)

	def test(self, X, y):
		'''
			Test the DT model from data matrix and target vector.
		'''
		preds = array([ self._get_prediction(X[:,i]) for i in range(X.shape[1]) ])
		# Compute loss score
		S = sum(abs(y - preds))*1.0 / size(y)
		print "ZERO-ONE-LOSS-DT %.4f" % S
		return S

	def _find_best_split(self, X, y, curr, depth):
		'''
			Private function for finding the best split for the current node.
		'''
		# Recursion-termination condition
		if depth > self.depth_limit or size(y) < self.sample_limit or size(unique(y)) == 1:
			return None
		# Gini index before split (optional)
		gini_before = _gini(y)
		# Gini index for all possible splits
		gini_gains = array([ self._get_gini_gain(X, y, k, gini_before) for k in range(X.shape[0]) ])
		# See who is the best
		k_best = argmax(gini_gains)
		i_neg = X[k_best] == 0
		i_pos = X[k_best] > 0
		# print depth, unique(y, return_counts=True), k_best, gini_gains[k_best:k_best+3]
		# Recurse
		curr = _Node(feature_index=k_best, label=self._get_majority_label(y))
		curr.left = self._find_best_split(X[:, i_neg], y[i_neg], curr.left, depth+1)
		curr.right = self._find_best_split(X[:, i_pos], y[i_pos], curr.right, depth+1)
		return curr

	def _get_majority_label(self, y):
		'''
			Return the majority label, invariant to the number of different labels.
		'''
		labels, counts = unique(y, return_counts=True)
		return labels[argmax(counts)]

	def _get_gini_gain(self, X, y, k, gini_before):
		'''
			Return the Gini gain for using binary feature k.
		'''
		# Split using feature k
		y_pos = y[ X[k] > 0 ]
		y_neg = y[ X[k] == 0 ]
		return gini_before - (size(y_pos)*_gini(y_pos) + size(y_neg)*_gini(y_neg)) / size(y)

	def _get_prediction(self, x):
		'''
			Traverse the decision tree using feature vector x and return the predicted label.
		'''
		curr = self.root
		label = None
		# While not leaf node
		while not curr == None:
			label = curr.label
			if x[curr.feature_index] > 0:
				next = curr.right
			else:
				next = curr.left
			curr = next
		return label