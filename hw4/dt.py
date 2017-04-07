from pylab import *
from preprocess import training_preprocess_from_csv, testing_preprocess_from_csv

_gini = lambda y: 1 - sum(( 1.0*unique(y, return_counts=True)[1] / size(y) )**2)

class _Node(object):
	"""Container for a node in DecisionTree"""
	def __init__(self, feature_index=0, label=None):
		super(_Node, self).__init__()
		self.left = None
		self.right = None
		self.label = label
		self.feature_index = feature_index

class DecisionTree(object):
	"""DecisionTree model"""
	def __init__(self, max_depth=10, min_samples_split=10, rf_tree=False):
		super(DecisionTree, self).__init__()
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.rf_tree = rf_tree
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
		preds = array([ self.predict(X[:,i]) for i in range(X.shape[1]) ])
		# Compute loss score
		S = sum(abs(y - preds))*1.0 / size(y)
		print "ZERO-ONE-LOSS-DT %.4f" % S
		return S

	def predict(self, x):
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

	def _find_best_split(self, X, y, curr, depth):
		'''
			Private function for finding the best split for the current node.
		'''
		# Recursion-termination condition
		if size(unique(y)) == 1:
			return _Node(label=y[0])
		if depth > self.max_depth or size(y) < self.min_samples_split:
			return None
		# Gini index before split (optional)
		gini_before = _gini(y)
		# Gini index for all possible splits
		# See who is the best
		# FIXME: unique features?
		if self.rf_tree:
			perm = permutation(X.shape[0])
			gini_gains = array([ self._get_gini_gain(X, y, k, gini_before) for k in perm[:int(sqrt(X.shape[0]))] ])
			k_best = perm[ argmax(gini_gains) ]
		else:
			gini_gains = array([ self._get_gini_gain(X, y, k, gini_before) for k in range(X.shape[0]) ])
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

class BaggedDecisionTrees(object):
	"""Bag of decision trees."""
	def __init__(self, max_depth=10, n_estimators=50):
		super(BaggedDecisionTrees, self).__init__()
		self.n_estimators = n_estimators
		self.max_depth = max_depth
		self.trees = []
		
	def train_from_csv(self, csv_file_name, num_words=1000):
		'''
			Train the BDT model from a CSV file.
		'''
		self.feature_words, X, y = training_preprocess_from_csv(csv_file_name, num_words)
		self.train(X, y)

	def train(self, X, y):
		'''
			Train the BDT model from data matrix and target vector.
		'''
		for i in range(self.n_estimators):
			dt = DecisionTree(max_depth=self.max_depth)
			# Sample with replacement
			indices = randint(0, X.shape[1], X.shape[1])
			dt.train(X[:,indices], y[indices])
			self.trees.append(dt)

	def test_from_csv(self, csv_file_name):
		'''
			Test the BDT model on a CSV file and return 0/1 loss.
		'''
		X, y = testing_preprocess_from_csv(csv_file_name, self.feature_words)
		return self.test(X, y)

	def test(self, X, y):
		'''
			Test the BDT model from data matrix and target vector.
		'''
		preds = array([ self.predict(X[:,i]) for i in range(X.shape[1]) ])
		# Compute loss score
		S = sum(abs(y - preds))*1.0 / size(y)
		print "ZERO-ONE-LOSS-BT %.4f" % S
		return S

	def predict(self, x):
		'''
			Traverse the decision trees using feature vector x and return the predicted label.
		'''
		# Majority vote
		preds = array([ dt.predict(x) for dt in self.trees])
		return self._get_majority_label(preds)

	def _get_majority_label(self, y):
		'''
			Return the majority label, invariant to the number of different labels.
		'''
		labels, counts = unique(y, return_counts=True)
		return labels[argmax(counts)]
			
class BoostedDecisionTrees(BaggedDecisionTrees):
	"""BoostedDecisionTrees using AdaBoost"""
	def __init__(self, max_depth=10, n_estimators=50):
		super(BaggedDecisionTrees, self).__init__()
		self.n_estimators = n_estimators
		self.max_depth = max_depth
		self.trees = []
		self.W = zeros(n_estimators)

	def train(self, X, y):
		'''
			Train the BDT model from data matrix and target vector.
		'''
		# Initialize the sample weights
		D = ones(X.shape[1])
		yp = copy(y)
		yp[y == 0] = -1
		for k in range(self.n_estimators):
			# Train a weak classifier
			dt = DecisionTree(max_depth=self.max_depth)
			D = D / sum(D)
			# Sample with replacement
			indices = randint(0, X.shape[1], X.shape[1])
			dt.train(X[:,indices], y[indices])
			# Get the predictions
			preds = array([ dt.predict(X[:,i]) for i in range(X.shape[1]) ])
			preds[preds == 0] = -1
			# Update the weights
			# http://cs.nyu.edu/~dsontag/courses/ml12/slides/lecture13.pdf
			epsilon = 0.5 - 0.5 * dot(D, preds*yp)
			alpha = 0.5 * log((1-epsilon) / epsilon)
			D = multiply(D, exp(-alpha*preds*yp))
			# Remember the confidence value
			self.W[k] = alpha
			self.trees.append(dt)

	def test(self, X, y):
		'''
			Test the BODT model from data matrix and target vector.
		'''
		preds = array([ self.predict(X[:,i]) for i in range(X.shape[1]) ])
		# Compute loss score
		S = sum(abs(y - preds))*1.0 / size(y)
		print "ZERO-ONE-LOSS-BOT %.4f" % S
		return S

	def predict(self, x):
		'''
			Traverse the decision trees using feature vector x and return the predicted label.
		'''
		# Weighted vote
		preds = array([ dt.predict(x) for dt in self.trees])
		preds[preds == 0] = -1
		return int(dot(self.W, preds) >= 0)

class RandomForest(BaggedDecisionTrees):
	"""RandomForest"""
	def train(self, X, y):
		'''
			Train the RF model from data matrix and target vector.
		'''
		for i in range(self.n_estimators):
			dt = DecisionTree(max_depth=self.max_depth, rf_tree=True)
			# Sample with replacement
			indices = randint(0, X.shape[1], X.shape[1])
			dt.train(X[:,indices], y[indices])
			self.trees.append(dt)

	def test(self, X, y):
		'''
			Test the RF model from data matrix and target vector.
		'''
		preds = array([ self.predict(X[:,i]) for i in range(X.shape[1]) ])
		# Compute loss score
		S = sum(abs(y - preds))*1.0 / size(y)
		print "ZERO-ONE-LOSS-RF %.4f" % S
		return S