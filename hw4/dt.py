from pylab import *
from preprocess import training_preprocess_from_csv, testing_preprocess_from_csv

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
		self.root = self._find_best_split(X, y, 1)

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

	def _find_best_split(self, X, y, depth):
		'''
			Private function for finding the best split for the current node.
		'''
		num_features, num_samples = X.shape
		# Recursion-termination condition
		if size(unique(y)) == 1:
			return _Node(label=y[0])
		if depth > self.max_depth or size(y) < self.min_samples_split:
			return None
		# Gini index before split (optional)
		gini_before = 0
		# Gini index for all possible splits
		# See who is the best
		# FIXME: unique features?
		if self.rf_tree:
			perm = permutation(num_features)[ :int(sqrt(num_features)) ]
			k_temp = self._get_best_gini_gain_index(X[perm], y, gini_before)
			k_best = perm[k_temp]
		else:
			k_best = self._get_best_gini_gain_index(X, y, gini_before)
		i_neg = X[k_best] == 0
		i_pos = X[k_best] > 0
		# Recurse
		curr = _Node(feature_index=k_best, label=self._get_majority_label(y))
		curr.left = self._find_best_split(X[:, i_neg], y[i_neg], depth+1)
		curr.right = self._find_best_split(X[:, i_pos], y[i_pos], depth+1)
		return curr

	def _get_majority_label(self, y):
		'''
			Return the majority label, invariant to the number of different labels.
		'''
		return round(sum(y)*1.0 / size(y))

	def _get_gini_gain(self, X, y, k, gini_before):
		'''
			Return the Gini gain for using binary feature k.
		'''
		# Split using feature k
		y_pos = y[ X[k] > 0 ]
		y_neg = y[ X[k] == 0 ]
		if size(y_pos) == 0 or size(y_neg) == 0: return gini_before
		return gini_before - (size(y_pos)*_gini(y_pos, sum(y_pos)) + size(y_neg)*_gini(y_neg, sum(y_neg))) / size(y)		

	def _get_best_gini_gain_index(self, X, y, gini_before):
		'''
			Return the Gini gains for all possible splits.
		'''
		X_num_pos = sum(X, axis=1) + 1e-8 # Hacky tweak to avoid division by zero
		X_num_neg = X.shape[1] - X_num_pos
		X_pos_y_pos = dot(X, y)
		X_pos_y_neg = X_num_pos - X_pos_y_pos
		X_neg_y_pos = dot( abs(X-1), y )
		X_neg_y_neg = X_num_neg - X_neg_y_pos
		gini_pos = 1 - (1.*X_pos_y_pos / X_num_pos)**2 - (1.*X_pos_y_neg / X_num_pos)**2
		gini_neg = 1 - (1.*X_neg_y_pos / X_num_neg)**2 - (1.*X_neg_y_neg / X_num_neg)**2
		return argmax( gini_before - ( gini_pos*X_num_pos + gini_neg*X_num_neg ) / X.shape[1] )

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
		return round(sum(y)*1.0 / size(y))
			
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
		num_features, num_samples = X.shape
		D = ones(num_samples)
		yp = copy(y)
		yp[y == 0] = -1
		for k in range(self.n_estimators):
			# Train a weak classifier
			dt = DecisionTree(max_depth=self.max_depth)
			D = D / sum(D)
			# Sample with replacement
			# FIXME: weighted distrbution
			indices = choice(arange(num_samples), size=num_samples, p=D)
			dt.train(X[:,indices], y[indices])
			# Get the predictions
			preds = array([ dt.predict(X[:,i]) for i in range(num_samples) ])
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
		num_features, num_samples = X.shape
		for i in range(self.n_estimators):
			dt = DecisionTree(max_depth=self.max_depth, rf_tree=True)
			# Sample with replacement
			indices = randint(0, num_samples, num_samples)
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