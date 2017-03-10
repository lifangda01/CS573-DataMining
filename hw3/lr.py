from pylab import *
from preprocess import training_preprocess_from_csv, testing_preprocess_from_csv

class LogisticRegression(object):
	"""LogisticRegression model"""
	def __init__(self, reg_lambda=0.01, gd_eta=0.01, max_iter=100, min_update=1e-6, num_words=4000):
		super(LogisticRegression, self).__init__()
		# Object-wise variables
		self.reg_lambda = reg_lambda
		self.gd_eta = gd_eta
		self.max_iter = max_iter
		self.min_update = min_update
		self.num_words = num_words
		self.w = None
		self.feature_words = None

	def train_from_csv(self, csv_file_name):
		'''
			Train the LR model from a CSV file.
		'''
		self.feature_words, X, y = training_preprocess_from_csv(csv_file_name, 4000)
		self.train(X, y)

	def train(self, X, y):
		'''
			Train the LR model from data matrix and target vector.
		'''
		num_features, num_samples = X.shape
		# Plus one for bias
		self.w = zeros(num_features+1)
		X = vstack((ones(num_samples), X))
		i = 1
		weight_update = 1
		last_weight_norm, curr_weight_norm = 0, 1
		# Main gradient ascent loop
		while i <= self.max_iter and abs(last_weight_norm - curr_weight_norm) >= self.min_update:
			last_weight_norm = curr_weight_norm
			# Make predictions using current weight
			y_hat = 1.0 / (1 + exp( -dot(self.w, X) ))
			# Calculate the gradients for weights
			dw = dot( y - y_hat, X.T) - self.w * self.reg_lambda
			# Gradient ascent
			self.w += dw * self.gd_eta
			# Check update amount
			curr_weight_norm = norm(self.w)
			i +=1

	def test_from_csv(self, csv_file_name):
		'''
			Test the LR model on a CSV file and return 0/1 loss.
		'''
		X, y = testing_preprocess_from_csv(csv_file_name, self.feature_words)
		return self.test(X, y)

	def test(self, X, y):
		'''
			Test the LR model on data matrix and target vector.
		'''
		num_features, num_samples = X.shape
		X = vstack((ones(num_samples), X))
		y_hat = 1.0 / (1 + exp( -dot(self.w, X) ))
		preds = zeros(num_samples)
		preds[y_hat >= 0.5] = 1.0
		# Compute loss score
		S = sum(abs(y - preds))*1.0 / num_samples
		print "ZERO-ONE-LOSS-LR %.4f" % S
		return S
