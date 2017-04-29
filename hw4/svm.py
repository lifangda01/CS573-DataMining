from pylab import *
from preprocess import training_preprocess_from_csv, testing_preprocess_from_csv

class SupportVectorMachine(object):
	"""SupportVectorMachine model"""
	def __init__(self, reg_lambda=0.01, gd_eta=0.5, max_iter=100, min_update=1e-6):
		super(SupportVectorMachine, self).__init__()
		# Object-wise variables
		self.reg_lambda = reg_lambda
		self.gd_eta = gd_eta
		self.max_iter = max_iter
		self.min_update = min_update
		self.w = None
		self.feature_words = None

	def train_from_csv(self, csv_file_name, num_words=1000):
		'''
			Train the LR model from a CSV file.
		'''
		self.feature_words, X, y = training_preprocess_from_csv(csv_file_name, num_words)
		self.train(X, y)

	def train(self, X, y):
		'''
			Train the LR model from data matrix and target vector.
		'''
		num_features, num_samples = X.shape
		# Since SVM requires labels to be -1 and 1, rectify the labels
		y[y==0] = -1
		# Plus one for bias
		self.w = zeros(num_features+1)
		X = vstack((ones(num_samples), X))
		i = 1
		last_weights = self.w+1
		# Main gradient descent loop
		while i <= self.max_iter and norm(last_weights - self.w) >= self.min_update:
			last_weights = copy(self.w)
			# Make predictions using current weight
			y_hat = dot(self.w, X)
			# Calculate the gradients for weights
			# yy_hat = dot( y , y_hat)
			yy_hat = multiply( y , y_hat) # Mask per sample
			# dh = - dot( y , X.T ) # Hinge loss gradient
			dh = - multiply( X , y ) # Hinge loss gradient per sample
			dh[:, yy_hat >= 1] = 0
			# dw = ( self.reg_lambda * self.w + dh ) / num_samples
			dw = self.reg_lambda * self.w + sum(dh, axis=1) / num_samples
			# Gradient descent
			self.w -= dw * self.gd_eta
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
		y_hat = dot(self.w, X)
		preds = zeros(num_samples)
		preds[y_hat >= 0] = 1.0
		# Compute loss score
		S = sum(abs(y - preds))*1.0 / num_samples
		print "ZERO-ONE-LOSS-SVM %.4f" % S
		return S
