from pylab import *
from preprocess import *
import argparse

parser = argparse.ArgumentParser(description='Train and test a Naive Bayes Classifier.')
# Required arguments
parser.add_argument('trainingDataFilename',	help="Name of the training data csv file")
parser.add_argument('testDataFilename', help="Name of the training data csv file")
args = parser.parse_args()

def train_from_csv(csv_file_name):
	'''
		Given the training csv file, construct the NB knowledge matrix.
	'''
	# Preprocess the csv file
	feature_words, X, y = training_preprocess_from_csv(csv_file_name)
	num_features = len(feature_words)
	num_samples = len(y)
	# Initialization with Laplace smoothing
	M = ones((num_features,2,2))
	# Training: populate the knowledge matrix
	for i in range(num_samples):
		M[range(num_features), y[i], X[:,i]] += 1
	return feature_words, M

def test_from_csv(csv_file_name, feature_words, M):
	'''
		Given the testing csv file, evaluate the NB classifier
	'''
	# Preprocess the csv file
	X, y = testing_preprocess_from_csv(csv_file_name, feature_words)
	num_features = len(feature_words)
	num_samples = len(y)	
	y_hat = zeros(num_samples).astype(int)
	# Compute the probability for state of nature
	P_theta = sum(sum(M, axis=0), axis=1).astype(float) / sum(M)
	# Normalization along axis 2
	M /= sum(M, axis=2)[:,:,None]
	# Predict
	for i in range(num_samples):
		# Calculate the likelihoods
		l0 = P_theta[0] * cumprod(M[range(num_features), 0, X[:,i]])[-1]
		l1 = P_theta[1] * cumprod(M[range(num_features), 1, X[:,i]])[-1]
		y_hat[i] = argmax([l0, l1])
	# Compute loss score
	S = sum(abs(y - y_hat))*1.0 / num_samples
	print "ZERO-ONE-LOSS %.4f" % S
	return S

def main():
	generate_train_and_test_files('yelp_data.csv', 0.8)
	feature_words, knowledge_matrix = train_from_csv(args.trainingDataFilename)
	loss = test_from_csv(args.testDataFilename, feature_words, knowledge_matrix)

if __name__ == '__main__':
	main()