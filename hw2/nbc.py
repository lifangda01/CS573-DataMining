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
	# Populate the knowledge matrix
	M = zeros((num_features,2,2))
	for i in range(num_samples):
		M[range(num_features), y[i], X[:,i]] += 1
	return feature_words, M

def test_from_csv(csv_file_name, feature_words, M):
	'''
		Given the testing csv file, evaluate the NB classifier
	'''
	X, y = testing_preprocess_from_csv(csv_file_name, feature_words)
	y_hat = 

def main():
	print args.trainingDataFilename
	print args.testDataFilename
	generate_train_and_test_files('yelp_data.csv', 0.8):


if __name__ == '__main__':
	main()