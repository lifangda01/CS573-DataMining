from pylab import *
import argparse
from preprocess import *
from lr import LogisticRegression
from svm import SupportVectorMachine

parser = argparse.ArgumentParser(description='Train and test a Naive Bayes Classifier.')
# Required arguments
parser.add_argument('trainingDataFilename',	help="Name of the training data csv file")
parser.add_argument('testDataFilename', help="Name of the test data csv file")
parser.add_argument('modelIdx', help="Model to use: 0 for NBC, 1 for LR and 2 for SVM")
args = parser.parse_args()

def evaluate_lr():
	lr = LogisticRegression()
	lr.train_from_csv('train-set.dat')
	lr.test_from_csv('test-set.dat')

def evaluate_svm():
	svm = SupportVectorMachine()
	svm.train_from_csv('train-set.dat')
	svm.test_from_csv('test-set.dat')

def main():
	# generate_train_and_test_files_cv('yelp_data.csv', 10)
	generate_train_and_test_files('yelp_data.csv', 0.3)
	# evaluate_lr()
	evaluate_svm()

if __name__ == '__main__':
	main()