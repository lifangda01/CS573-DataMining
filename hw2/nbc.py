import argparse
from preprocess import *

parser = argparse.ArgumentParser(description='Train and test a Naive Bayes Classifier.')

# Required arguments
parser.add_argument('trainingDataFilename',	help="Name of the training data csv file")
parser.add_argument('testDataFilename', help="Name of the training data csv file")

args = parser.parse_args()

def function():
	pass

def main():
	print args.trainingDataFilename
	print args.testDataFilename
	generate_train_and_test_files('yelp_data.csv', 0.8):


if __name__ == '__main__':
	main()