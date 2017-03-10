from pylab import *
from preprocess import *
import argparse

parser = argparse.ArgumentParser(description='Train and test a Naive Bayes Classifier.')
# Required arguments
parser.add_argument('trainingDataFilename',	help="Name of the training data csv file")
parser.add_argument('testDataFilename', help="Name of the test data csv file")
parser.add_argument('modelIdx', help="Model to use: 0 for NBC, 1 for LR and 2 for SVM")
args = parser.parse_args()

def main():
	print args

if __name__ == '__main__':
	main()