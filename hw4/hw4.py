from pylab import *
import argparse
from preprocess import *
from dt import DecisionTree

parser = argparse.ArgumentParser(description='Train and test a LR or SVM Classifier.')
# Required arguments
parser.add_argument('trainingDataFilename',	help="Name of the training data csv file")
parser.add_argument('testDataFilename', help="Name of the test data csv file")
parser.add_argument('modelIdx', help="Model to use: 1 for Decision Tree, \
													2 for Bagged Decision Tree, \
													3 for Random Forest.")
args = parser.parse_args()

def main():
	if args.modelIdx == '1':
		model = DecisionTree()
	elif args.modelIdx == '2':
		# model = SupportVectorMachine()
		pass
	elif args.modelIdx == '3':
		# tssp = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15]
		# analysis_1(tssp)
		return
	elif args.modelIdx == '4':
		# tssp = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15]
		# analysis_2(tssp)
		return
	else:
		return
	model.train_from_csv(args.trainingDataFilename)
	model.test_from_csv(args.testDataFilename)

if __name__ == '__main__':
	main()