from pylab import *
import argparse
from preprocess import *
from dt import DecisionTree, BaggedDecisionTrees, RandomForest
from svm import SupportVectorMachine

parser = argparse.ArgumentParser(description='Train and test a LR or SVM Classifier.')
# Required arguments
parser.add_argument('trainingDataFilename',	help="Name of the training data csv file")
parser.add_argument('testDataFilename', help="Name of the test data csv file")
parser.add_argument('modelIdx', help="Model to use: 1 for Decision Tree, \
													2 for Bagged Decision Tree, \
													3 for Random Forest \
													4 for SVM.")
args = parser.parse_args()

def main():
	if args.modelIdx == '1':
		model = DecisionTree()
	elif args.modelIdx == '2':
		model = BaggedDecisionTrees(num_trees=50)
	elif args.modelIdx == '3':
		model = RandomForest(num_trees=50)
	elif args.modelIdx == '4':
		model = SupportVectorMachine()
	else:
		# tssp = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15]
		# analysis_1(tssp)
		return
	model.train_from_csv(args.trainingDataFilename)
	model.test_from_csv(args.testDataFilename)

if __name__ == '__main__':
	main()