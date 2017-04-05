import argparse
from analysis import analysis_1

parser = argparse.ArgumentParser(description='Train and test a LR or SVM Classifier.')
# Required arguments
parser.add_argument('trainingDataFilename',	help="Name of the training data csv file")
parser.add_argument('testDataFilename', help="Name of the test data csv file")
parser.add_argument('modelIdx', help="Model to use: 1 for Decision Tree, \
													2 for Bagged Decision Tree, \
													3 for Random Forest, \
													4 for SVM. \
													A1..A4 for analysis 1 to 4.")
args = parser.parse_args()

def main():
	if args.modelIdx == '1':
		model = DecisionTree()
	elif args.modelIdx == '2':
		model = BaggedDecisionTrees(n_estimators=50)
	elif args.modelIdx == '3':
		model = RandomForest(n_estimators=50)
	elif args.modelIdx == '4':
		model = SupportVectorMachine()
	elif args.modelIdx == 'A1':
		models = ['DT', 'BDT', 'RF', 'SVM']
		tssp = [0.025, 0.05, 0.125, 0.25]
		num_words = [1000]
		max_depth = [10]
		n_estimators = [50]
		analysis_1(	models, tssp, num_words, max_depth, n_estimators, debug=True)
		return
	elif args.modelIdx == 'A2':
		tssp = [0.25]
		analysis_2(tssp, debug=True)
		return
	else:
		return
	model.train_from_csv(args.trainingDataFilename)
	model.test_from_csv(args.testDataFilename)

if __name__ == '__main__':
	main()