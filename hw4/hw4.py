from pylab import *
import argparse
from preprocess import *
from dt import DecisionTree, BaggedDecisionTrees, RandomForest
from svm import SupportVectorMachine
from sklearn.tree import DecisionTreeClassifier as skDecisionTree
from sklearn.ensemble import RandomForestClassifier as skRandomForest

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

def cross_validate(csv_file_name, tssp, losses_file_name, new_feature=False, debug=False, num_words=1000):
	'''
		Perform 10-fold incremental cross validation.
	'''
	total_num = 2000
	lists_of_dict = []
	losses = zeros((4, len(tssp), 10)) # #models, #tss, #folds
	sklosses = zeros((2, len(tssp), 10))
	generate_train_and_test_files_cv(csv_file_name, 10)
	for i in range(10):
		lists_of_dict.append( csv_to_dict('cv%d.dat'%(i)) )
	for i, proportion in enumerate(tssp):
		for j in range(10):
			# Contruct train set
			training_lists_of_dict = lists_of_dict[:j] + lists_of_dict[j+1:]
			training_list_of_dict = [item for sublist in training_lists_of_dict for item in sublist]
			testing_list_of_dict = lists_of_dict[j]
			# Randomly select samples
			random_indices = permutation(len(training_list_of_dict))
			random_indices = random_indices[:int(total_num*proportion)]
			training_list_of_dict = [training_list_of_dict[k] for k in random_indices]
			# Find the word features
			feature_words = construct_word_feature(training_list_of_dict, num_words)
			# Extract features and labels
			training_X, training_y = extract_word_feature_and_label(training_list_of_dict, feature_words, new_feature=new_feature)
			testing_X, testing_y = extract_word_feature_and_label(testing_list_of_dict, feature_words, new_feature=new_feature)
			# DT
			dt = DecisionTree()
			dt.train(training_X, training_y)
			losses[0,i,j] = dt.test(testing_X, testing_y)
			# BDT
			bdt = BaggedDecisionTrees(n_estimators=50)
			bdt.train(training_X, training_y)
			losses[1,i,j] = bdt.test(testing_X, testing_y)
			# RF
			rf = RandomForest(n_estimators=50)
			rf.train(training_X, training_y)
			losses[2,i,j] = rf.test(testing_X, testing_y)
			# SVM
			svm = SupportVectorMachine()
			svm.train(training_X, training_y)
			losses[3,i,j] = svm.test(testing_X, testing_y)
			# Libary functions
			if debug:
				training_y[training_y==0] = -1
				testing_y[testing_y==0] = -1
				skdt = skDecisionTree(max_depth=10, min_samples_split=10)
				skdt.fit(training_X.T, training_y)
				sklosses[0,i,j] = 1-skdt.score(testing_X.T, testing_y)
				print "ZERO-ONE-LOSS-SKDT %.4f" % sklosses[0,i,j]
				skrf = skRandomForest(n_estimators=50, max_depth=10, min_samples_split=10)
				skrf.fit(training_X.T, training_y)
				sklosses[1,i,j] = 1-skrf.score(testing_X.T, testing_y)
				print "ZERO-ONE-LOSS-SKRF %.4f" % sklosses[1,i,j]
	save(losses_file_name, losses)
	save('debug_' + losses_file_name, sklosses)

def analysis_1(tssp, debug=False):
	csv_file_name = 'yelp_data.csv'
	losses_file_name = 'a1_losses.npy'
	cross_validate(csv_file_name, tssp, losses_file_name, debug=debug)
	# T-test
	losses = load(losses_file_name)
	# _, pvalues = ttest_ind(losses[models[0]].T, losses[models[1]].T)
	# print "P-values are", pvalues
	# print "Average P is", mean(pvalues)
	# Calculate mean and standard error
	means = mean(losses, axis=2) # #models, #tss
	stds = std(losses, axis=2) # #models, #tss
	sterrs = stds / sqrt(10)
	# Plot
	fig = figure()
	ax = fig.add_subplot(111)
	for i, model, color in [(0,'DT','r'), (1,'BDT','g'), (2,'RF','b'), (3,'SVM','c')]:
		ax.errorbar(tssp,means[i],sterrs[i],c=color,marker='o', label=model)
	if debug:
		sklosses = load('debug_' + losses_file_name)
		skmeans = mean(sklosses, axis=2) # #models, #tss
		skstds = std(sklosses, axis=2) # #models, #tss
		sksterrs = skstds / sqrt(10)
		for i, model, color in [(0,'skDT','y'), (1,'skRF','m')]:	
			ax.errorbar(tssp,skmeans[i],sksterrs[i],c=color,marker='o', label=model)
	ax.legend(loc=1)
	ax.set_xlabel('Portion')
	ax.set_ylabel('Loss')
	ax.set_title('Training Set Size v.s. Zero-one-loss')
	show()

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
		# tssp = [0.025, 0.05, 0.125, 0.25]
		tssp = [0.25]
		analysis_1(tssp, debug=True)
		return
	else:
		return
	model.train_from_csv(args.trainingDataFilename)
	model.test_from_csv(args.testDataFilename)

if __name__ == '__main__':
	main()