from pylab import *
import argparse
from preprocess import *
from lr import LogisticRegression
from svm import SupportVectorMachine
from nbc import nbc_train, nbc_test
from sklearn.linear_model import LogisticRegression as skLogisticRegresssion
from sklearn.svm import LinearSVC as skSupportVectorMachine
from scipy.stats import ttest_ind

parser = argparse.ArgumentParser(description='Train and test a LR or SVM Classifier.')
# Required arguments
parser.add_argument('trainingDataFilename',	help="Name of the training data csv file")
parser.add_argument('testDataFilename', help="Name of the test data csv file")
parser.add_argument('modelIdx', help="Model to use: 1 for LR and 2 for SVM")
args = parser.parse_args()

def cross_validate(csv_file_name, tssp, losses_file_name, new_feature=False, debug=False):
	'''
		Perform 10-fold incremental cross validation.
	'''
	total_num = 2000
	num_words = 4000
	lists_of_dict = []
	losses = zeros((3, len(tssp), 10)) # #models, #tss, #folds
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
			# NBC
			M = nbc_train(feature_words, training_X, training_y, new_feature=new_feature)
			losses[0,i,j],_ =  nbc_test(M, testing_X, testing_y)
			# LR
			lr = LogisticRegression()
			lr.train(training_X, training_y)
			losses[1,i,j] = lr.test(testing_X, testing_y)
			# SVM
			svm = SupportVectorMachine()
			svm.train(training_X, training_y)
			losses[2,i,j] = svm.test(testing_X, testing_y)
			# Libary functions
			if debug:
				training_y[training_y==0] = -1
				testing_y[testing_y==0] = -1
				sklr = skLogisticRegresssion()
				sklr.fit(training_X.T, training_y)
				sklosses[0,i,j] = 1-sklr.score(testing_X.T, testing_y)
				sksvm = skSupportVectorMachine()
				sksvm.fit(training_X.T, training_y)
				sklosses[1,i,j] = 1-sksvm.score(testing_X.T, testing_y)
	save(losses_file_name, losses)
	save('debug_' + losses_file_name, sklosses)

def analysis_1(tssp, debug=False):
	csv_file_name = 'yelp_data.csv'
	losses_file_name = 'a1_losses.npy'
	cross_validate(csv_file_name, tssp, losses_file_name, debug=debug)
	# T-test
	losses = load(losses_file_name)
	_, pvalues = ttest_ind(losses[models[0]].T, losses[models[1]].T)
	print "P-values are", pvalues
	print "Average P is", mean(pvalues)
	# Calculate mean and standard error
	means = mean(losses, axis=2) # #models, #tss
	stds = std(losses, axis=2) # #models, #tss
	sterrs = stds / sqrt(10)
	# Plot
	fig = figure()
	ax = fig.add_subplot(111)
	for i, model, color in [(0,'NBC','r'), (1,'LR','g'), (2,'SVM','b')]:
		ax.errorbar(tssp,means[i],sterrs[i],c=color,marker='o', label=model)
	if debug:
		sklosses = load('debug_' + losses_file_name)
		skmeans = mean(sklosses, axis=2) # #models, #tss
		skstds = std(sklosses, axis=2) # #models, #tss
		sksterrs = skstds / sqrt(10)
		for i, model, color in [(0,'skLR','y'), (1,'skSVM','m')]:	
			ax.errorbar(tssp,skmeans[i],sksterrs[i],c=color,marker='o', label=model)
	ax.legend(loc=1)
	ax.set_xlabel('Portion')
	ax.set_ylabel('Loss')
	ax.set_title('Training Set Size v.s. Zero-one-loss')
	show()

def analysis_2(tssp, debug=False):
	csv_file_name = 'yelp_data.csv'
	losses_file_name = 'a2_losses.npy'
	losses1 = load('a1_losses.npy')
	# cross_validate(csv_file_name, tssp, losses_file_name, new_feature=True, debug=debug)
	losses2 = load(losses_file_name)
	# Calculate mean and standard error
	means2 = mean(losses2, axis=2) # #models, #tss
	stds2 = std(losses2, axis=2) # #models, #tss
	sterrs2 = stds2 / sqrt(10)
	# Plot
	fig = figure()
	ax = fig.add_subplot(111)
	for i, model, color in [(0,'NBC','r'), (1,'LR','g'), (2,'SVM','b')]:
		ax.errorbar(tssp,means2[i],sterrs2[i],c=color,marker='o', label=model)
	if debug:
		sklosses = load('debug_' + losses_file_name)
		skmeans = mean(sklosses, axis=2) # #models, #tss
		skstds = std(sklosses, axis=2) # #models, #tss
		sksterrs = skstds / sqrt(10)
		for i, model, color in [(0,'skLR','y'), (1,'skSVM','m')]:	
			ax.errorbar(tssp,skmeans[i],sksterrs[i],c=color,marker='o', label=model)
	ax.legend(loc=1)
	ax.set_xlabel('Portion')
	ax.set_ylabel('Loss')
	ax.set_title('Training Set Size v.s. Zero-one-loss')
	# Compare with results in analysis 1 (without new features)
	# Calculate mean and standard error
	# Algorithm of interest
	aoi = 1
	# T-test
	_, pvalues = ttest_ind(losses1[aoi].T, losses2[aoi].T)
	print "P-values are", pvalues
	print "Average P is", mean(pvalues)
	means1 = mean(losses1, axis=2) # #models, #tss
	stds1 = std(losses1, axis=2) # #models, #tss
	sterrs1 = stds1 / sqrt(10)
	fig = figure()
	ax = fig.add_subplot(111)
	ax.errorbar(tssp,means1[aoi],sterrs1[aoi],c='g',marker='o', label='binary')
	ax.errorbar(tssp,means2[aoi],sterrs2[aoi],c='b',marker='o', label='trinary')
	ax.legend(loc=1)
	ax.set_xlabel('Portion')
	ax.set_ylabel('Loss')
	ax.set_title('NBC: Training Set Size v.s. Zero-one-loss')
	show()

def main():
	if args.modelIdx == '1':
		model = LogisticRegression()
	elif args.modelIdx == '2':
		model = SupportVectorMachine()
	elif args.modelIdx == '3':
		tssp = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15]
		analysis_1(tssp)
		return
	elif args.modelIdx == '4':
		tssp = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15]
		analysis_2(tssp)
		return
	else:
		return
	model.train_from_csv(args.trainingDataFilename)
	model.test_from_csv(args.testDataFilename)

if __name__ == '__main__':
	main()