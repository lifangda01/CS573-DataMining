from pylab import *
import argparse
from preprocess import *
from lr import LogisticRegression
from svm import SupportVectorMachine
from nbc import nbc_train, nbc_test

parser = argparse.ArgumentParser(description='Train and test a LR or SVM Classifier.')
# Required arguments
parser.add_argument('trainingDataFilename',	help="Name of the training data csv file")
parser.add_argument('testDataFilename', help="Name of the test data csv file")
parser.add_argument('modelIdx', help="Model to use: 1 for LR and 2 for SVM")
args = parser.parse_args()

def evaluate_lr():
	lr = LogisticRegression()
	lr.train_from_csv('train-set.dat')
	lr.test_from_csv('test-set.dat')

def evaluate_svm():
	svm = SupportVectorMachine()
	svm.train_from_csv('train-set.dat')
	svm.test_from_csv('test-set.dat')

def cross_validate(csv_file_name, tssp, newFeature=False):
	'''
		Perform 10-fold incremental cross validation.
	'''
	total_num = 2000
	num_words = 4000
	lists_of_dict = []
	losses = zeros((3, len(tssp), 10)) # #models, #tss, #folds
	generate_train_and_test_files_cv(csv_file_name, 10)
	for i in range(10):
		lists_of_dict.append( csv_to_dict('cv%d.dat'%(i)) )
	for i, proportion in enumerate(tssp):
		print 'Training on proportion', proportion
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
			training_X, training_y = extract_word_feature_and_label(training_list_of_dict, feature_words)
			testing_X, testing_y = extract_word_feature_and_label(testing_list_of_dict, feature_words)
			# NBC
			M = nbc_train(feature_words, training_X, training_y)
			losses[0,i,j],_ =  nbc_test(M, testing_X, testing_y)
			# LR
			lr = LogisticRegression()
			lr.train(training_X, training_y)
			losses[1,i,j] = lr.test(testing_X, testing_y)
			# SVM
			svm = SupportVectorMachine()
			svm.train(training_X, training_y)
			losses[2,i,j] = svm.test(testing_X, testing_y)
	save('losses.npy', losses)
	# losses = load('losses.npy')
	# Calculate mean and standard error
	means = mean(losses, axis=2) # #models, #tss
	stds = std(losses, axis=2) # #models, #tss
	sterrs = stds / sqrt(10)
	# Plot
	fig = figure()
	ax = fig.add_subplot(111)
	for i, model, color in [(0,'NBC','r'), (1,'LR','g'), (2,'SVM','b')]:
		ax.errorbar(tssp,means[i],sterrs[i],c=color,marker='o', label=model)
	ax.legend(loc=1)
	ax.set_xlabel('Portion')
	ax.set_ylabel('Loss')
	ax.set_title('Training Set Size v.s. Zero-one-loss')
	show()
	return losses

def main():
	if args.modelIdx == '1':
		model = LogisticRegression()
	elif args.modelIdx == '2':
		model = SupportVectorMachine()
	else:
		cross_validate('yelp_data.csv', [0.01, 0.03, 0.05, 0.08, 0.1, 0.15])
	model.train_from_csv(args.trainingDataFilename)
	model.test_from_csv(args.testDataFilename)


if __name__ == '__main__':
	main()