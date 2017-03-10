from pylab import *
from preprocess import *
import argparse

parser = argparse.ArgumentParser(description='Train and test a Naive Bayes Classifier.')
# Required arguments
parser.add_argument('trainingDataFilename',	help="Name of the training data csv file")
parser.add_argument('testDataFilename', help="Name of the test data csv file")
args = parser.parse_args()

def train(feature_words, X, y):
	'''
		Train the NBC by populating the knowledge matrix.
	'''
	num_features = len(feature_words)
	num_samples = len(y)
	# Initialization with Laplace smoothing
	M = ones((num_features,2,2))
	# Training: populate the knowledge matrix
	for i in range(num_samples):
		M[range(num_features), y[i], X[:,i]] += 1
	return M

def test(M, X, y):
	'''
		Test the NBC and return the zero-one-loss.
	'''
	num_features = M.shape[0]
	num_samples = len(y)	
	y_hat = zeros(num_samples).astype(int)
	# Compute the probability for state of nature
	P_theta = sum(sum(M, axis=0), axis=1).astype(float) / sum(M)
	# Normalization along axis 2
	M /= sum(M, axis=2)[:,:,None]
	# Predict
	for i in range(num_samples):
		# Calculate the likelihoods
		l0 = P_theta[0] * cumprod(M[range(num_features), 0, X[:,i]])[-1]
		l1 = P_theta[1] * cumprod(M[range(num_features), 1, X[:,i]])[-1]
		y_hat[i] = argmax([l0, l1])
	# Compute loss score
	S = sum(abs(y - y_hat))*1.0 / num_samples
	print "ZERO-ONE-LOSS %.4f" % S
	# Also return the baseline score
	return S, sum(abs(y - argmax(bincount(y))))*1.0 / num_samples

def train_from_csv(csv_file_name, num_words=500):
	'''
		Given the training csv file, return the NB knowledge matrix.
	'''
	# Preprocess the csv file
	feature_words, X, y = training_preprocess_from_csv(csv_file_name, num_words=num_words)
	knowledge_matrix = train(feature_words, X, y)
	return feature_words, knowledge_matrix

def test_from_csv(csv_file_name, feature_words, knowledge_matrix):
	'''
		Given the testing csv file, evaluate the NB classifier.
		Return both the NBC loss and the baseline loss.
	'''
	# Preprocess the csv file
	X, y = testing_preprocess_from_csv(csv_file_name, feature_words)
	nbc_loss, baseline_loss = test(knowledge_matrix, X, y)
	return nbc_loss, baseline_loss

def evaluate_wrt_train_size(portions, num_repeat=10):
	'''
		Repeat the experiment each time on different portions.
	'''
	nbc_losses = zeros((num_repeat, len(portions)))
	baseline_losses = zeros((num_repeat, len(portions)))
	for i, p in enumerate(portions):
		for j in range(num_repeat):
			generate_train_and_test_files('yelp_data.csv', p)
			feature_words, knowledge_matrix = train_from_csv('train-set.dat')
			nbc_losses[j,i], baseline_losses[j,i] = test_from_csv('test-set.dat', feature_words, knowledge_matrix)
	# Save the results
	savetxt('nbc_losses_q3.dat', nbc_losses), savetxt('baseline_losses_q3.dat', baseline_losses)
	# Load back in case of need
	# nbc_losses , baseline_losses = loadtxt('nbc_losses_q3.dat'), loadtxt('baseline_losses_q3.dat')
	# Analysis and plot
	nbc_means, nbc_stds = mean(nbc_losses, axis=0), std(nbc_losses, axis=0)
	baseline_means, baseline_stds = mean(baseline_losses, axis=0), std(baseline_losses, axis=0)
	fig = figure()
	ax = fig.add_subplot(111)
	ax.errorbar(portions,nbc_means,nbc_stds,c='r',marker='o', label='NBC')
	ax.errorbar(portions,baseline_means,baseline_stds,c='g',marker='o', label='Baseline')
	ax.legend(loc='right')
	ax.set_xlabel('Portion')
	ax.set_ylabel('Loss')
	ax.set_title('Training Set Size v.s. Zero-one-loss')
	show()

def evaluate_wrt_feature_size(words, num_repeat=10):
	'''
		Repeat the experiment each time on different portions.
	'''
	nbc_losses = zeros((num_repeat, len(words)))
	baseline_losses = zeros((num_repeat, len(words)))
	for i, w in enumerate(words):
		for j in range(num_repeat):
			generate_train_and_test_files('yelp_data.csv', 0.5)
			feature_words, knowledge_matrix = train_from_csv('train-set.dat', w)
			nbc_losses[j,i], baseline_losses[j,i] = test_from_csv('test-set.dat', feature_words, knowledge_matrix)
	# Save the results
	savetxt('nbc_losses_q4.dat', nbc_losses), savetxt('baseline_losses_q4.dat', baseline_losses)
	# Analysis and plot
	nbc_means, nbc_stds = mean(nbc_losses, axis=0), std(nbc_losses, axis=0)
	baseline_means, baseline_stds = mean(baseline_losses, axis=0), std(baseline_losses, axis=0)
	fig = figure()
	ax = fig.add_subplot(111)
	ax.errorbar(words,nbc_means,nbc_stds,c='r',marker='o', label='NBC')
	ax.errorbar(words,baseline_means,baseline_stds,c='g',marker='o', label='Baseline')
	ax.legend(loc='right')
	ax.set_xscale('log')
	ax.set_xlabel('# words')
	ax.set_ylabel('Loss')
	ax.set_title('Feature Size v.s. Zero-one-loss')
	show()
	
def main():
	# generate_train_and_test_files('yelp_data.csv', 0.5)
	feature_words, knowledge_matrix = train_from_csv(args.trainingDataFilename)
	loss = test_from_csv(args.testDataFilename, feature_words, knowledge_matrix)
	# evaluate_wrt_train_size([0.01, 0.05, 0.10, 0.20, 0.50, 0.90], num_repeat=10)
	# evaluate_wrt_feature_size([10, 50, 250, 500, 1000, 4000], num_repeat=10)

if __name__ == '__main__':
	main()