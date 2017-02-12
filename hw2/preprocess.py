from pylab import *
import string
import csv
from collections import Counter

def csv_to_dict(csv_file_name):
	'''
		Parse CSV file into a list of dict with three fields: 
		'reviewID', 'classLabel' and 'reviewText'.
	'''
	list_of_dict = []
	with open(csv_file_name, 'rU') as f:
		reader = csv.DictReader(f, dialect=csv.excel_tab, fieldnames=['reviewID', 'classLabel', 'reviewText'])
		for row in reader:
			list_of_dict.append(row)
	return format_dict(list_of_dict)

def dict_to_csv(list_of_dict, csv_file_name):
	'''
		Write the given list of dicts into a csv file.
	'''
	with open(csv_file_name, 'w') as f:
		field_names = ['reviewID', 'classLabel', 'reviewText']
		writer = csv.DictWriter(f, dialect=csv.excel_tab, fieldnames=field_names)
		for d in list_of_dict:
			writer.writerow(d)

def generate_train_and_test_files(csv_file_name, train_percentage):
	'''
		Read a csv file and divide into train and test files.
	'''
	with open(csv_file_name, 'rU') as f:
		lines = f.readlines()
	total_num = len(lines)
	train_num = int(total_num * train_percentage)
	test_num = total_num - train_num
	rand_ind = permutation(total_num)
	with open('train-set.dat', 'w') as f:
		for i in rand_ind[:train_num]:
			f.write(lines[i])
	with open('test-set.dat', 'w') as f:
		for i in rand_ind[train_num:]:
			f.write(lines[i])

def format_dict(list_of_dict):
	'''
		Return a formated version of the list of dictionaries.
	'''
	for entry in list_of_dict:
		# Lower case only
		# Strip away the punctuations
		entry['reviewText'] = entry['reviewText'].lower().translate(None, string.punctuation)
	return list_of_dict

def dict_to_hist(list_of_dict):
	'''
		Given a list of dictionaries, generate its bag of words histogram.
	'''
	hist = Counter()
	for entry in list_of_dict:
		# Split and count appearances
		words = entry['reviewText'].split(' ')
		hist = hist + Counter(words)
	return hist

def construct_word_feature(list_of_dict):
	'''
		Given a csv file, construct the word feature with the 101-500 most frequent words.
		Return a list of the words.
	'''
	hist = dict_to_hist(list_of_dict)
	# A list of tuples (word, count)
	top_600 = hist.most_common(600)
	# Print the top 10 words in the feature
	for i, (w,_) in enumerate(top_600[100:110]):
		print "WORD%d %s" % (i+1, w)
	# Return top 101-600
	return [w for w,_ in top_600[100:]]

def extract_word_feature_and_label(list_of_dict, feature_words):
	'''
		Given the reviews in a list of dictionaries and a list of feature words,
		return a numpy array of feature vectors (num_features by num_reviews) with only 1 and 0.
	'''
	num_reviews = len(list_of_dict)
	num_features = len(feature_words)
	# Feature vectors
	X = zeros((num_features, num_reviews))
	# Ground-truth labels
	y = zeros(num_reviews)
	for i, entry in enumerate(list_of_dict):
		y[i] = entry['classLabel']
		mask = [word in entry['reviewText'].split(' ') for word in feature_words]
		X[array(mask), i] = 1
	return X, y

if __name__ == '__main__':
	list_of_dict = csv_to_dict('train-set.dat')
	# dict_to_csv(list_of_dict, 'test.csv')
	# list_of_dict = csv_to_dict('test.csv')
	# generate_train_and_test_files('yelp_data.csv', 0.80)
	# dict_to_hist('test-set.dat')
	feature_words = construct_word_feature(list_of_dict)
	X, y = extract_word_feature_and_label(list_of_dict, feature_words)