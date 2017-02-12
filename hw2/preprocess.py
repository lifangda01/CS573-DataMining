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
	return list_of_dict

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

def csv_to_hist(csv_file_name):
	'''
		Given a csv file, generate its bag of words histogram.
	'''
	bag_of_words = Counter()
	list_of_dict = csv_to_dict(csv_file_name)
	for entry in list_of_dict:
		# Lower case only
		s = entry['reviewText'].lower()
		# Strip away the punctuations
		s = s.translate(string.maketrans("",""), string.punctuation)
		# Split and count appearances
		words = s.split(' ')
		bag_of_words = bag_of_words + Counter(words)
	print bag_of_words

if __name__ == '__main__':
	# lod = csv_to_dict('yelp_data.csv')
	# dict_to_csv(lod, 'test.csv')
	# lod = csv_to_dict('test.csv')
	generate_train_and_test_files('yelp_data.csv', 0.80)
	csv_to_hist('test-set.dat')