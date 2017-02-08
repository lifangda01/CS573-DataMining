# from pylab import *
import csv

def csv_to_dict(csv_file_name):
	'''
		Parse CSV file into a list of dict with three fields: 
		'index', 'classLabel' and 'reviewText'.
	'''
	list_of_dict = []
	with open(csv_file_name, 'rU') as f:
		reader = csv.DictReader(f, dialect=csv.excel_tab, fieldnames=['index', 'classLabel', 'reviewText'])
		for row in reader:
			list_of_dict.append(row)
	return list_of_dict

def dict_to_csv(list_of_dict, csv_file_name):
	'''
		Write the given list of dicts into a csv file.
	'''
	with open(csv_file_name, 'w') as f:
		field_names = ['index', 'classLabel', 'reviewText']
		writer = csv.DictWriter(f, dialect=csv.excel_tab, fieldnames=field_names)
		for d in list_of_dict:
			writer.writerow(d)

def generate_train_and_test_set():
	pass

if __name__ == '__main__':
	lod = csv_to_dict('yelp_data.csv')
	dict_to_csv(lod, 'test.csv')
	lod = csv_to_dict('test.csv')
