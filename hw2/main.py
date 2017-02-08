import argparse

parser = argparse.ArgumentParser(description='Train and test a Naive Bayes Classifier.')

# Required arguments
parser.add_argument('trainingDataFilename',	help="Name of the training data csv file")
parser.add_argument('testDataFilename', help="Name of the training data csv file")

args = parser.parse_args()

def main():
	print args.trainingDataFilename
	print args.testDataFilename

if __name__ == '__main__':
	main()