import argparse
from pylab import *
from analysis import *

parser = argparse.ArgumentParser(description='Kmeans clustering.')
# Required arguments
parser.add_argument('dataFilename',	help="Name of the data csv file")
parser.add_argument('K', help="Number of clusters")
parser.add_argument('--analysis', help="Analysis to perform")
args = parser.parse_args()

set_printoptions(precision=3)

def main():
	if args.analysis == 'A1':
		A1()
	if args.analysis == 'A2':
		A2()
	if args.analysis == 'B1':
		B1()
	if args.analysis == 'Bonus2':
		Bonus2()
	if args.analysis == 'Bonus3':
		Bonus3()

if __name__ == '__main__':
	main()