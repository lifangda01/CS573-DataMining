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
	elif args.analysis == 'A2':
		A2()
	elif args.analysis == 'B1':
		B1()
	elif args.analysis == 'B3':
		B3()
	elif args.analysis == 'B4':
		B4()
	elif args.analysis == 'C1':
		C1()
	elif args.analysis == 'C2':
		C2()
	elif args.analysis == 'C3':
		C3()
	elif args.analysis == 'C5':
		C5()
	elif args.analysis == 'Bonus2':
		Bonus2()
	elif args.analysis == 'Bonus3':
		Bonus3()
	elif args.analysis == 'Bonus4':
		Bonus4()
	elif args.analysis == 'Bonus5':
		Bonus5()
	else:
		raw = genfromtxt(args.dataFilename, delimiter=',')
		X = raw[:, 2:]
		y = get_normalized_labels(raw[:, 1])
		kmeans = KMeans(n_clusters=int(args.K))
		ind = kmeans.fit(X, y)

if __name__ == '__main__':
	main()