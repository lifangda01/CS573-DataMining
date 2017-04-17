from pylab import *

def A1():
	'''
		Visualizing one digit.
	'''
	raw = genfromtxt('digits-raw.csv', delimiter=',')
	image = raw[randint(raw.shape[0])][2:].reshape(28, 28)
	imshow(image, cmap='gray')
	show()

def A2():
	'''
		Visualize 1000 embedding examples in 2D.
	'''
	emb = genfromtxt('digits-embedding.csv', delimiter=',')
	perm = permutation(emb.shape[0])[:1000]
	labels = array(emb[perm, 1], dtype=int)
	colors = rand(10, 3)[labels, :]
	scatter(emb[perm, 2], emb[perm, 3], c=colors, alpha=0.9, s=10)
	legend()
	show()