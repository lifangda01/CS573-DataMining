from pylab import *
from kmeans import KMeans
from pca import PCA

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
	show()

def B1():
	'''
		Plot WC_SSD and SC over K.
	'''
	K = [2, 4, 6, 8, 16, 32]
	fnames = ['digits-embedding.csv', 'digits-embedding-2467.csv', 'digits-embedding-67.csv']
	wc_ssd = zeros((len(fnames), len(K)))
	sc = zeros((len(fnames), len(K)))
	for i, fname in enumerate(fnames):	
		X = genfromtxt(fname, delimiter=',')[:, 2:]
		for j, k in enumerate(K):
			kmeans = KMeans(n_clusters=k)
			kmeans.fit(X)
			wc_ssd[i, j], sc[i, j], _ = kmeans.get_evals()
	# Plot WC_SSD
	figure()
	for i, fname in enumerate(fnames):
		plot(K, wc_ssd[i], label=fname)
	legend()
	title('WC_SSD v.s. K')
	figure()
	for i, fname in enumerate(fnames):
		plot(K, sc[i], label=fname)
	legend()
	title('SC v.s. K')
	show()

def Bonus2():
	'''
		Visualization of the first 10 eigen vectors.
	'''
	raw = genfromtxt('digits-raw.csv', delimiter=',')
	X = raw[:, 2:]
	pca = PCA(10)
	eigvec = pca.fit(X)
	eigimg = eigvec.reshape(10, 28, 28)
	for r in range(2):
		for c in range(5):
			i = r*5 + c
			subplot(2, 5, i + 1)
			imshow(eigimg[i], cmap='gray')
			title(str(i))
	show()

def Bonus3():
	'''
		Scatter plot of samples projected onto the first 
		two eigenvectors.
	'''
	raw = genfromtxt('digits-raw.csv', delimiter=',')
	X = raw[:, 2:]
	pca = PCA(2)
	X_new = pca.fit_transform(X)
	perm = permutation(X.shape[0])[:1000]
	labels = array(raw[perm, 1], dtype=int)
	colors = rand(10, 3)[labels, :]
	scatter(X_new[perm, 0], X_new[perm, 1], c=colors, alpha=0.9, s=10)
	show()
