from pylab import *
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score, mutual_info_score
from kmeans import KMeans, wc_ssd, sc, nmi
from pca import PCA

def get_normalized_labels(y):
	'''
		Replace the labels with values starting from 0.
	'''
	for i, l in enumerate(unique(y)):
		y[y == l] = i
	return y

def generate_pca_embedding_files():
	'''
		Generate PCA embedding csv files for the experiments.
	'''
	raw = genfromtxt('digits-raw.csv', delimiter=',')
	X = raw[:, 2:]
	pca = PCA(10)
	X_new = pca.fit_transform(X)
	raw_new = hstack((raw[:,:2], X_new))
	savetxt('digits-pca-embedding.csv', raw_new, delimiter=',')
	raw = genfromtxt('digits-raw-2467.csv', delimiter=',')
	X = raw[:, 2:]
	pca = PCA(10)
	X_new = pca.fit_transform(X)
	raw_new = hstack((raw[:,:2], X_new))
	savetxt('digits-pca-embedding-2467.csv', raw_new, delimiter=',')
	raw = genfromtxt('digits-raw-67.csv', delimiter=',')
	X = raw[:, 2:]
	pca = PCA(10)
	X_new = pca.fit_transform(X)
	raw_new = hstack((raw[:,:2], X_new))
	savetxt('digits-pca-embedding-67.csv', raw_new, delimiter=',')

def generate_subset_files():
	'''
		Generate embedding subsets: 2467 and 67.
	'''
	# tSNE embedding
	raw = genfromtxt('digits-embedding.csv', delimiter=',')
	y = raw[:, 1]
	raw_new = raw[logical_or(y == 2, logical_or(y == 4, logical_or(y == 6, y == 7)))]
	savetxt('digits-embedding-2467.csv', raw_new, delimiter=',')
	raw_new = raw[logical_or(y == 6, y == 7)]
	savetxt('digits-embedding-67.csv', raw_new, delimiter=',')
	# Raw files
	raw = genfromtxt('digits-raw.csv', delimiter=',')
	y = raw[:, 1]
	raw_new = raw[logical_or(y == 2, logical_or(y == 4, logical_or(y == 6, y == 7)))]
	savetxt('digits-raw-2467.csv', raw_new, delimiter=',')
	raw_new = raw[logical_or(y == 6, y == 7)]
	savetxt('digits-raw-67.csv', raw_new, delimiter=',')

def A1():
	'''
		Visualizing one digit from each class.
	'''
	raw = genfromtxt('digits-raw-small.csv', delimiter=',')
	for i in range(10):
		ind = raw[:, 1] == i
		r = randint(sum(ind))
		subplot(2, 5, i + 1)
		imshow(raw[ind][r, 2:].reshape(28, 28), cmap='gray')
	show()

def A2():
	'''
		Visualize 1000 embedding examples in 2D.
	'''
	emb = genfromtxt('digits-embedding.csv', delimiter=',')
	perm = permutation(emb.shape[0])[:1000]
	labels = array(emb[perm, 1], dtype=int)
	colors = rand(10, 3)[labels, :]
	scatter(emb[perm, 2], emb[perm, 3], c=colors, alpha=0.9, s=30)
	show()

def B1(pca=False):
	'''
		Plot WC_SSD and SC over K.
	'''
	K = [2, 4, 6, 8, 16, 32]
	fnames = ['digits-embedding.csv', 'digits-embedding-2467.csv', 'digits-embedding-67.csv']
	wc_ssd_val = zeros((len(fnames), len(K)))
	sc_val= zeros((len(fnames), len(K)))
	for i, fname in enumerate(fnames):  
		X = genfromtxt(fname, delimiter=',')[:, 2:]
		for j, k in enumerate(K):
			kmeans = KMeans(n_clusters=k)
			kmeans.fit(X)
			wc_ssd_val[i, j], sc_val[i, j], _ = kmeans.get_evals()
	# Plot WC_SSD
	figure()
	for i, fname in enumerate(fnames):
		plot(K, wc_ssd_val[i], label=fname)
	legend()
	title('WC_SSD v.s. K')
	figure()
	for i, fname in enumerate(fnames):
		plot(K, sc_val[i], label=fname)
	legend()
	title('SC v.s. K')
	show()

def B3():
	'''
		Repeat 10 times for each K.
	'''
	K = [2, 4, 6, 8, 16, 32]
	fnames = ['digits-embedding.csv', 'digits-embedding-2467.csv', 'digits-embedding-67.csv']
	wc_ssd_val = zeros((len(fnames), len(K), 10))
	sc_val= zeros((len(fnames), len(K), 10))
	for i, fname in enumerate(fnames):  
		X = genfromtxt(fname, delimiter=',')[:, 2:]
		for j, k in enumerate(K):
			for m in range(10):
				kmeans = KMeans(n_clusters=k)
				kmeans.fit(X)
				wc_ssd_val[i, j, m], sc_val[i, j, m], _ = kmeans.get_evals()
	save('B3_wc_ssd_val.npy', wc_ssd_val), save('B3_sc_val.npy', sc_val)
	wc_ssd_val = load('B3_wc_ssd_val.npy')
	sc_val = load('B3_sc_val.npy')
	ssd_means = mean(wc_ssd_val, axis=2)
	sc_means = mean(sc_val, axis=2)
	ssd_std = std(wc_ssd_val, axis=2)
	sc_std = std(sc_val, axis=2)
	# Plot WC_SSD
	figure()
	for i, fname in enumerate(fnames):
		errorbar(K, ssd_means[i], ssd_std[i], capsize=4, label=fname)
	legend()
	title('WC_SSD v.s. K')
	figure()
	for i, fname in enumerate(fnames):
		errorbar(K, sc_means[i], sc_std[i], capsize=4, label=fname)
	legend()
	title('SC v.s. K')
	show()

def B4(pca=False):
	'''
		Evaluate using NMI and visualize in 2D.
	'''
	fnames = ['digits-embedding.csv', 'digits-embedding-2467.csv', 'digits-embedding-67.csv']
	nmi = zeros(len(fnames))
	for i, k, fname in zip([0, 1, 2], [8, 4, 2], fnames):
		raw = genfromtxt(fname, delimiter=',')
		X = raw[:, 2:]
		y = get_normalized_labels(raw[:, 1])
		kmeans = KMeans(n_clusters=k)
		ind = kmeans.fit(X, y)
		_, _, nmi[i] = kmeans.get_evals()
		figure()
		perm = permutation(X.shape[0])[:1000]
		X = X[perm]
		ind = ind[perm]
		colors = rand(k, 3)[ind, :]
		scatter(X[:, 0], X[:, 1], c=colors, alpha=0.9, s=30)
	print fnames
	print "NMI =", nmi
	show()

def C1():
	'''
		Single linkage agglomerative clustering.
	'''
	raw = genfromtxt('digits-embedding.csv', delimiter=',')
	# Randomly select 10 images from each digit
	X = zeros((100, 2))
	y = zeros(100)
	for i in range(10):
		ind = raw[:, 1] == i
		perm = permutation(sum(ind))[:10]
		X[10 * i: 10 * i + 10] = raw[ind, 2:][perm]
		y[10 * i: 10 * i + 10] = raw[ind, 1][perm]
	Z = linkage(X, method='single')
	dendrogram(Z, leaf_font_size=16)
	title('Single Linkage')
	xlabel('Sample Index')
	ylabel('Distance')
	show()

def C2():
	'''
		Complete & average linkage agglomerative clustering.
	'''
	raw = genfromtxt('digits-embedding.csv', delimiter=',')
	# Randomly select 10 images from each digit
	X = zeros((100, 2))
	y = zeros(100)
	for i in range(10):
		ind = raw[:, 1] == i
		perm = permutation(sum(ind))[:10]
		X[10 * i: 10 * i + 10] = raw[ind, 2:][perm]
		y[10 * i: 10 * i + 10] = raw[ind, 1][perm]
	for m in ['complete', 'average']:
		figure()
		Z = linkage(X, method=m)
		dendrogram(Z, leaf_font_size=16)
		title(m + ' Linkage')
		xlabel('Sample Index')
		ylabel('Distance')
	show()

def C3():
	'''
		Plot WC_SSD and SC v.s. K.
	'''
	K = [2, 4, 8, 16, 32]
	methods = ['single', 'complete', 'average']
	raw = genfromtxt('digits-embedding.csv', delimiter=',')
	# Randomly select 10 images from each digit
	X = zeros((100, 2))
	y = zeros(100)
	wc_ssd_val = zeros((len(methods), len(K), 10))
	sc_val= zeros((len(methods), len(K), 10))
	for i in range(10):
		ind = raw[:, 1] == i
		perm = permutation(sum(ind))[:10]
		X[10 * i: 10 * i + 10] = raw[ind, 2:][perm]
		y[10 * i: 10 * i + 10] = raw[ind, 1][perm]
	for i, m in enumerate(methods):
		Z = linkage(X, method=m)
		for j, k in enumerate(K):
			for l in range(10):
				ind = fcluster(Z, k, criterion='maxclust')
				ind = get_normalized_labels(ind)
				# Find the cluster centers
				C = array([mean(X[ind == n], axis=0) for n in range(k)])
				wc_ssd_val[i, j, l] = wc_ssd(X, C, ind)
				sc_val[i, j, l] = sc(X, C, ind)
	ssd_means = mean(wc_ssd_val, axis=2)
	sc_means = mean(sc_val, axis=2)
	ssd_std = std(wc_ssd_val, axis=2)
	sc_std = std(sc_val, axis=2)
	figure()
	for i, m in enumerate(methods):
		errorbar(K, ssd_means[i], ssd_std[i], capsize=4, label=m)
	title('WC_SSD v.s. K')
	xlabel('K')
	ylabel('WC_SSD')
	legend()
	figure()
	for i, m in enumerate(methods):
		errorbar(K, sc_means[i], sc_std[i], capsize=4, label=m)
	title('SC v.s. K')
	xlabel('K')
	ylabel('SC')
	legend()
	show()

def C5():
	'''
		NMI across distance measures.
	'''

	K = [32, 16, 16]
	methods = ['single', 'complete', 'average']
	raw = genfromtxt('digits-embedding.csv', delimiter=',')
	# Randomly select 10 images from each digit
	X = zeros((100, 2))
	y = zeros(100)
	nmi_val = zeros((len(methods), 10))
	for i in range(10):
		ind = raw[:, 1] == i
		perm = permutation(sum(ind))[:10]
		X[10 * i: 10 * i + 10] = raw[ind, 2:][perm]
		y[10 * i: 10 * i + 10] = raw[ind, 1][perm]
	for i, m in enumerate(methods):
		for j in range(10):
			Z = linkage(X, method=m)
			ind = fcluster(Z, K[i], criterion='maxclust')
			ind = get_normalized_labels(ind)
			nmi_val[i, j] = nmi(y, ind)
	print "Average NMI across 10 trials:"
	print methods
	print mean(nmi_val, axis=1)
	show()

def Bonus2():
	'''
		Visualization of the first 10 eigen vectors.
	'''
	# raw = genfromtxt('digits-raw.csv', delimiter=',')
	raw = genfromtxt('digits-raw-small.csv', delimiter=',')
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

def Bonus4():
	'''
		Repeat B1, B2, B4 with PCA embedding.
	'''
	K = [2, 4, 6, 8, 16, 32]
	fnames = ['digits-pca-embedding.csv', 'digits-pca-embedding-2467.csv', 'digits-pca-embedding-67.csv']
	wc_ssd_val = zeros((len(fnames), len(K), 10))
	sc_val= zeros((len(fnames), len(K), 10))
	nmi_val= zeros((len(fnames), len(K), 10))
	for i, fname in enumerate(fnames):  
		raw = genfromtxt(fname, delimiter=',')
		X = raw[:, 2:]
		y = get_normalized_labels(raw[:, 1])
		for j, k in enumerate(K):
			for m in range(10):
				kmeans = KMeans(n_clusters=k)
				ind = kmeans.fit(X, y)
				wc_ssd_val[i, j, m], sc_val[i, j, m], nmi_val[i, j, m] = kmeans.get_evals()
	save('Bonus_wc_ssd_val.npy', wc_ssd_val)
	save('Bonus_sc_val.npy', sc_val)
	save('Bonus_nmi_val.npy', nmi_val)
	wc_ssd_val = load('Bonus_wc_ssd_val.npy')
	sc_val = load('Bonus_sc_val.npy')
	# nmi_val = load('Bonus_nmi_val.npy')
	ssd_means = mean(wc_ssd_val, axis=2)
	sc_means = mean(sc_val, axis=2)
	ssd_std = std(wc_ssd_val, axis=2)
	sc_std = std(sc_val, axis=2)
	# Plot WC_SSD
	figure()
	for i, fname in enumerate(fnames):
		errorbar(K, ssd_means[i], ssd_std[i], capsize=4, label=fname)
	legend()
	title('WC_SSD v.s. K')
	figure()
	for i, fname in enumerate(fnames):
		errorbar(K, sc_means[i], sc_std[i], capsize=4, label=fname)
	legend()
	title('SC v.s. K')
	print fnames
	print "NMI =", mean(nmi_val, axis=2)
	show()

def Bonus5():
	'''
		Visualize clustering results using PCA embedding.
	'''
	K = [8, 4, 2]
	fnames = ['digits-pca-embedding.csv', 'digits-pca-embedding-2467.csv', 'digits-pca-embedding-67.csv']
	for k, fname in zip(K, fnames):
		raw = genfromtxt(fname, delimiter=',')
		X = raw[:, 2:]
		y = get_normalized_labels(raw[:, 1])
		kmeans = KMeans(n_clusters=k)
		ind = kmeans.fit(X, y)
		figure()
		perm = permutation(X.shape[0])[:1000]
		X = X[perm]
		ind = ind[perm]
		colors = rand(k, 3)[ind, :]
		scatter(X[:, 0], X[:, 1], c=colors, alpha=0.9, s=30)
	show()