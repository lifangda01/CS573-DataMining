from pylab import *
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, mutual_info_score
import time

class KMeans(object):
    """KMeans clustering"""
    def __init__(self, n_clusters=10, max_iter=50, debug=False):
        super(KMeans, self).__init__()
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.debug = debug
        self.WC_SSD = -1.
        self.SC = -1.
        self.NMI = -1.

    def fit(self, X, y=None):
        '''
            Compute k-means clustering.
            X is n_samples x n_features.
            y is n_samples.
        '''
        n_samples, n_features = X.shape
        # Randomly select K samples as initial centers
        perm = permutation(n_samples)[:self.n_clusters]
        C = X[perm]

        # Nearest neighbor function
        def nn(x):
            return argmin(norm(C - x, axis=1))
        # Iterate 50 times
        for i in xrange(50):
            # Assign points to nearest cluster centers
            NN = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(C)
            dist, ind = NN.kneighbors(X)
            ind = ind.flatten()
            # ind = apply_along_axis(nn, 1, X)
            # Update the cluster centers
            for k in range(self.n_clusters):
                C[k] = mean(X[ind == k], axis=0)
        t1 = time.time()
        # Compute the WC_SSD
        self.WC_SSD = sum((X - C[ind])**2)

        # Compute the SC
        def sc(x):
            '''
                Compute SC for a single sample.
            '''
            dist = norm(X - x[:-1], axis=1)
            nc_ind = argsort(norm(C - x[:-1], axis=1))[1]
            # x[-1] is encoded as the ind of x
            a = sum(dist[ind == x[-1]]) / (sum(ind == x[-1]) - 1)
            # b = mean(dist[ind != x[-1]])
            b = mean(dist[ind == nc_ind])
            return (b - a) / max(a, b)
        # Encode ind along with X
        X_enc = hstack((X, ind.reshape(-1, 1)))
        self.SC = mean(apply_along_axis(sc, 1, X_enc))
        t2 = time.time()
        # Compute the NMI
        if isinstance(y, ndarray):
            y = array(y, dtype=int)
            pc = unique(y, return_counts=True)[1]
            pc = pc * 1.0 / sum(pc)
            logpc = log(pc)
            n_classes = size(pc)
            pg = unique(ind, return_counts=True)[1]
            pg = pg * 1.0 / sum(pg)
            logpg = log(pg)
            # Compute the contingency table
            # pcg is n_clusters x n_classes
            pcg = bincount(self.n_clusters * ind + y,
                            minlength=self.n_clusters * n_classes) \
                            .reshape(self.n_clusters, n_classes)
            pcg = pcg * 1.0 / sum(pcg)
            numer = sum(pcg * log(pcg / pc / pg[:, None]))
            denom = - dot(pc, logpc) - dot(pg, logpg)
            self.NMI = numer / denom
        print "WC-SSD %.3f" % self.WC_SSD
        print "SC %.3f" % self.SC
        print "NMI %.3f" % self.NMI
        if self.debug: 
            print "sklearn SC %.3f" % silhouette_score(X, ind)
            if isinstance(y, ndarray): 
                print "sklearn NMI %.3f" % mutual_info_score(y, ind)
            print t2 - t1, time.time() - t2
        return ind

    def get_evals(self):
        '''
            Return WC_SSD, SC, NMI.
        '''
        return self.WC_SSD, self.SC, self.NMI

    def main():
        n_clusters = 10
        X = randn(1000, 2)
        y = randint(n_clusters, size=1000)
        kmeans = KMeans(n_clusters, debug=True)
        ind = kmeans.fit(X, y)
        colors = rand(n_clusters, 3)[ind, :]
        scatter(X[:, 0], X[:, 1], c=colors, alpha=0.9, s=30)
        show()     

if __name__ == '__main__':
    main()