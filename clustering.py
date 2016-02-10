import sys, logging
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from utils import loadWord2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('food2vec')

filename = 'food2vec.model.txt'

food2vec = loadWord2Vec(filename)

vectors = food2vec.syn0
clustersNo = 10

logger.info("Preparing clusters...")

kmeans = KMeans( n_clusters = clustersNo )
idx = kmeans.fit_predict( vectors )

logger.info("Clusters are ready!")

wordMap = dict(zip( food2vec.index2word, idx ))

for cluster in xrange(0,clustersNo):
    print "\nCluster %d" % cluster
    words = []
    for i in xrange(0,len(wordMap.values())):
        if( wordMap.values()[i] == cluster ):
            words.append(wordMap.keys()[i])
    print words
