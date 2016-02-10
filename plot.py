import sys, logging
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from utils import loadWord2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('food2vec')

filename = 'food2vec.model.txt'

food2vec = loadWord2Vec(filename)

labels = set(food2vec.index2word)
vectors = food2vec.syn0

logger.info("Preparing tsne transformation...")

tsne = TSNE(perplexity=15, n_components=2, init='pca', n_iter=4000, early_exaggeration=8.0)
vectors2d = tsne.fit_transform(vectors)

logger.info('Trying to plot food2vec results...')

plt.figure(figsize=(15, 15))
for i, label in enumerate(labels):
	x, y = vectors2d[i,:]
	plt.scatter(x, y)
	plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
plt.savefig('tsne.png')