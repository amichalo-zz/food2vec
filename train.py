import logging
from gensim.models import Word2Vec
from utils import saveWord2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('food2vec')

class Receipts(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename, 'r').read().split('\n'):
            yield line.split(',')[1:]


sentences = Receipts('receipts_data')
food2vec = Word2Vec(sentences, min_count=10, size=64, iter=100, workers=6, sg=0, window=20, negative=2)

logger.info('Trying to save the model...')

saveWord2Vec('food2vec.model.txt', food2vec)