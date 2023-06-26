import numpy as np
from src.NLP.BoW import BoW
from collections import Counter
import math
class TFIDF:
    def __init__(self,doc):
        self.doc = doc
        self.tokinize = BoW(doc).tokenized()
        self.vocab =  BoW(doc).vocabulary()
    def TF(self):
        tf_matrix = np.zeros((len(self.tokinize), len(self.vocab)))
        for i, doc in enumerate(self.tokinize):
            word_counts = Counter(doc)
            total_words = len(doc)
            for word in word_counts:
                tf = word_counts[word] / total_words
                tf_matrix[i, self.vocab.index(word)] = tf
        return tf_matrix
    def IDF(self):
        idf_vector = np.zeros(len(self.vocab))
        total_documents = len(self.tokinize)
        for i, word in enumerate(self.vocab):
            doc_with_word_count = sum([1 for doc in self.tokinize if word in doc])
            idf = math.log(total_documents / (doc_with_word_count + 1))  
            idf_vector[i] = idf
        return idf_vector
    def TFIDFvectorize(self):
        self.document = self.TF() * self.IDF()
        return self.document