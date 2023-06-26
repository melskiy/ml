import numpy as np
class BoW:
    def __init__(self,doc):
        self.doc = doc
    def tokenized(self):
        return  [doc.lower().split() for doc in self.doc]

    def vocabulary(self):
        vocab = set()
        docs = self.tokenized()
        for doc in docs:
            vocab.update(doc)
        return sorted(list(vocab))
    def CountVectorized(self):
        tokenized_documents = self.tokenized()
        vocab = self.vocabulary()
        word_count_matrix = np.zeros((len(tokenized_documents), len(vocab)))
        for i, doc in enumerate(tokenized_documents):
            for word in doc:
                word_count_matrix[i, vocab.index(word)] += 1
        return word_count_matrix