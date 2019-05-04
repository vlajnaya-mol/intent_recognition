from bert_serving.client import BertClient
import numpy as np
from nltk import sent_tokenize
import pickle
import sys

bc = BertClient()
text = sys.argv[0]
sentences = sent_tokenize(text)
test_embeddings = bc.encode(sentences)

with open('models/ensemble_sorted.pickle', 'rb') as ensemble_file:
    classifiers = pickle.load(ensemble_file)

def predict(X_test, topn):
    ensemble = [tup[0] for tup in classifiers[:topn]]
    predictions = sum(classifier.predict(X_test) for classifier in ensemble)
    y_pred = np.array([1 if x > topn / 2 else 0 for x in predictions])
    print(predictions)
    print(y_pred)
    return y_pred


y_pred = predict(test_embeddings, 3)
for i in range(sentences):
    if y_pred[i] == 1:
        print(sentences[i])

