import numpy as np
from sklearn import datasets, linear_model
from scipy.stats import randint as sp_randint
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score, f1_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
import pickle

with open("datasets/test_embedded.txt", 'r') as f:
    lines = f.readlines()
    X = []
    y = []
    for line in lines:
        y.append(int(line[0]))
        emb = line[5:-3]
        X.append(np.fromstring(emb, dtype=float, sep=' '))

with open('models/ensemble_sorted2.pickle', 'rb') as ensemble_file:
    classifiers = pickle.load(ensemble_file)

#
# sorted_classifiers = sorted(classifiers, key=lambda tup: tup[1], reverse=True)
# with open('models/ensemble_sorted2.pickle', 'wb') as ensemble_file:
#     pickle.dump(sorted_classifiers[:9], ensemble_file)

def predict(X_test, topn):
    ensemble = [c[0] for c in classifiers[:topn]]
    predictions = sum(classifier.predict(X_test) for classifier in ensemble)
    y_pred = np.array([1 if x > topn / 2 else 0 for x in predictions])
    print(predictions)
    print(y_pred)
    print(f"Mean squared error: {mean_squared_error(y, y_pred)}")
    print(f"F1 score: {f1_score(y, y_pred)}")
    print(f"Recall score: {recall_score(y, y_pred)}")
    print(f"Precision score: {precision_score(y, y_pred)}")
    print(f"Accuracy score: {accuracy_score(y, y_pred)}")
    return y_pred


predict(X, 1)
predict(X, 3)
predict(X, 5)
predict(X, 7)
predict(X, 9)