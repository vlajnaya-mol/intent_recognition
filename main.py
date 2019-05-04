# from bert_serving.client import BertClient
# import numpy as np
#
# bc = BertClient()
# print(bc.encode(['First do it', 'then do it right', 'then do it better']))

# with open('datasets/data.txt', 'r') as data:
#     with open('datasets/data_embedded.txt', 'w') as data_embedded:
#         i = 1
#         for line in data:
#             print(i)
#             i += 1
#             embedding = bc.encode([line[4:]])
#             intent = '1' if line[0] == 'Y' else '0'
#             print(intent, np.array_str(embedding, max_line_width=np.inf), file=data_embedded)

# with open('datasets/test.txt', 'r') as data:
#     with open('datasets/test_embedded.txt', 'w') as data_embedded:
#         i = 1
#         for line in data:
#             print(i)
#             i += 1
#             embedding = bc.encode([line[4:]])
#             intent = '1' if line[0] == 'Y' else '0'
#             print(intent, np.array_str(embedding, max_line_width=np.inf), file=data_embedded)
#
#
import numpy as np
from sklearn import datasets, linear_model
from scipy.stats import randint as sp_randint
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score, f1_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
import pickle

with open("datasets/another_data_embedded.txt", 'r') as f:
    lines = f.readlines()
    X = []
    y = []
    for line in lines:
        y.append(int(line[0]))
        emb = line[5:-3]
        X.append(np.fromstring(emb, dtype=float, sep=' '))

# param_dist = {"activation": ['tanh', 'relu', 'logistic']}
#
# random_search = GridSearchCV(classifier, param_grid=param_dist, cv=3)
#
# random_search.fit(X_train, y_train)
#
# results = random_search.cv_results_
# n_top = 10
# for i in range(1, n_top + 1):
#     candidates = np.flatnonzero(results['rank_test_score'] == i)
#     for candidate in candidates:
#         print("Model with rank: {0}".format(i))
#         print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
#             results['mean_test_score'][candidate],
#             results['std_test_score'][candidate]))
#         print("Parameters: {0}".format(results['params'][candidate]))
#         print("")


best_classifier = None
best_score = 0
for i in range(100):
    print(i)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
    classifier = MLPClassifier(max_iter=10000, hidden_layer_sizes=350)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    score = f1_score(y_test, y_pred) + accuracy_score(y_test, y_pred)
    print(score / 2)
    print(f"Mean squared error: {mean_squared_error(y_test, y_pred)}")
    print(f"F1 score: {f1_score(y_test, y_pred)}")
    print(f"Recall score: {recall_score(y_test, y_pred)}")
    print(f"Precision score: {precision_score(y_test, y_pred)}")
    print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")
    if score > best_score:
        best_classifier = classifier
        best_score = score
        with open('models/classifier.pickle', 'wb') as clf_file:
            pickle.dump(best_classifier, clf_file)
#
# classifiers = []
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
# for i in range(20):
#     print(i)
#     classifier = MLPClassifier(hidden_layer_sizes=350)
#     classifier.fit(X_train, y_train)
#     y_pred = classifier.predict(X_test)
#     score = f1_score(y_test, y_pred) + accuracy_score(y_test, y_pred)
#     print(score / 2)
#     classifiers.append((classifier, score))
#     with open('models/ensemble_all.pickle', 'wb') as ensemble_file:
#         pickle.dump(classifiers, ensemble_file)
#
# classifiers.sort(key=lambda tup: tup[1], reverse=True)
#
# def predict(X_test, topn):
#     ensemble = [c[0] for c in classifiers[:topn]]
#     predictions = sum(classifier.predict(X_test) for classifier in ensemble)
#     y_pred = np.array([1 if x > topn / 2 else 0 for x in predictions])
#     print(predictions)
#     print(y_pred)
#     print(f"Mean squared error: {mean_squared_error(y_test, y_pred)}")
#     print(f"F1 score: {f1_score(y_test, y_pred)}")
#     print(f"Recall score: {recall_score(y_test, y_pred)}")
#     print(f"Precision score: {precision_score(y_test, y_pred)}")
#     print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")
#     print('*****************************************')
#     return y_pred
#
# predict(X_test, 1)
# predict(X_test, 3)
# predict(X_test, 5)
# predict(X_test, 7)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
# with open('models/classifier.pickle', 'rb') as clf_file:
#     classifier = pickle.load(clf_file)
# y_pred = classifier.predict(X_test)


print(f"Mean squared error: {mean_squared_error(y_test, y_pred)}")
print(f"F1 score: {f1_score(y_test, y_pred)}")
print(f"Recall score: {recall_score(y_test, y_pred)}")
print(f"Precision score: {precision_score(y_test, y_pred)}")
print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")
#
# test_sentences = ['Please turn off the music', 'I will be there tomorrow', 'What the fuck is intent?']
# test_embeddings = bc.encode(test_sentences)
# print(*zip(test_sentences, classifier.predict(test_embeddings)))
#
# with open('datasets/all_embedded.txt', 'w') as all_embedded:
#     with open('datasets/data_embedded.txt', 'r') as train_embedded:
#         for line in train_embedded:
#             print(line, file=all_embedded)
#     with open('datasets/test_embedded.txt', 'r') as test_embedded:
#         for line in test_embedded:
#             print(line, file=all_embedded)
