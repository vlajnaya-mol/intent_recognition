from bert_serving.client import BertClient
import numpy as np

bc = BertClient()
with open('datasets/new_test.txt', 'r') as data:
    with open('datasets/new_test_embedded.txt', 'w') as data_embedded:
        i = 1
        for line in data:
            print(i)
            i += 1
            embedding = bc.encode([line[2]])
            intent = line[0]
            print(intent, np.array_str(embedding, max_line_width=np.inf), file=data_embedded)