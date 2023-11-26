import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel


def get_metric(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


def to_embedding(x):
    global num
    print(num)
    num += 1
    tokens = tokenizer.encode(str(x))
    tokens = tokens[:512]  # Truncate the input to maximum allowed length
    input_ids = torch.tensor(tokens).unsqueeze(0)
    input_ids = torch.nn.functional.pad(input_ids, (0, 512 - len(tokens)), 'constant',
                                        0)  # Pad or truncate input_ids to length 512

    with torch.no_grad():
        outputs = model(input_ids)
        embeddings = outputs[0][0]  # Extract the embeddings from the output
    return embeddings


num = 0

data = pd.read_csv('bbc-news-data.csv', delimiter='\t')
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained('bert-base-uncased')
data['all_tokens'] = (data['title'] + data['content']).apply(lambda x: to_embedding(x))
label_encoder = LabelEncoder()
data['category'] = label_encoder.fit_transform(data['category'])
labels = data['category'].to_list()
vals = data['all_tokens'].to_list()
vals = [emb.view(-1).numpy() for emb in vals]
X_train, X_test, y_train, y_test = train_test_split(vals, labels, test_size=0.20)


def get_nearest(vec):
    global num
    print("enter get_nearest")
    print(num)
    num += 1
    res_vecs = [{
        'dist': 999999999999999999,
        'label': -1
    }] * 5
    for train_vec, train_label in zip(X_train, y_train):
        dist = get_metric(vec, train_vec)
        if dist < res_vecs[-1]['dist']:
            res_vecs.append({
                'dist': dist,
                'label': train_label
            })
            res_vecs = sorted(res_vecs, key=lambda x: x['dist'])
            res_vecs = res_vecs[:-1]
    lbls = [item['label'] for item in res_vecs]
    label_counts = Counter(lbls)
    most_common_label = label_counts.most_common(1)[0][0]
    return most_common_label


res = {
    'business': {
        'ok': 0,
        'bad': 0,
    },
    'entertainment': {
        'ok': 0,
        'bad': 0,
    },
    'politics': {
        'ok': 0,
        'bad': 0,
    },
    'sport': {
        'ok': 0,
        'bad': 0,
    },
    'tech': {
        'ok': 0,
        'bad': 0,
    }
}
ok = 0
bad = 0
matrix = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
t = time.time()
preds = Parallel(n_jobs=16)(delayed(get_nearest)(X) for X in X_test)
print(time.time() - t)
for pred, should_be in zip(preds, y_test):
    if pred == should_be:
        res[label_encoder.classes_[should_be]]['ok'] += 1
        ok += 1
    else:
        res[label_encoder.classes_[should_be]]['bad'] += 1
        bad += 1
    matrix[should_be][pred] += 1

print(label_encoder.classes_)
print(res)
print(matrix)
print(ok / (ok + bad))