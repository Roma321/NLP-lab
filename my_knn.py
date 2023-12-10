import time
from collections import Counter

import numpy as np
from joblib import Parallel, delayed

from common import get_data, test


def get_metric(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


num = 0


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


(X_train, X_test, y_train, y_test), label_encoder = get_data()

preds = Parallel(n_jobs=16)(delayed(get_nearest)(X) for X in X_test)

test(preds, y_test, label_encoder)
