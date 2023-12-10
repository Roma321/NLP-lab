import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel

num = 0


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


model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained('bert-base-uncased')


def get_data():
    data = pd.read_csv('bbc-news-data.csv', delimiter='\t')
    data['all_tokens'] = (data['title'] + data['content']).apply(lambda x: to_embedding(x))
    label_encoder = LabelEncoder()
    data['category'] = label_encoder.fit_transform(data['category'])
    labels = data['category'].to_list()
    vals = data['all_tokens'].to_list()
    vals = [emb.view(-1).numpy() for emb in vals]
    return train_test_split(vals, labels, test_size=0.20), label_encoder


def test(preds, y_test, label_encoder):
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
