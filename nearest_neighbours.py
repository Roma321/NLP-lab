from transformers import BertTokenizer, BertModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import torch
import numpy as np
from common import test


def to_ids(x):
    global num
    print(num)
    num += 1
    tokenized_text = tokenizer.tokenize(x)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    return indexed_tokens


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
mode = 'embeddings'

data = pd.read_csv('bbc-news-data.csv', delimiter='\t')
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained('bert-base-uncased')

label_encoder = LabelEncoder()
if mode == 'ids':
    data['all_tokens'] = (data['title'] + data['content']).apply(lambda x: to_ids(x))
if mode == 'embeddings':
    data['all_tokens'] = (data['title'] + data['content']).apply(lambda x: to_embedding(x))
data['category'] = label_encoder.fit_transform(data['category'])

vals = data['all_tokens'].to_list()
labels = data['category'].to_list()

if mode == 'embeddings':
    vals = [emb.view(-1).numpy().astype(np.float16) for emb in vals]
if mode == 'ids':
    max_len = max(len(x) for x in vals)

    # vals = [[x[i] if i < len(x) else 0 for i in range(max_len)] for x in vals]  # 24%
    # vals = [[x[i % len(x)] for i in range(max_len)][:200] for x in vals] # 23%
    # vals = [[x[i % len(x)] for i in range(max_len)] for x in vals]  # 23%
    # vals = [[x[i % len(x)] for i in range(max_len)] for x in vals] # 23%
    vals = [sorted([x[i] if i < len(x) else 0 for i in range(max_len)]) for x in vals]  # 37%
    # vals = [sorted([x[i % len(x)] for i in range(max_len)]) for x in vals]  # 32%

X_train, X_test, y_train, y_test = train_test_split(vals, labels, test_size=0.20)

classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(X_train, y_train)

preds = classifier.predict(X_test)

test(preds, y_test, label_encoder)
