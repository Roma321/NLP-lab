from sklearn.naive_bayes import GaussianNB
from common import get_squashed_data, test

(X_train, X_test, y_train, y_test), label_encoder = get_squashed_data()

clf = GaussianNB()
clf.fit(X_train, y_train)
preds = clf.predict(y_test)
test(preds, y_test, label_encoder)
