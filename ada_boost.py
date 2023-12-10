from common import get_data, test
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(n_estimators=100, random_state=0)

(X_train, X_test, y_train, y_test), label_encoder = get_data()

clf.fit(X_train, y_train)

preds = clf.predict(X_test)
test(preds, y_test, label_encoder)
