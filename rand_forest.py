from sklearn.ensemble import RandomForestClassifier
from common import get_squashed_data, test

(X_train, X_test, y_train, y_test), label_encoder = get_squashed_data()

clf = RandomForestClassifier(max_depth=10, random_state=0)
clf.fit(X_train, y_train)
preds = clf.predict(y_test)
test(preds, y_test, label_encoder)
