from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import get_squashed_data, test, get_data

(X_train, X_test, y_train, y_test), label_encoder = get_data()

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
test(preds, y_test, label_encoder)
