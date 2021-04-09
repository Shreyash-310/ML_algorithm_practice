import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

dataset = pd.read_csv("iphone_purchase_records.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

labelEncoder_gender = LabelEncoder()
X[:, 0] = labelEncoder_gender.fit_transform(X[:, 0])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
print(precision)
print(recall)
