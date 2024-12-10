
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn import svm, metrics
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

defaults = pd.read_csv("defaults.csv", sep = ";")
defaults.head(10)

defaults.describe()
defaults.isnull().sum()[defaults.isnull().sum() > 0].count()
class_counts = defaults['default payment next month'].value_counts()
proportion = class_counts[1] / class_counts[0]
proportion

"""CZYSZCZENIE DANYCH:"""

defaults["SEX"] = defaults["SEX"].replace({2: 1, 1: 0})
defaults["EDUCATION"] = defaults["EDUCATION"].replace({5: 4, 6: 4})

columns_to_exclude = ['ID']
defaults_selected = defaults.drop(columns=columns_to_exclude)

defaults_selected.head(10)

"""STANDARYZACJA"""

num_cols = [
    "LIMIT_BAL", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

scaler = StandardScaler()

"""Ograniczenie zbioru do 2000 losowo wybranych rekordów."""

smlpDFl = defaults_selected.sample(n=2000)

"""PRZYGOTOWANIE DANYCH DO MODELOWANIA"""

X = smlpDFl.drop(columns=["default payment next month"])  # cechy
y = smlpDFl["default payment next month"]  # klasa decyzyjna

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print(X_train_scaled.shape, X_test_scaled.shape)

"""KLASYFIKATOR SVM"""

mdl = svm.SVC(C = 1, kernel = 'rbf')
resSVM = mdl.fit(X_train_scaled, y_train)

predictions = resSVM.predict(X_test_scaled)

print(metrics.confusion_matrix(y_test, predictions))
print(metrics.classification_report(y_test, predictions))

metrics.accuracy_score(y_test, predictions)

metrics.balanced_accuracy_score(y_test, predictions)

parameters = dict()
parameters["C"] = [0.1, 0.5, 1, 2, 5, 10, 50]
parameters["gamma"] = [0.01, 0.05, 0.1,  1, 10, 50]

cv = KFold(n_splits=10, shuffle=True)
mdl = svm.SVC()

gridMdl = GridSearchCV(mdl, parameters, scoring='accuracy', cv=cv, refit=True)
resMdls = gridMdl.fit(X_train_scaled, y_train)

bestMdl = resMdls.best_estimator_
bestMdl

best_predictions = bestMdl.predict(X_test_scaled)

print(bestMdl)
print(metrics.confusion_matrix(y_test, best_predictions))
print(metrics.classification_report(y_test, best_predictions))

"""KRZYWA ROC"""

RocCurveDisplay.from_estimator(bestMdl, X_test_scaled, y_test)
plt.show()

"""WALIDACJA"""

defaults_valid = pd.read_csv("defaults_valid.csv", sep = ";")

defaults_valid["SEX"] = defaults_valid["SEX"].replace({2: 1, 1: 0})
defaults_valid["EDUCATION"] = defaults_valid["EDUCATION"].replace({5: 4, 6: 4})

columns_to_exclude_v = ['ID']
defaults_selected_v = defaults_valid.drop(columns=columns_to_exclude_v)

defaults_selected_v.head(10)

X_valid = defaults_selected_v.drop(columns=["default payment next month"])
y_valid = defaults_selected_v["default payment next month"]


X_valid_scaled = scaler.fit_transform(X_valid)

valid_predictions = bestMdl.predict(X_valid_scaled)

print(metrics.confusion_matrix(y_valid, valid_predictions))
print(metrics.classification_report(y_valid, valid_predictions))
print(metrics.balanced_accuracy_score(y_valid, valid_predictions))

RocCurveDisplay.from_estimator(bestMdl, X_valid_scaled, y_valid)
plt.show()