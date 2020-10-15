
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import statistics
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pymysql
import numpy as np
import pandas
from sqlalchemy import create_engine


def classification_performance_eval(y, y_predict):

    matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    for y, p in zip(y, y_predict):
        if p == 'A':
            if y == 'A':
                matrix[0][0] += 1
            if y == 'B':
                matrix[0][1] += 1
            if y == 'C':
                matrix[0][2] += 1
        if p == 'B':
            if y == 'A':
                matrix[1][0] += 1
            if y == 'B':
                matrix[1][1] += 1
            if y == 'C':
                matrix[1][2] += 1
        if p == 'C':
            if y == 'A':
                matrix[2][0] += 1
            if y == 'B':
                matrix[2][1] += 1
            if y == 'C':
                matrix[2][2] += 1

    matrix = np.array(matrix)
    sums = matrix.sum()

    tp, tn, fp, fn = 0, 0, 0, 0
    accuracy = []
    precision = []
    recall = []
    f1_score = []
    for i in range(3):
        tp = matrix[i][i]
        for j in range(3):
            fp += matrix[i][j] if (i != j) else 0
            fn += matrix[j][i] if (i != j) else 0
        tn = sums - tp - fp - fn
        acc = (tp + tn)/(tp + tn + fp + fn)
        pre = (tp)/(tp + fp)
        rec = (tp)/(tp + fn)
        f1 = 2*pre*rec/(pre+rec)
        accuracy.append(acc)
        precision.append(pre)
        recall.append(rec)
        f1_score.append(f1)

    return statistics.mean(accuracy), statistics.mean(precision), statistics.mean(recall), statistics.mean(f1_score)


conn = pymysql.connect(host='localhost', user='lyunj',
                       password='Dldbswo77@', db='konkuk_datascience', charset='utf8')
curs = conn.cursor(pymysql.cursors.DictCursor)

sql = "select * from db_score_3"
curs.execute(sql)

data = curs.fetchall()

curs.close()
conn.close()


X = [(t['homework'], t['discussion'], t['midterm']) for t in data]
X = np.array(X)

y = [t['grade'] for t in data]
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

clf = svm.SVC(kernel='rbf', C=1, gamma=10)
clf.fit(X_train_std, y_train)
y_predict = clf.predict(X_test_std)

acc, prec, rec, f1 = classification_performance_eval(y_test, y_predict)

print("svm_accuracy=%f" % acc)
print("svm_precision=%f" % prec)
print("svm_recall=%f" % rec)
print("svm_f1_score=%f" % f1)

LR = LogisticRegression(C=100, random_state=0)
LR.fit(X_train_std, y_train)
y_predict = LR.predict(X_test_std)

acc, prec, rec, f1 = classification_performance_eval(
    y_test, y_predict)

print("LR_accuracy=%f" % acc)
print("LR_precision=%f" % prec)
print("LR_recall=%f" % rec)
print("LR_f1_score=%f" % f1)


RF = RandomForestClassifier(
    criterion='entropy', n_estimators=10, n_jobs=2, random_state=1)
RF.fit(X_train_std, y_train)
y_predict = RF.predict(X_test_std)
acc, prec, rec, f1 = classification_performance_eval(
    y_test, y_predict)

print("RF_accuracy=%f" % acc)
print("RF_precision=%f" % prec)
print("RF_recall=%f" % rec)
print("RF_f1_score=%f" % f1)

kf = KFold(n_splits=4, random_state=42, shuffle=True)

accuracy = []
precision = []
recall = []
f1_score = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    sc = StandardScaler()
    sc.fit(X_train)

    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    clf = svm.SVC(kernel='rbf', C=1, gamma=10)
    clf.fit(X_train_std, y_train)
    y_predict = clf.predict(X_test_std)

    acc, prec, rec, f1 = classification_performance_eval(y_test, y_predict)
    accuracy.append(acc)
    precision.append(prec)
    recall.append(rec)
    f1_score.append(f1)

print("svm_average_accuracy =", statistics.mean(accuracy))
print("svm_average_precision =", statistics.mean(precision))
print("svm_average_recall =", statistics.mean(recall))
print("svm_average_f1_score =", statistics.mean(f1_score))

kf = KFold(n_splits=4, random_state=42, shuffle=True)

accuracy = []
precision = []
recall = []
f1_score = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    sc = StandardScaler()
    sc.fit(X_train)

    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    LR = LogisticRegression(C=100, random_state=0)
    LR.fit(X_train_std, y_train)
    y_predict = LR.predict(X_test_std)
    acc, prec, rec, f1 = classification_performance_eval(
        y_test, y_predict)
    accuracy.append(acc)
    precision.append(prec)
    recall.append(rec)
    f1_score.append(f1)

print("LR_average_accuracy =", statistics.mean(accuracy))
print("LR_average_precision =", statistics.mean(precision))
print("LR_average_recall =", statistics.mean(recall))
print("LR_average_f1_score =", statistics.mean(f1_score))

kf = KFold(n_splits=4, random_state=42, shuffle=True)

accuracy = []
precision = []
recall = []
f1_score = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    sc = StandardScaler()
    sc.fit(X_train)

    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    RF = RandomForestClassifier(
        criterion='entropy', n_estimators=10, n_jobs=2, random_state=1)
    RF.fit(X_train_std, y_train)
    y_predict = RF.predict(X_test_std)
    acc, prec, rec, f1 = classification_performance_eval(
        y_test, y_predict)
    accuracy.append(acc)
    precision.append(prec)
    recall.append(rec)
    f1_score.append(f1)

print("RF_average_accuracy =", statistics.mean(accuracy))
print("RF_average_precision =", statistics.mean(precision))
print("RF_average_recall =", statistics.mean(recall))
print("RF_average_f1_score =", statistics.mean(f1_score))
