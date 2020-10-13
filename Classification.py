
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
    tp, tn, fp, fn = 0, 0, 0, 0
    for y, yp in zip(y, y_predict):
        if y == 1 and yp == 1:
            tp += 1
        elif y == 1 and yp == -1:
            fn += 1
        elif y == -1 and yp == 1:
            fp += 1
        else:
            tn += 1

    accuracy = (tp + tn)/(tp + tn + fp + fn)
    precision = (tp)/(tp + fp)
    recall = (tp)/(tp + fn)
    f1_score = 2*precision*recall/(precision+recall)

    return accuracy, precision, recall, f1_score


# xlsxfile = 'db_score_3_labels.xlsx'
# df = pandas.read_excel(xlsxfile)
# conn = create_engine(
#     'mysql+pymysql://lyunj:Dldbswo77@@localhost:3306/konkuk_datascience', echo=False)
# df.to_sql(name='db_score_3', con=conn, if_exists='append', index=False)

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

y = [1 if (t['grade'] == 'B') else -1 for t in data]
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42)

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

LR_y_train = [0 if (t == -1) else 1 for t in y_train]
LR_y_train = np.array(LR_y_train)
LR_y_test = [0 if (t == -1) else 1 for t in y_test]
LR_y_test = np.array(LR_y_test)

LR = LogisticRegression(C=100, random_state=0)
LR.fit(X_train_std, LR_y_train)
y_predict = LR.predict(X_test_std)

acc, prec, rec, f1 = classification_performance_eval(LR_y_test, y_predict)

print("LR_accuracy=%f" % acc)
print("LR_precision=%f" % prec)
print("LR_recall=%f" % rec)
print("LR_f1_score=%f" % f1)

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

    LR_y_train = [0 if (t == -1) else 1 for t in y_train]
    LR_y_train = np.array(LR_y_train)
    LR_y_test = [0 if (t == -1) else 1 for t in y_test]
    LR_y_test = np.array(LR_y_test)

    LR = LogisticRegression(C=1, random_state=0)
    LR.fit(X_train_std, LR_y_train)
    y_predict = LR.predict(X_test_std)
    acc, prec, rec, f1 = classification_performance_eval(LR_y_test, y_predict)
    accuracy.append(acc)
    precision.append(prec)
    recall.append(rec)
    f1_score.append(f1)

print("LR_average_accuracy =", statistics.mean(accuracy))
print("LR_average_precision =", statistics.mean(precision))
print("LR_average_recall =", statistics.mean(recall))
print("LR_average_f1_score =", statistics.mean(f1_score))
