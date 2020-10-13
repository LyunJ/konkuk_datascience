import statistics
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.model_selection import train_test_split
import pymysql
import numpy as np


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


conn = pymysql.connect(host='localhost', user='lyunj',
                       password='Dldbswo77@', db='konkuk_datascience', charset='utf8')
curs = conn.cursor(pymysql.cursors.DictCursor)

sql = "select * from iris"
curs.execute(sql)

data = curs.fetchall()

curs.close()
conn.close()

X = [(t['sepal_length'], t['sepal_width'], t['petal_length'], t['petal_width'])
     for t in data]
X = np.array(X)

y = [1 if (t['species'] == 'versicolor') else -1 for t in data]
y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

dtree = tree.DecisionTreeClassifier()

dtree_model = dtree.fit(X_train, y_train)

y_predict = dtree_model.predict(X_test)

acc, prec, rec, f1 = classification_performance_eval(y_test, y_predict)

print("accuracy=%f" % acc)
print("precision=%f" % prec)
print("recall=%f" % rec)
print("f1_score=%f" % f1)

kf = KFold(n_splits=5, random_state=42, shuffle=True)

accuracy = []
precision = []
recall = []
f1_score = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    dtree = tree.DecisionTreeClassifier()
    dtree_model = dtree.fit(X_train, y_train)
    y_predict = dtree_model.predict(X_test)
    acc, prec, rec, f1 = classification_performance_eval(y_test, y_predict)
    accuracy.append(acc)
    precision.append(prec)
    recall.append(rec)
    f1_score.append(f1)


print("average_accuracy =", statistics.mean(accuracy))
print("average_precision =", statistics.mean(precision))
print("average_recall =", statistics.mean(recall))
print("average_f1_score =", statistics.mean(f1_score))
