import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymysql
import time


def load_dbscore_data():
    conn = pymysql.connect(host='localhost', user='lyunj',
                           password='Dldbswo77@', db='konkuk_datascience')
    curs = conn.cursor(pymysql.cursors.DictCursor)

    sql = "select * from db_score"
    curs.execute(sql)

    data = curs.fetchall()

    curs.close()
    conn.close()

    X = [(t['midterm']) for t in data]

    X = np.array(X)

    y = [(t['score'])for t in data]
    y = np.array(y)

    return X, y


X, y = load_dbscore_data()
'''
plt.scatter(X, y)
plt.show()
'''
# y = mx + c

X_const = sm.add_constant(X)

model = sm.OLS(y, X_const)
ls = model.fit()

print(ls.summary())


ls_c = ls.params[0]
ls_m = ls.params[1]

y_pred = ls_m*X + ls_c
plt.scatter(X, y)
plt.plot([min(X), max(X)], [min(y_pred), max(y_pred)], color='red')
plt.show()


def gradient_descent_naive(X, y):
    epochs = 100000
    min_grad = 0.0001
    learning_rate = 0.001

    m = 0.0
    c = 0.0

    n = len(y)

    c_grad = 0.0
    m_grad = 0.0

    for epoch in range(epochs):

        for i in range(n):
            y_pred = m*X[i] + c
            m_grad += 2*(y_pred-y[i]) * X[i]
            c_grad += 2*(y_pred - y[i])

        c_grad /= n
        m_grad /= n

        m = m - learning_rate * m_grad
        c = c - learning_rate * c_grad

        if(epoch % 1000 == 0):
            print("epoch %d: m_grad=%f, c_grad=%f, m%f, c=%f" %
                  (epoch, m_grad, c_grad, m, c))
        if(abs(m_grad) < min_grad and abs(c_grad) < min_grad):
            break

    return m, c


'''
start_time = time.time()
m, c = gradient_descent_naive(X, y)
end_time = time.time()

print("%f seconds" % (end_time - start_time))

print("\n\nFinal:")
print("gdn_m=%f, gdn_c=%f" % (m, c))
print("ls_m=%f, ls_c=%f" % (ls_m, ls_c))
'''


def gradient_descent_vectorized(X, y):
    epochs = 100000
    min_grad = 0.0001
    learning_rate = 0.001

    m = 0.0
    c = 0.0

    n = len(y)

    c_grad = 0.0
    m_grad = 0.0

    for epoch in range(epochs):
        y_pred = m * X + c
        m_grad = (2*(y_pred - y)*X).sum()/n
        c_grad = (2*(y_pred - y)).sum()/n

        m = m - learning_rate * m_grad
        c = c - learning_rate * c_grad

        if(epoch % 1000 == 0):
            print("epoch %d: m_grad=%f, c_grad=%f, m%f, c=%f" %
                  (epoch, m_grad, c_grad, m, c))
        if(abs(m_grad) < min_grad and abs(c_grad) < min_grad):
            break

    return m, c


start_time = time.time()
m, c = gradient_descent_naive(X, y)
end_time = time.time()

print("%f seconds" % (end_time - start_time))

print("\n\nFinal:")
print("gdv_m=%f, gdv_c=%f" % (m, c))
print("ls_m=%f, ls_c=%f" % (ls_m, ls_c))
