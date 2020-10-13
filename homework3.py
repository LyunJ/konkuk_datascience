import mysql_conn
import pymysql
import pandas
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

engine = create_engine(
    'mysql+pymysql://lyunj:Dldbswo77@@localhost:3306/konkuk_datascience', echo=False)
conn = engine.connect()

sql = 'select sno,attendance, homework, discussion, midterm, final, score from db_score where midterm >= 20 and final >= 20 order by sno;'
data = pandas.read_sql_query(sql, conn)
print("mean")
print(data.mean())
print("median")
print(data.median())

sql = 'select grade from db_score where midterm >= 20 and final >= 20 order by sno;'
grade = pandas.read_sql_query(sql, conn)
print("mode")
print(grade.mode())

print("variance")
print(data.var())
print("standard deviation")
print(data.std())
print("AAD")
print(data.mad())

print("precentile")
print(data.quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]))
data.quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]).plot()
plt.show()

print("boxplot")
data.boxplot()
plt.show()

sql = 'select * from db_score where midterm >= 20 and final >= 20 order by sno;'
hist = pandas.read_sql_query(sql, conn)
columns = ["sno", "attendance", "homework",
           "discussion", "midterm", "final", "score"]

print("histogram")
for col in columns:
    hist[col].plot.hist()
    plt.title(col, loc='center')
    plt.show()
hist["grade"].value_counts().plot(kind='bar', subplots=True)
plt.show()


print("scatter plot")
x_count = 0
y_count = 0

for x in columns:
    x_count += 1
    for y in columns:
        y_count += 1
        if y_count > x_count:
            data.plot.scatter(x, y)
            plt.show()
    y_count = 0
