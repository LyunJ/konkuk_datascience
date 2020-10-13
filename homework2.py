import mysql_conn
import pymysql
import pandas
from sqlalchemy import create_engine

engine = create_engine('mysql+pymysql://lyunj:Dldbswo77@@localhost:3306/konkuk_datascience',echo=False)
conn = engine.connect();

sql = 'select sno,midterm,final from db_score where midterm >= 20 and final >= 20 order by sno;'
data = pandas.read_sql_query(sql,conn)
print(data)