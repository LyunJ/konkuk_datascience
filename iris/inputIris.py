import pymysql
import pandas
from sqlalchemy import create_engine

csvfile = './iris/iris.csv'
df = pandas.read_csv(csvfile)
conn = create_engine(
    'mysql+pymysql://lyunj:Dldbswo77@@localhost:3306/konkuk_datascience', echo=False)
df.to_sql(name='iris', con=conn, if_exists='append', index=False)
