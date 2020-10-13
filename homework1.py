import mysql_conn
import pymysql
import pandas
from sqlalchemy import create_engine


mysql_conn.insert_table('CREATE TABLE db_score(`sno`  INT   NOT NULL  AUTO_INCREMENT,`attendance`  DOUBLE  NULL,`homework`    DOUBLE     NULL,`discussion`  DOUBLE     NULL,`midterm`     DOUBLE     NULL,`final`       DOUBLE     NULL,`score`       DOUBLE     NULL,`grade`       CHAR(1)    NULL,    PRIMARY KEY (sno));')

xlfile = 'db_score.xlsx'
df=pandas.read_excel(xlfile)
conn = create_engine('mysql+pymysql://lyunj:Dldbswo77@@localhost:3306/konkuk_datascience',echo=False)
df.to_sql(name='db_score',con=conn,if_exists='append',index=False)