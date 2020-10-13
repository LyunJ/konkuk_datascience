import pymysql
def connect_mysql():
    return pymysql.connect(host='localhost',user='lyunj',password='Dldbswo77@',db='konkuk_datascience',charset='utf8')

def insert_table(query):
    try:
        conn = connect_mysql()
        with conn.cursor() as curs:
            sql = query
            curs.execute(sql)
        conn.commit() 
    finally:
        conn.close()      