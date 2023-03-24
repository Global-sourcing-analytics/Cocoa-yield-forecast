# -*- coding: utf-8 -*-
"""
Created on Mon Feb  13 11:10:09 2023

@author: BC127735
"""
import pyodbc
import pandas as pd

from sqlalchemy import create_engine, event, text
from sqlalchemy.pool import StaticPool


class DatabaseManager:
    def __init__(self, str_server, str_database, str_uid = "", str_pwd = "", trusted_connection=True):
        # connection to the database
        print(str_uid, str_pwd)
        if trusted_connection and ((str_uid == "" and str_pwd == "") or (str_uid == None and str_pwd == None)):
            self.conn = pyodbc.connect('driver={ODBC Driver 17 for SQL Server};server=%s;database=%s;Trusted_Connection=yes;' % \
                                ( str_server, str_database ))
        else:
            self.conn = pyodbc.connect('driver={ODBC Driver 17 for SQL Server};server=%s;database=%s;uid=%s;pwd=%s' % \
                                ( str_server, str_database, str_uid, str_pwd ))
        self.engine = create_engine("mssql+pyodbc://", poolclass=StaticPool, creator=lambda: self.conn)

        # %% Uploading to database
        @event.listens_for(self.engine, 'before_cursor_execute')
        def receive_before_cursor_execute(cursor, statement, params, context, executemany, conn=self.conn):
            if executemany:
                cursor.fast_executemany = True
        
        print("Connection with database established.")
    
    def upload_data(self, obj_data, str_table_name, str_schema):
        if len(obj_data) > 0:
            obj_data.to_sql(name=str_table_name,
                            con=self.engine,
                            schema=str_schema,
                            index=False,
                            if_exists='append')

        self.conn.commit()
    
    def run_query(self, str_query):
        self.conn.execute(str_query)
        self.conn.commit()

    def get_data(self, str_query):
        data = pd.read_sql(text(str_query), self.engine.connect())
        return data

    def get_engine(self):
        return self.engine

if __name__ == "__main__":
    print("You can't run the module itself!")