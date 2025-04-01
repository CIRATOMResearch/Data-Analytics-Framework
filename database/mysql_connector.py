import mysql.connector
from mysql.connector import Error
import pandas as pd


class MySQLConnector:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        
    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            return True
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            return False
            
    def execute_query(self, query):
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()
            
            df = pd.read_sql(query, self.connection)
            return df
        except Error as e:
            print(f"Error executing query: {e}")
            return None
            
    def close(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()