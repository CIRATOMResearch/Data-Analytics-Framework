import psycopg2
from psycopg2 import Error
import pandas as pd


class PostgreSQLConnector:
    def __init__(self, host, user, password, database, port=5432):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        self.connection = None
        
    def connect(self):
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                port=self.port
            )
            return True
        except Error as e:
            print(f"Error connecting to PostgreSQL: {e}")
            return False
            
    def execute_query(self, query):
        try:
            if not self.connection:
                self.connect()
            
            df = pd.read_sql(query, self.connection)
            return df
        except Error as e:
            print(f"Error executing query: {e}")
            return None
            
    def close(self):
        if self.connection:
            self.connection.close()