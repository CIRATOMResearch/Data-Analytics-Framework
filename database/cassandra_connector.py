from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import pandas as pd


class CassandraConnector:
    def __init__(self, contact_points, port=9042, username=None, password=None):
        self.contact_points = contact_points
        self.port = port
        self.username = username
        self.password = password
        self.session = None
        self.cluster = None
        
    def connect(self):
        try:
            if self.username and self.password:
                auth_provider = PlainTextAuthProvider(
                    username=self.username,
                    password=self.password
                )
                self.cluster = Cluster(
                    contact_points=self.contact_points,
                    port=self.port,
                    auth_provider=auth_provider
                )
            else:
                self.cluster = Cluster(
                    contact_points=self.contact_points,
                    port=self.port
                )
            
            self.session = self.cluster.connect()
            return True
        except Exception as e:
            print(f"Error connecting to Cassandra: {e}")
            return False
            
    def execute_query(self, keyspace, query):
        try:
            if not self.session:
                self.connect()
            
            self.session.set_keyspace(keyspace)
            rows = self.session.execute(query)
            df = pd.DataFrame(list(rows))
            return df
        except Exception as e:
            print(f"Error executing query: {e}")
            return None
            
    def close(self):
        if self.session:
            self.session.shutdown()
        if self.cluster:
            self.cluster.shutdown()