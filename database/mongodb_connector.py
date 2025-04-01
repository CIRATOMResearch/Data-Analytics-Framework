from pymongo import MongoClient
import pandas as pd


class MongoDBConnector:
    def __init__(self, host, port=27017, username=None, password=None):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.client = None
        
    def connect(self):
        try:
            if self.username and self.password:
                connection_string = f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}"
            else:
                connection_string = f"mongodb://{self.host}:{self.port}"
            
            self.client = MongoClient(connection_string)
            return True
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            return False
            
    def get_collection_data(self, database, collection):
        try:
            if not self.client:
                self.connect()
            
            db = self.client[database]
            collection = db[collection]
            data = list(collection.find())
            df = pd.DataFrame(data)
            
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
                
            return df
        except Exception as e:
            print(f"Error getting collection data: {e}")
            return None
            
    def close(self):
        if self.client:
            self.client.close()