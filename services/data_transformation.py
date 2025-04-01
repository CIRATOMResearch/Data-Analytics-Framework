import pandas as pd


def convert_to_datetime(date_series):
    return pd.to_datetime(date_series, errors='coerce')

a = convert_to_datetime('...')
print(a)