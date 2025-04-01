import polars as pl
import pyarrow
import pandas as pd


class AutoCleaning:
    def __init__(self, df):
        self.df = df


    def clean_data(self):
        for col in self.df.columns:
            if self.df[col].dtype == pl.Utf8:
                self.df = self.df.with_columns(pl.col(col).str.replace_all(r"[^\w\s-]", ""))

        print("Cleaned DataFrame:")
        print(self.df)

        df_new = pl.DataFrame(self.df)
        df_new.write_csv('sample1.csv')


if __name__ == "__main__":
    df_polars = pl.read_csv("sample1.csv")

    auto_data_cleaning = AutoCleaning(df_polars)
    auto_data_cleaning.clean_data()