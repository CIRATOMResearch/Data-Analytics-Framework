from category_encoders import *
from category_encoders.helmert import HelmertEncoder
from category_encoders.backward_difference import BackwardDifferenceEncoder
from category_encoders.woe import WOEEncoder
from category_encoders.sum_coding import SumEncoder
from category_encoders.m_estimate import MEstimateEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.cat_boost import CatBoostEncoder
from category_encoders.james_stein import JamesSteinEncoder

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, TargetEncoder
import polars as pl

import matplotlib.pyplot as plt



class AutoCategoricalEncoders:
    def __init__(self, df):
        self.df = df


    def autoselection_method(self):
        """Автоматически выбирает и применяет наиболее подходящие методы кодирования."""
        # Используем LazyFrame для фильтрации
        categorical_columns = [col for col in self.df.columns if self.df[col].dtype in [pl.Utf8, pl.Categorical]]
        
        if not categorical_columns:
            print("В наборе данных не найдено категориальных столбцов.")
            return self.df
        
        encoding_decisions = {}
        
        for col in categorical_columns:
            unique_count = self.df[col].n_unique()
            total_rows = self.df.shape[0]
            min_value_count = self.df[col].value_counts().min()
            cardinality_ratio = unique_count / total_rows
            
            # Улучшенная логика принятия решений для выбора метода кодирования
            if unique_count == total_rows:
                encoding_decisions[col] = {
                    'method': 'label',
                    'reason': 'Все значения уникальны'
                }
                
            elif unique_count == 2:
                encoding_decisions[col] = {
                    'method': 'binary',
                    'reason': 'Бинарная категориальная переменная'
                }

            elif unique_count <= 10 and len(categorical_columns) <= 10:
                encoding_decisions[col] = {
                    'method': 'onehot',
                    'reason': 'Небольшое количество категорий и столбцов'
                }

            elif unique_count > 30 and min_value_count > 10:
                if cardinality_ratio > 0.5:
                    encoding_decisions[col] = {
                        'method': 'catboost',
                        'reason': 'Высокая мощность с достаточным количеством выборок'
                    }

                else:
                    encoding_decisions[col] = {
                        'method': 'target',
                        'reason': 'Множество категорий с достаточным количеством образцов'
                    }

            elif 10 < unique_count <= 30:
                encoding_decisions[col] = {
                    'method': 'helmert',
                    'reason': 'Среднее количество категорий'
                }

            else:
                encoding_decisions[col] = {
                    'method': 'ordinal',
                    'reason': 'Выбор по умолчанию для остальных случаев'
                }
        
        return self.apply_encodings(encoding_decisions)


    def apply_encodings(self, encoding_decisions):
        """Применяет выбранные методы кодирования к кадру данных."""
        df_encoded = self.df.clone()
        encoding_stats = {}
        
        try:
            # Группировать столбцы по методу кодирования
            method_groups = {}
            for col, decision in encoding_decisions.items():
                method = decision['method']
                if method not in method_groups:
                    method_groups[method] = []
                method_groups[method].append(col)
            
            # Применить кодировки для каждой группы методов
            for method, columns in method_groups.items():
                print(f"\nApplying {method} encoding to columns: {columns}")
                
                if method == 'binary':
                    encoder = BinaryEncoder(cols=columns)
                    encoded_data = encoder.fit_transform(df_encoded[columns])
                    df_encoded = df_encoded.drop(columns).hstack(pl.DataFrame(encoded_data))
                    
                elif method == 'onehot':
                    encoded_data = pl.get_dummies(df_encoded.select(columns))
                    df_encoded = df_encoded.drop(columns).hstack(encoded_data)
                    
                elif method == 'label':
                    for col in columns:
                        df_encoded = df_encoded.with_columns(
                            pl.col(col).cast(pl.Int32).alias(col)
                        )
                        
                elif method == 'ordinal':
                    encoder = OrdinalEncoder()
                    encoded_data = encoder.fit_transform(df_encoded[columns])
                    for i, col in enumerate(columns):
                        df_encoded = df_encoded.with_columns(
                            pl.Series(name=col, values=encoded_data[:, i])
                        )
                        
                elif method == 'target':
                    encoder = TargetEncoder()
                    for col in columns:
                        encoded_data = encoder.fit_transform(df_encoded[col])
                        df_encoded = df_encoded.with_columns(
                            pl.Series(name=col, values=encoded_data)
                        )
                        
                elif method == 'helmert':
                    encoder = HelmertEncoder(cols=columns)
                    encoded_data = encoder.fit_transform(df_encoded[columns])
                    df_encoded = df_encoded.drop(columns).hstack(pl.DataFrame(encoded_data))
                    
                elif method == 'catboost':
                    encoder = CatBoostEncoder(cols=columns)
                    encoded_data = encoder.fit_transform(df_encoded[columns])
                    df_encoded = df_encoded.drop(columns).hstack(pl.DataFrame(encoded_data))
                
                # Запись статистики
                encoding_stats[method] = {
                    'columns': columns,
                    'original_size': len(columns),
                    'encoded_size': df_encoded.shape[1] - len(self.df.columns) + len(columns)
                }
            
            # Печать статистики кодирования
            print("\nСтатистика кодирования:")
            for method, stats in encoding_stats.items():
                print(f"\n{method.upper()} Encoding:")
                print(f"Столбцы закодированы: {stats['columns']}")
                print(f"Оригинальные признаки: {stats['original_size']}")
                print(f"Закодированные признаки: {stats['encoded_size']}")
            
            return df_encoded
            
        except Exception as e:
            print(f"Ошибка во время кодирования: {str(e)}")
            return None


    def search_ordinal_encoder(self):
        """Определяет столбцы, подходящие для порядкового кодирования."""
        categorical_columns = [col for col in self.df.columns if self.df[col].dtype in [pl.Utf8, pl.Categorical]]

        ordinal_columns = []
        
        for col in categorical_columns:
            unique_count = self.df[col].n_unique()
            if 10 < unique_count <= 30:
                ordinal_columns.append(col)
                
        print(f"Столбцы, подходящие для порядкового кодирования: {ordinal_columns}")
        return ordinal_columns


    def search_bde(self):
        """Идентифицирует столбцы, подходящие для обратного разностного кодирования."""
        categorical_columns = [col for col in self.df.columns if self.df[col].dtype in [pl.Utf8, pl.Categorical]]
        bde_columns = []
        
        for col in categorical_columns:
            unique_count = self.df[col].n_unique()
            if 5 < unique_count <= 15:
                bde_columns.append(col)
                
        print(f"Столбцы, подходящие для обратного разностного кодирования: {bde_columns}")
        return bde_columns


    def be(self, filename, X_train, X_test):
        """Применяет двоичное кодирование к данным"""
        try:
            categorical_columns = [col for col in self.df.columns if self.df[col].dtype in [pl.Utf8, pl.Categorical]]
            encoder = BinaryEncoder(cols=categorical_columns)
            
            X_train_encoded = encoder.fit_transform(X_train[categorical_columns])
            X_test_encoded = encoder.transform(X_test[categorical_columns])
            
            X_train = X_train.drop(categorical_columns).hstack(pl.DataFrame(X_train_encoded))
            X_test = X_test.drop(categorical_columns).hstack(pl.DataFrame(X_test_encoded))
            
            print(X_train.head())
            X_train.write_csv(filename)
            
            return X_train, X_test
            
        except Exception as e:
            print(f"Ошибка при двоичном кодировании: {str(e)}")
            return None, None


    def plot_encoder_futures(self):
        plt.figure(figsize=(10, 8))

        categorical_col = [col for col in self.df.columns if self.df[col].dtype in [pl.Utf8, pl.Categorical]]

        plt.bar(categorical_col, len(categorical_col))
        
        plt.xticks(rotation="vertical")
        plt.title('Кол-во категориальных данных')
        plt.show()


    def encode_and_save(self, output_filename, train_filename=None, test_filename=None):
        """Кодирует данные и сохраняет в файлы"""
        try:
            # Применить автоматическое кодирование
            encoded_df = self.autoselection_method()
            
            if encoded_df is not None:
                # Сохранить полный закодированный набор данных
                encoded_df.write_csv(output_filename)
                print(f"\nEncoded data saved to {output_filename}")
                
                # Разделите и сохраните поезд/тест, если указаны имена файлов.
                if train_filename and test_filename:
                    train_size = int(0.8 * len(encoded_df))
                    train_df = encoded_df.slice(0, train_size)
                    test_df = encoded_df.slice(train_size)
                    
                    train_df.write_csv(train_filename)
                    test_df.write_csv(test_filename)

                    print(f"Тренировочные данные сохранены {train_filename}")
                    print(f"Тестовые данные сохранены {test_filename}")
                
                print("\nПервые несколько строк закодированных данных:")
                print(encoded_df.head())
                
                return encoded_df
                
        except Exception as e:
            print(f"Ошибка при двоичном кодировании: {str(e)}")
            return None