from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, OneHotEncoder, LabelEncoder
import polars as pl
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from .emissions import AutoDataEmissions


class AutoStandartization:
    def __init__(self, df, p_value_threshold=0.05, iqr_multiplier=1.5, z_score_threshold=3, outlier_strategy='robust'):
        self.df = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
        self.p_value_threshold = p_value_threshold
        self.iqr_multiplier = iqr_multiplier
        self.z_score_threshold = z_score_threshold
        self.outlier_strategy = outlier_strategy
        self.emissions = AutoDataEmissions(df)


    def auto_selection_method(self):
        results = {}
        for col in self.df.columns:
            if self.df[col].dtype in [pl.Int64, pl.Float64]:
                # Проверка на нормальное распределение
                _, p_value = stats.shapiro(self.df[col].to_numpy())
                
                # Проверка на наличие выбросов
                outliers, _, _ = self.emissions.detect_outliers(col, method='iqr')
                outlier_count = len(outliers)
                
                # Выбор метода
                if p_value > self.p_value_threshold:
                    if outlier_count / len(self.df) < 0.01:
                        method = 'StandardScaler'
                    else:
                        method = 'RobustScaler'
                else:
                    method = 'MinMaxScaler'
                
                results[col] = method
        return results


    def examination_on_normal_distribution(self):
        results = {}
        for col in self.df.columns:
            if self.df[col].dtype in [pl.Int64, pl.Float64]:
                data = self.df[col].to_numpy()
                shapiro_test = stats.shapiro(data)
                ks_test = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
                
                if shapiro_test[1] >= self.p_value_threshold and ks_test[1] >= self.p_value_threshold:
                    results[col] = 'Нормальное распределение'
                else:
                    results[col] = 'Не нормальное распределение'
        return results


    def normalize(self, columns=None, method='auto'):
        """
        Нормализация данных
        Args:
            columns: список колонок для нормализации
            method: метод нормализации ('auto', 'minmax', 'zscore', 'robust', 'maxabs')
        Returns:
            pl.DataFrame: нормализованный DataFrame
        """
        if columns is None:
            columns = [col for col in self.df.columns if self.df[col].dtype in [pl.Int64, pl.Float64]]
        
        result_df = self.df.clone()
        
        if method == 'auto':
            auto_methods = self.auto_selection_method()
            for col in columns:
                if col in auto_methods:
                    try:
                        scaler = self._get_scaler(auto_methods[col])
                        scaled_values = scaler.fit_transform(
                            result_df[col].to_numpy().reshape(-1, 1)
                        ).flatten()
                        result_df = result_df.with_columns(pl.Series(name=col, values=scaled_values))
                    except Exception as e:
                        print(f"Ошибка при масштабировании колонки {col}: {str(e)}")
                        continue
        else:
            try:
                scaler = self._get_scaler(method)
                for col in columns:
                    scaled_values = scaler.fit_transform(
                        result_df[col].to_numpy().reshape(-1, 1)
                    ).flatten()
                    result_df = result_df.with_columns(pl.Series(name=col, values=scaled_values))
            except Exception as e:
                print(f"Ошибка при масштабировании: {str(e)}")
                raise
        
        return result_df


    def _get_scaler(self, method):
        if method in ['minmax', 'MinMaxScaler']:
            return MinMaxScaler()
        elif method in ['zscore', 'StandardScaler']:
            return StandardScaler()
        elif method in ['robust', 'RobustScaler']:
            return RobustScaler()
        elif method in ['maxabs', 'MaxAbsScaler']:
            return MaxAbsScaler()
        else:
            raise ValueError(f"Неподдерживаемый метод масштабирования: {method}")


    def get_scaling_stats(self, columns=None):
        if columns is None:
            columns = [col for col in self.df.columns if self.df[col].dtype in [pl.Int64, pl.Float64]]
        
        stats = {}
        for col in columns:
            stats[col] = {
                'mean': float(self.df[col].mean()),
                'std': float(self.df[col].std()),
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max())
            }
        return stats


    def get_outliers_three_sigma(self):
        """Определяет выбросы методом 3-х сигм"""
        results = []
        columns = []
        
        for col in self.df.columns:
            if self.df[col].dtype in [pl.Int64, pl.Float64]:
                data = self.df[col].to_numpy()
                mean = float(self.df[col].mean())
                std = float(self.df[col].std())
                outliers = data[abs(data - mean) > 3 * std]
                outlier_percentage = (len(outliers) / len(data)) * 100
                
                columns.append(col)
                results.append({
                    'mean': mean,
                    'std': std,
                    'n_outliers': len(outliers),
                    'outlier_percentage': outlier_percentage
                })
        
        # Создаем DataFrame с правильной структурой
        return pl.DataFrame(
            results,
            schema=['mean', 'std', 'n_outliers', 'outlier_percentage']
        ).with_columns(pl.Series('column', columns))


    def encode_categorical(self, columns=None, method='onehot'):
        if columns is None:
            columns = [col for col in self.df.columns if self.df[col].dtype == pl.Utf8]
        
        result_df = self.df.clone()
        
        for col in columns:
            if method == 'onehot':
                encoder = OneHotEncoder(sparse=False)
                encoded_values = encoder.fit_transform(result_df[col].to_numpy().reshape(-1, 1))
                feature_names = [f"{col}_{i}" for i in range(encoded_values.shape[1])]
                
                for i, name in enumerate(feature_names):
                    result_df = result_df.with_columns(pl.Series(name=name, values=encoded_values[:, i]))
                
                # Опционально удаляем исходный столбец
                result_df = result_df.drop(col)
                
            elif method == 'label':
                encoder = LabelEncoder()
                encoded_values = encoder.fit_transform(result_df[col].to_numpy())
                result_df = result_df.with_columns(pl.Series(name=col, values=encoded_values))
        
        return result_df


    def plot_distribution(self, columns=None):
        if columns is None:
            columns = [col for col in self.df.columns if self.df[col].dtype in [pl.Int64, pl.Float64]]
        
        plots = {}
        for col in columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # До масштабирования
            self.df[col].to_numpy().hist(bins=30, ax=ax1, color='blue', alpha=0.7)
            ax1.set_title(f'До масштабирования: {col}')
            
            # После масштабирования
            scaled_df = self.normalize([col])
            scaled_df[col].to_numpy().hist(bins=30, ax=ax2, color='green', alpha=0.7)
            ax2.set_title(f'После масштабирования: {col}')
            
            # Сохраняем график в буфер
            from io import BytesIO
            import base64
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            image = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            plots[col] = image
        
        return plots