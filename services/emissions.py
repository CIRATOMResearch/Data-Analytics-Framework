import polars as pl
import numpy as np

from scipy import stats

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn import svm



class AutoDataEmissions:
    def __init__(self, df):
        # Конвертируем в polars если нужно
        if not isinstance(df, pl.DataFrame):
            df = pl.from_pandas(df)
        
        self.df = df  # Инициализируем атрибут df


    def get_visualization_data(self, column):
        """Получить данные для визуализации"""
        data = self.df[column].to_numpy()
        # Данные для boxplot
        q1 = np.percentile(data, 25)
        median = np.percentile(data, 50)
        q3 = np.percentile(data, 75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        boxplot_data = {
            'q1': float(q1),
            'median': float(median),
            'q3': float(q3),
            'whiskers': [float(lower), float(upper)],
            'outliers': [float(x) for x in data[(data < lower) | (data > upper)]]
        }
        
        # Данные для гистограммы
        hist, bins = np.histogram(data, bins='auto')
        histogram_data = {
            'values': hist.tolist(),
            'bins': [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]
        }
        
        return {
            'boxplot': boxplot_data,
            'histogram': histogram_data
        }
        

    def detect_outliers(self, column, method='iqr'):
        if method == 'iqr':
            return self.get_outliers_iqr(column)
        elif method == 'zscore':
            return self.get_outliers_zscore(column)
        elif method == 'isolation_forest':
            return self.isolation_forest(column)
        elif method == 'dbscan':
            return self.dbscan(column)
        elif method == 'lof':
            return self.LOF(column)
        else:
            raise ValueError(f"Unknown method: {method}")


    def get_outliers_iqr(self, column):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = self.df.filter((pl.col(column) < lower_bound) | (pl.col(column) > upper_bound))
        return outliers, lower_bound, upper_bound


    def get_outliers_zscore(self, column, threshold=3):
        """Получите выбросы, используя метод z-оценки"""
        z_scores = (self.df[column] - self.df[column].mean()) / self.df[column].std()
        # Использовать фильтр Polars вместо логического индексирования
        outliers = self.df.filter(abs(pl.col(column) - self.df[column].mean()) > threshold * self.df[column].std())
        return outliers, -threshold, threshold


    def isolation_forest(self, column):
        """Получите выбросы, используя Isolation Forest"""
        X = self.df[column].to_numpy().reshape(-1, 1)
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        predictions = iso_forest.fit_predict(X)
        # Преобразуйте прогнозы в логическую маску и используйте фильтр
        outliers = self.df.filter(pl.Series(predictions == -1))
        return outliers, None, None


    def dbscan(self, column):
        """Получите выбросы с помощью DBSCAN"""
        X = self.df[column].to_numpy().reshape(-1, 1)
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(X)
        # Преобразуйте метки в логическую маску и используйте фильтр.
        outliers = self.df.filter(pl.Series(labels == -1))
        return outliers, None, None


    def LOF(self, column):
        """Получите выбросы, используя локальный коэффициент выбросов"""
        X = self.df[column].to_numpy().reshape(-1, 1)
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        predictions = lof.fit_predict(X)
        # Преобразуйте прогнозы в логическую маску и используйте фильтр
        outliers = self.df.filter(pl.Series(predictions == -1))
        return outliers, None, None


    def auto_detect_outliers(self):
        results = {}

        for column in self.df.columns:
            if self.df[column].dtype in [pl.Float64, pl.Int64]:

                methods = ['iqr', 'zscore', 'isolation_forest', 'dbscan', 'lof']
                outliers_count = []

                for method in methods:
                    outliers, _, _ = self.detect_outliers(column, method)
                    outliers_count.append(len(outliers))
                best_method = methods[np.argmin(outliers_count)]
                
                outliers, lower_bound, upper_bound = self.detect_outliers(column, best_method)
                
                results[column] = {
                    'method': best_method,
                    'outliers': outliers.to_dicts(),
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }

        return results


    def remove_outliers(self, column, method='iqr'):
        """Удалить выбросы из набора данных"""
        outliers, _, _ = self.detect_outliers(column, method)
        if len(outliers) > 0:
            # Создайте логическую маску для строк, не являющихся выбросами.
            mask = ~self.df[column].is_in(outliers[column])
            self.df = self.df.filter(mask)
        return self.df


    def cap_outliers(self, column, method='iqr'):
        outliers, lower_bound, upper_bound = self.detect_outliers(column, method)
        # Используйте with_columns для выполнения операции обрезки.
        self.df = self.df.with_columns(
            pl.col(column).clip(lower_bound, upper_bound).alias(column)
        )
        return self.df


    def analyze_outliers(self):
        results = {}
        for column in self.df.columns:
            if self.df[column].dtype in [pl.Float64, pl.Int64]:
                outliers, _, _ = self.detect_outliers(column, 'iqr')
                total_count = len(self.df[column])
                outlier_count = len(outliers)
                percentage = (outlier_count / total_count) * 100
                results[column] = {
                    'count': outlier_count,
                    'percentage': percentage,
                    'recommended_method': self._recommend_method(column)
                }
        return results


    def _recommend_method(self, column):
        """Рекомендации по расширенному методу для обнаружения выбросов"""
        # Получить характеристики данных
        n_samples = len(self.df)
        n_unique = self.df[column].n_unique()
        skewness = self.df[column].skew()
        # Логика принятия решения
        if n_samples < 1000:
            if abs(skewness) < 1:  # Нормальное распределение
                return 'zscore'
            else:
                return 'iqr'
        else:
            if n_unique / n_samples < 0.1:  # Низкое разнообразие
                return 'iqr'
            elif abs(skewness) > 2:  # Сильно перекошенный
                return 'isolation_forest'
            else:
                return 'lof'


    def get_visualization_data(self, column):
        """
        Создает визуализацию данных с помощью seaborn
        """
        
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
            import base64
            from io import BytesIO
            from scipy import stats

            # Преобразуем данные в numpy array
            data = self.df[column].to_numpy()
            
            # Создаем figure с подграфиками
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
            plt.subplots_adjust(hspace=0.4)

            # 1. Box plot с violin plot
            sns.boxplot(y=data, ax=ax1, color='#10a37f')
            sns.stripplot(y=data, ax=ax1, color='red', alpha=0.3, size=4)
            ax1.set_title(f'Box Plot и Strip Plot для {column}')

            # 2. Histogram с KDE
            sns.histplot(data=data, kde=True, ax=ax2, color='#10a37f')
            ax2.axvline(np.mean(data), color='red', linestyle='--', label='Среднее')
            ax2.axvline(np.median(data), color='green', linestyle='--', label='Медиана')
            ax2.set_title(f'Распределение значений {column}')
            ax2.legend()

            # 3. QQ plot
            stats.probplot(data, dist="norm", plot=ax3)
            ax3.set_title('Q-Q Plot (проверка на нормальность)')

            # Сохраняем график
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
            buffer.seek(0)
            plt.close()

            # Кодируем в base64
            image = base64.b64encode(buffer.getvalue()).decode()

            # Считаем статистики
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = data[(data < lower_bound) | (data > upper_bound)]

            stats_dict = {
                'basic_stats': {
                    'mean': float(np.mean(data)),
                    'median': float(np.median(data)),
                    'std': float(np.std(data)),
                    'skew': float(stats.skew(data)),
                    'kurtosis': float(stats.kurtosis(data))
                },
                'outliers_stats': {
                    'count': len(outliers),
                    'percentage': float((len(outliers) / len(data)) * 100),
                    'boundaries': {
                        'lower': float(lower_bound),
                        'upper': float(upper_bound)
                    }
                }
            }

            return {
                'plot': image,
                'statistics': stats_dict
            }
            
        except Exception as e:
            print(f"Ошибка в get_visualization_data: {str(e)}")
            return None