import polars as pl
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List, Dict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import silhouette_score, calinski_harabasz_score



class StatisticalAnalysis:
    def __init__(self, data: Union[pl.DataFrame, np.ndarray]):
        """
        Инициализация класса для статистического анализа
        Args:
            data: DataFrame или numpy array с данными
        """
        if isinstance(data, np.ndarray):
            self.data = pl.DataFrame(data)
        elif isinstance(data, pl.DataFrame):
            self.data = data
        else:
            self.data = pl.from_pandas(data)


    def percentiles(self, percentiles: List[float] = None) -> Dict:
        """
        Расчет процентилей
        Args:
            percentiles: список процентилей для расчета (0-100)
        Returns:
            Dict с процентилями для каждой колонки
        """
        if percentiles is None:
            percentiles = [0, 10, 25, 50, 75, 90, 100]
        
        percentiles_dict = {}
        numeric_cols = self.data.select(pl.col('^.*$').is_numeric()).columns
        
        for column in numeric_cols:
            percentiles_dict[column] = {
                f'p{p}': self.data.select(pl.col(column).quantile(p / 100)).item()
                for p in percentiles
            }
        return percentiles_dict



    def get_numeric_columns(self):
        """Получение списка числовых колонок"""
        return [col for col in self.data.columns 
                if self.data[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]


    def correlation_analysis(self) -> pl.DataFrame:
        """
        Корреляционный анализ
        Returns:
            DataFrame с корреляционной матрицей
        """
        numeric_cols = self.get_numeric_columns()
        return self.data.select(numeric_cols).corr()


    def factor_analysis(self) -> pl.DataFrame:
        """
        Факторный анализ
        Returns:
            DataFrame с компонентами
        """
        numeric_cols = self.get_numeric_columns()
        numeric_data = self.data.select(numeric_cols)
        
        # Заполняем пропуски медианой
        for col in numeric_cols:
            numeric_data = numeric_data.with_columns(
                pl.col(col).fill_null(pl.col(col).median())
            )
        
        fa = FactorAnalysis(n_components=2, random_state=42)
        fa.fit(numeric_data.to_numpy())
        
        return pl.DataFrame(
            fa.components_,
            schema=numeric_cols
        )


    def cluster_analysis(self, n_clusters: int = 3) -> pl.DataFrame:
        """
        Кластерный анализ с использованием K-Means
        Args:
            n_clusters: Количество кластеров
        Returns:
            DataFrame с добавленным столбцом 'cluster'
        """
        numeric_cols = self.get_numeric_columns()
        numeric_data = self.data.select(numeric_cols)
        
        # Заполняем пропуски медианой
        for col in numeric_cols:
            numeric_data = numeric_data.with_columns(
                pl.col(col).fill_null(pl.col(col).median())
            )
        
        # Масштабируем данные
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data.to_numpy())
        
        # Применяем K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Добавляем метки кластеров
        return self.data.with_columns(pl.Series(name="cluster", values=clusters))


    def regression_analysis(self, target_column: str, test_size: float = 0.2) -> Dict:
        """
        Регрессионный анализ
        Args:
            target_column: Целевая переменная
            test_size: Размер тестовой выборки
        Returns:
            Dict с результатами регрессии
        """
        numeric_cols = self.get_numeric_columns()
        
        if target_column not in numeric_cols:
            raise ValueError(f"Target column '{target_column}' not found in numeric data")
        
        feature_cols = [col for col in numeric_cols if col != target_column]
        
        # Подготовка данных
        X = self.data.select(feature_cols).to_numpy()
        y = self.data[target_column].to_numpy()
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Обучение модели
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Предсказания и метрики
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = model.score(X_test, y_test)
        
        return {'RMSE': rmse, 'R^2': r2}



    def frequency_analysis(self, column: str, bins: int = None) -> pl.Series:
        """
        Частотный анализ для одной колонки
        Args:
            column: название колонки
            bins: количество интервалов для числовых данных
        Returns:
            Series с частотами
        """
        if pl.col(column).is_numeric() and bins:
            # Для числовых данных с разбиением на интервалы
            min_val = self.data[column].min()
            max_val = self.data[column].max()
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            binned = pl.Series(np.digitize(self.data[column], bin_edges))
            return binned.value_counts()
        return self.data[column].value_counts()


    def frequency_analysis_all(self, bins: int = 10) -> Dict:
        """
        Частотный анализ для всех колонок
        Args:
            bins: количество интервалов для числовых данных
        Returns:
            Dict с частотами для каждой колонки
        """
        return {
            column: self.frequency_analysis(column, bins=bins)
            for column in self.data.columns
        }


    def plot_distribution(self, column: str):
        """
        Визуализация распределения данных
        Args:
            column: название колонки
        """
        data_np = self.data[column].to_numpy()
        
        plt.figure(figsize=(12, 6))
        
        # Гистограмма с кривой плотности
        plt.subplot(121)
        sns.histplot(data_np, kde=True)
        plt.title(f'Распределение {column}')
        
        # Box plot
        plt.subplot(122)
        sns.boxplot(y=data_np)
        plt.title(f'Box plot {column}')
        
        plt.tight_layout()
        plt.show()


    def plot_correlation_heatmap(self):
        """
        Визуализация корреляционной матрицы
        """
        corr_matrix = self.correlation_analysis().to_pandas()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix,
                    annot=True,
                    cmap='coolwarm',
                    center=0)
        plt.title('Корреляционная матрица')
        plt.show()


    def advanced_factor_analysis(self) -> Dict:
        """Расширенный факторный анализ с дополнительными метриками"""
        fa_results = self.factor_analysis()
        
        # Вычисляем объясненную дисперсию
        explained_variance = pl.DataFrame(fa_results).select([
            pl.all().var()
        ]).to_series().to_list()
        
        # Вычисляем кумулятивную дисперсию
        cumulative_variance = pl.Series(explained_variance).cum_sum().to_list()
        
        # Добавляем факторные нагрузки
        numeric_cols = self.get_numeric_columns()
        corr_matrix = self.data.select(numeric_cols).corr()
        
        loadings = pl.DataFrame(
            corr_matrix.to_numpy() @ fa_results.to_numpy().T,
            schema=['Factor1', 'Factor2']
        )
        
        # Значимые факторы
        significant_loadings = loadings.with_columns([
            pl.when(pl.col(col).abs() > 0.7)
            .then(pl.lit("Значимый"))
            .otherwise(pl.lit("Незначимый"))
            .alias(f"{col}_significance")
            for col in loadings.columns
        ])
        
        return {
            'components': fa_results.to_dict(as_series=False),
            'explained_variance': explained_variance,
            'cumulative_variance': cumulative_variance,
            'loadings': loadings.to_dict(as_series=False),
            'significant_factors': significant_loadings.to_dict(as_series=False)
        }


    def advanced_cluster_analysis(self) -> Dict:
        """Расширенный кластерный анализ с метриками качества"""
        # Базовая кластеризация
        clustered_data = self.cluster_analysis()
        numeric_data = clustered_data.select(self.get_numeric_columns()).to_numpy()
        clusters = clustered_data['cluster'].to_numpy()
        
        # Оценка качества кластеризации
        silhouette = silhouette_score(numeric_data, clusters)
        calinski = calinski_harabasz_score(numeric_data, clusters)
        
        # Статистика по кластерам с уникальными именами колонок
        cluster_stats = (clustered_data
            .group_by('cluster')
            .agg([
                pl.count().alias('cluster_size'),
                *[pl.col(col).mean().alias(f'mean_{col}') for col in self.get_numeric_columns()],
                *[pl.col(col).std().alias(f'std_{col}') for col in self.get_numeric_columns()],
                *[pl.col(col).min().alias(f'min_{col}') for col in self.get_numeric_columns()],
                *[pl.col(col).max().alias(f'max_{col}') for col in self.get_numeric_columns()]
            ]))
        
        # Межкластерные расстояния
        centroids = {}
        
        for cluster in range(int(clusters.max()) + 1):
            mask = clusters == cluster
            centroids[cluster] = numeric_data[mask].mean(axis=0)
        
        distances = {}

        for i in centroids:
            for j in centroids:
                if i < j:
                    dist = float(np.linalg.norm(centroids[i] - centroids[j]))
                    distances[f"cluster_{i}-{j}"] = dist
        
        return {
            'clustered_data': clustered_data.to_dict(as_series=False),
            'quality_metrics': {
                'silhouette_score': float(silhouette),
                'calinski_harabasz_score': float(calinski)
            },
            'cluster_statistics': cluster_stats.to_dict(as_series=False),
            'inter_cluster_distances': distances
        }



    def advanced_regression_analysis(self, target_column: str) -> Dict:
        """Расширенный регрессионный анализ с дополнительными метриками"""
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        from sklearn.model_selection import cross_val_score
        
        # Базовый регрессионный анализ
        basic_results = self.regression_analysis(target_column)
        
        # Подготовка данных
        numeric_cols = self.get_numeric_columns()
        feature_cols = [col for col in numeric_cols if col != target_column]
        
        X = self.data.select(feature_cols).to_numpy()
        y = self.data[target_column].to_numpy()
        
        # Кросс-валидация
        cv_scores = cross_val_score(LinearRegression(), X, y, cv=5)
        
        # Важность признаков
        model = LinearRegression()
        model.fit(X, y)
        feature_importance = pl.DataFrame({
            'feature': feature_cols,
            'importance': np.abs(model.coef_)
        }).sort('importance', descending=True)
        
        # Дополнительные метрики
        y_pred = model.predict(X)
        metrics = {
            'r2': r2_score(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        # Остатки
        residuals = y - y_pred
        residuals_stats = {
            'mean': float(residuals.mean()),
            'std': float(residuals.std()),
            'normality_test': stats.normaltest(residuals)
        }
        
        return {
            'basic_metrics': basic_results,
            'advanced_metrics': metrics,
            'feature_importance': feature_importance,
            'cross_validation': {
                'scores': cv_scores.tolist(),
                'mean': float(cv_scores.mean()),
                'std': float(cv_scores.std())
            },
            'residuals': {
                'values': residuals.tolist(),
                'statistics': residuals_stats
            }
        }

    def advanced_correlation_analysis(self) -> Dict:
        """Расширенный корреляционный анализ"""
        # Базовая корреляционная матрица
        corr_matrix = self.correlation_analysis()
        
        # Значимые корреляции
        significant_corr = (corr_matrix
            .with_columns([
                pl.when(pl.col(col).abs() > 0.7)
                .then(pl.lit("Сильная"))
                .when(pl.col(col).abs() > 0.3)
                .then(pl.lit("Средняя"))
                .otherwise(pl.lit("Слабая"))
                .alias(f"{col}_strength")
                for col in corr_matrix.columns
            ]))
        
        # Частичные корреляции
        numeric_data = self.data.select(self.get_numeric_columns())
        partial_corr = {}
        for col1 in numeric_data.columns:
            for col2 in numeric_data.columns:
                if col1 < col2:
                    other_cols = [c for c in numeric_data.columns if c not in (col1, col2)]
                    if other_cols:
                        # Вычисляем частичную корреляцию
                        partial = self._partial_correlation(
                            numeric_data[col1].to_numpy(),
                            numeric_data[col2].to_numpy(),
                            numeric_data.select(other_cols).to_numpy()
                        )
                        partial_corr[f"{col1}-{col2}"] = partial
        
        return {
            'correlation_matrix': corr_matrix,
            'significant_correlations': significant_corr,
            'partial_correlations': partial_corr
        }


    def _partial_correlation(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        """Вычисление частичной корреляции"""
        from sklearn.linear_model import LinearRegression
        
        # Регрессия x на z
        model_x = LinearRegression()
        model_x.fit(z, x)
        x_residuals = x - model_x.predict(z)
        
        # Регрессия y на z
        model_y = LinearRegression()
        model_y.fit(z, y)
        y_residuals = y - model_y.predict(z)
        
        # Корреляция остатков
        return float(np.corrcoef(x_residuals, y_residuals)[0, 1])


    def get_distribution_metrics(self) -> Dict:
        """Анализ распределений"""
        numeric_cols = self.get_numeric_columns()
        results = {}
        
        for col in numeric_cols:
            data = self.data[col].drop_nulls()
            
            # Базовые статистики
            stats_dict = {
                'mean': float(data.mean()),
                'median': float(data.median()),
                'std': float(data.std()),
                'skew': float(data.skew()),
                'kurtosis': float(data.kurtosis()),
            }
            
            # Тесты на нормальность
            _, shapiro_p = stats.shapiro(data.to_numpy())
            _, ks_p = stats.kstest(data.to_numpy(), 'norm')
            
            stats_dict.update({
                'shapiro_test_p': float(shapiro_p),
                'ks_test_p': float(ks_p),
                'is_normal': shapiro_p > 0.05 and ks_p > 0.05
            })
            
            results[col] = stats_dict
        
        return results


    def plot_encoder_futures(df):
        plt.figure(figsize=(10, 8))
        
        value_counts = df.group_by('Country').count().sort('count', descending=True)
        total = value_counts['count'].sum()
        percentages = (value_counts['count'] / total * 100).round(1)
        
        plt.bar(value_counts['Country'], percentages)
        
        for i, percentage in enumerate(percentages):
            plt.text(i, percentage, f'{percentage}%', ha='center', va='bottom')
        
        plt.xticks(rotation="vertical")
        plt.title('Распределение по странам (%)')
        plt.ylabel('Процент (%)')
        plt.show()



    def generate_report(self):
        """
        Генерация полного статистического отчета
        """
        print("=== Статистический анализ данных №1 ===\n")
        
        print(f'Размер до дубликатов - {self.data.shape}')
        
        df_no_duplicates = self.data.unique()
        print(str(df_no_duplicates))
        print(f'Размер после удаления дубликатов - {df_no_duplicates.shape}')
        print('Вот первые 10 строк')
        print(str(self.data.head(10)))
        print(f'Количество строк: {self.data.shape[0]}, Количество столбцов: {self.data.shape[1]}')
        
        print('Расширенная описательная статистика')
        print(str(self.data.describe()))
        
        null_counts = self.data.null_count()
        print('Кол-во пропущенных значений в столбцах:')
        print(str(null_counts))
        
        percentage_nulls = (null_counts / len(self.data)) * 100
        print('Кол-во пропущенных значений в столбцах в %:')
        print(str(percentage_nulls))
        
        print("\n=== Статистический анализ данных №2 ===\n")

        # Процентили
        print("\n2. Процентили:")
        percs = self.percentiles()
        for column, values in percs.items():
            print(f"\n{column}:")
            for p, value in values.items():
                print(f"{p}: {value:.4f}")

        # Корреляционный анализ
        print("\n3. Корреляционный анализ:")
        print(self.correlation_analysis())

        # Кластерный анализ
        print("\n4. Кластерный анализ:")
        try:
            cluster_results = self.cluster_analysis()
            print(cluster_results.group_by('cluster').agg(pl.count()))
        except Exception as e:
            print(f"Ошибка при кластерном анализе: {e}")

        # Факторный анализ
        print("\n5. Факторный анализ:")
        try:
            factor_components = self.factor_analysis()
            print(factor_components)
        except Exception as e:
            print(f"Ошибка при факторном анализе: {e}")

        # Регрессионный анализ
        print("\n6. Регрессионный анализ:")
        try:
            regression_results = self.regression_analysis(target_column="Potability")
            print(regression_results)
        except Exception as e:
            print(f"Ошибка при регрессионном анализе: {e}")

        # Визуализация
        print("\n7. Визуализация распределений:")
        numeric_cols = self.data.select(pl.col('^.*$').is_numeric()).columns
        for column in numeric_cols:
            self.plot_distribution(column)

        print("\n8. Корреляционная матрица:")
        self.plot_correlation_heatmap()

        print("\n9. Количество категориальных признаков в датасете в %")
        self.plot_encoder_futures()