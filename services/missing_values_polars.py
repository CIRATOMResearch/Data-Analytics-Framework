import polars as pl
from typing import Optional, List, Dict
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns


class MissingValuesAnalyzer:
    def __init__(self, df):
        # Конвертируем pandas DataFrame в polars DataFrame если нужно
        if not isinstance(df, pl.DataFrame):
            self.df = pl.from_pandas(df)
        else:
            self.df = df
        self.missing_patterns = None
        self.missing_correlations = None
        self.analysis_results = {}


    def analyze(self) -> Dict:
        self.analysis_results = {
            'basic_stats': self._get_basic_stats(),
            'missing_patterns': self._analyze_missing_patterns(),
            'correlations': self._analyze_correlations(),
            'mechanism': self._analyze_missing_mechanism(),
            'recommendations': self._get_recommendations()
        }
        return self.analysis_results


    def _get_basic_stats(self) -> Dict:
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_cells = sum(self.df.select([pl.col(col).null_count() for col in self.df.columns]).row(0))

        columns_with_missing = {
            col: self.df[col].null_count() 
            for col in self.df.columns
        }
        
        return {
            'total_missing': missing_cells,
            'missing_percentage': (missing_cells / total_cells) * 100,
            'columns_with_missing': columns_with_missing
        }


    def _analyze_missing_patterns(self) -> Dict:
        missing_patterns = {
            col: self.df[col].null_count() 
            for col in self.df.columns
        }

        nullity_correlation = {}
        for col1 in self.df.columns:
            for col2 in self.df.columns:
                if col1 != col2:
                    try:
                        corr = (
                            self.df
                            .select([
                                pl.col(col1).is_null().cast(pl.Float64),
                                pl.col(col2).is_null().cast(pl.Float64)
                            ])
                            .corr()
                            .item()
                        )
                        nullity_correlation[f"{col1}-{col2}"] = corr
                    except Exception as e:
                        print(f"Error calculating correlation between {col1} and {col2}: {e}")
                        nullity_correlation[f"{col1}-{col2}"] = None

        return {
            'pattern_matrix': missing_patterns,
            'nullity_correlation': nullity_correlation
        }


    def _analyze_correlations(self) -> Dict:
        corr_matrix = {}
        for col1 in self.df.columns:
            for col2 in self.df.columns:
                if col1 != col2:
                    try:
                        corr = (
                            self.df
                            .select([
                                pl.col(col1).is_null().cast(pl.Float64),
                                pl.col(col2).is_null().cast(pl.Float64)
                            ])
                            .corr()
                            .item()
                        )
                        if abs(corr) > 0.5:
                            corr_matrix[f"{col1}-{col2}"] = corr
                    except Exception as e:
                        print(f"Error calculating correlation between {col1} and {col2}: {e}")
                        corr_matrix[f"{col1}-{col2}"] = None
        return corr_matrix


    def _determine_missing_mechanism(self, column: str) -> str:
        missing_mask = self.df[column].is_null()
        other_cols = [col for col in self.df.columns if col != column]

        mcar_pvalues = []
        
        for other_col in other_cols:
            if self.df[other_col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                try:
                    group1 = self.df.filter(~pl.col(column).is_null())[other_col].drop_nulls()
                    group2 = self.df.filter(pl.col(column).is_null())[other_col].drop_nulls()

                    if len(group1) > 0 and len(group2) > 0:
                        t_stat, p_value = stats.ttest_ind(group1.to_numpy(), group2.to_numpy())
                        mcar_pvalues.append(p_value)
                except Exception as e:
                    print(f"Error during t-test for {other_col}: {e}")
                    continue

        if mcar_pvalues and np.mean(mcar_pvalues) > 0.05:
            return 'MCAR'

        mar_correlations = []
        for other_col in other_cols:
            if self.df[other_col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                try:
                    filled_col = self.df[other_col].fill_null(self.df[other_col].mean())
                    corr, _ = stats.pointbiserialr(missing_mask.to_numpy(), filled_col.to_numpy())
                    mar_correlations.append(abs(corr))
                except Exception as e:
                    print(f"Ошибка во время pointbiserialr для {other_col}: {e}")
                    continue

        if mar_correlations and np.max(mar_correlations) > 0.3:
            return 'MAR'

        return 'MNAR'


    def _analyze_missing_mechanism(self) -> Dict:
        mechanisms = {}
        for col in self.df.columns:
            if self.df[col].null_count() > 0:
                mechanisms[col] = self._determine_missing_mechanism(col)
        return mechanisms


    def _get_recommendations(self) -> Dict:
        recommendations = {}
        basic_stats = self._get_basic_stats()
        for column in self.df.columns:
            missing_count = basic_stats['columns_with_missing'].get(column, 0)
            if missing_count > 0:
                missing_percentage = (missing_count / self.df.shape[0]) * 100
                recommendations[column] = self._get_column_recommendation(
                    column, missing_percentage
                )
        return recommendations


    def _get_column_recommendation(self, column: str, missing_percentage: float) -> Dict:
        recommendation = {
            'missing_percentage': missing_percentage,
            'suggested_methods': []
        }

        if missing_percentage > 75:
            recommendation['suggested_methods'].append({
                'method': 'drop_column',
                'reason': 'Слишком много пропущенных значений'
            })
            return recommendation

        if self.df[column].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
            if missing_percentage < 5:
                recommendation['suggested_methods'].extend([
                    {'method': 'mean', 'reason': 'Небольшое количество пропусков'},
                    {'method': 'median', 'reason': 'Альтернатива среднему значению'}
                ])
            else:
                recommendation['suggested_methods'].extend([
                    {'method': 'knn', 'reason': 'Учитывает взаимосвязи между признаками'},
                    {'method': 'iterative', 'reason': 'Сохраняет распределение данных'}
                ])
        else:
            if missing_percentage < 5:
                recommendation['suggested_methods'].append(
                    {'method': 'mode', 'reason': 'Наиболее частое значение'}
                )
            else:
                recommendation['suggested_methods'].append(
                    {'method': 'knn', 'reason': 'Учитывает взаимосвязи между признаками'}
                )

        return recommendation


class MissingValuesImputer:
    def __init__(self, df):
        # Конвертируем pandas DataFrame в polars DataFrame если нужно
        if not isinstance(df, pl.DataFrame):
            self.df = pl.from_pandas(df)
        else:
            self.df = df
        self.original_df = self.df.clone()  # Сохраняем копию исходного датафрейма
        self.imputed_values = {}  # Словарь для хранения замененных значений


    def impute(self, method: str, columns: Optional[List[str]] = None) -> pl.DataFrame:
        if columns is None:
            columns = self.df.columns

        for column in columns:
            if self.df[column].null_count() > 0:
                self.df = self._impute_column(column, method)

        return self.df


    def _impute_column(self, column: str, method: str) -> pl.DataFrame:
        if method == 'knn':
            # Получаем все числовые столбцы
            numeric_cols = [col for col in self.df.columns 
                        if self.df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
            
            # Создаем массив данных для импутации
            X = self.df.select(numeric_cols).to_numpy()
            
            # Создаем и применяем импутер
            imputer = KNNImputer(n_neighbors=5)
            imputed_data = imputer.fit_transform(X)
            
            # Находим индекс нужного столбца
            col_idx = numeric_cols.index(column)
            
            # Обновляем только нужный столбец
            return self.df.with_columns(pl.Series(name=column, values=imputed_data[:, col_idx]))
    

        elif method == 'mean':
            return self.df.with_columns(pl.col(column).fill_null(self.df[column].mean()))
        

        elif method == 'median':
            return self.df.with_columns(pl.col(column).fill_null(self.df[column].median()))


        elif method == 'mode':
            mode_value = self.df[column].mode()[0]
            return self.df.with_columns(pl.col(column).fill_null(mode_value))
        

        elif method == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            imputed_data = imputer.fit_transform(self.df.select(pl.col(column)).to_numpy())
            return self.df.with_columns(pl.Series(name=column, values=imputed_data.flatten()))
        

        elif method == 'iterative':
            imputer = IterativeImputer(random_state=42)
            imputed_data = imputer.fit_transform(self.df.select(pl.col(column)).to_numpy())
            return self.df.with_columns(pl.Series(name=column, values=imputed_data.flatten()))
        

        elif method == 'linear':
            return self._impute_linear(column)
        
        else:
            raise ValueError(f"Неизвестный метод: {method}")


    def _impute_linear(self, target_column: str) -> pl.DataFrame:
        numeric_cols = [col for col in self.df.columns 
                       if self.df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        predictor_columns = [col for col in numeric_cols if col != target_column]

        if not predictor_columns:
            return self.df.with_columns(pl.col(target_column).fill_null(self.df[target_column].mean()))

        train_data = self.df.filter(~pl.col(target_column).is_null())
        test_data = self.df.filter(pl.col(target_column).is_null())

        if len(train_data) == 0 or len(test_data) == 0:
            return self.df.with_columns(pl.col(target_column).fill_null(self.df[target_column].mean()))

        model = LinearRegression()
        
        try:
            X_train = train_data.select(predictor_columns).to_numpy()
            y_train = train_data[target_column].to_numpy()
            X_test = test_data.select(predictor_columns).to_numpy()
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Создаем маску для пропущенных значений
            return self.df.with_columns(
                pl.when(pl.col(target_column).is_null())
                .then(pl.Series(predictions))
                .otherwise(pl.col(target_column))
                .alias(target_column)
            )

        except Exception as e:
            print(f"Ошибка во время линейного вменения для столбца {target_column}: {e}")
            return self.df.with_columns(pl.col(target_column).fill_null(self.df[target_column].mean()))


    def auto_impute(self) -> pl.DataFrame:
        analyzer = MissingValuesAnalyzer(self.df)
        recommendations = analyzer.analyze()['recommendations']

        for column, rec in recommendations.items():
            if rec['suggested_methods']:
                method = rec['suggested_methods'][0]['method']
                if method != 'drop_column':
                    self.df = self._impute_column(column, method)
                else:
                    self.df = self.df.drop(column)

        return self.df


    def visualize_imputation(self, column: str, method: str):
        """
        Визуализирует распределение значений до и после импутации
        """
        original_values = self.original_df[column].drop_nulls()
        
        # Получаем индексы пропущенных значений
        null_indices = (
            self.original_df
            .with_row_count("index")
            .filter(pl.col(column).is_null())
            .get_column("index")
        )
        
        # Применяем импутацию
        imputed_df = self._impute_column(column, method)
        
        # Получаем импутированные значения
        imputed_values = (
            imputed_df
            .with_row_count("index")
            .filter(pl.col("index").is_in(null_indices))
            .get_column(column)
        )
        
        # Создаем график
        plt.figure(figsize=(15, 6))
        
        # График оригинальных значений
        plt.subplot(1, 2, 1)
        sns.histplot(data=original_values, color='blue', alpha=0.5, label='Существующие значения')
        plt.axvline(original_values.mean(), color='red', linestyle='--', label='Среднее')
        plt.axvline(original_values.median(), color='green', linestyle='--', label='Медиана')
        plt.title(f'Распределение существующих значений\n{column}')
        plt.xlabel('Значения')
        plt.ylabel('Частота')
        plt.legend()
        
        # График импутированных значений
        plt.subplot(1, 2, 2)
        sns.histplot(data=original_values, color='blue', alpha=0.5, label='Существующие значения')
        sns.histplot(data=imputed_values, color='red', alpha=0.5, label='Импутированные значения')
        plt.axvline(original_values.mean(), color='blue', linestyle='--', label='Среднее (существующих)')
        plt.axvline(imputed_values.mean(), color='red', linestyle='--', label='Среднее (импутированных)')
        plt.title(f'Сравнение распределений\n{column} (метод: {method})')
        plt.xlabel('Значения')
        plt.ylabel('Частота')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Выводим статистики
        print("\nСтатистики:")
        print("-" * 50)
        print("Оригинальные значения:")
        print(f"Количество: {len(original_values)}")
        print(f"Среднее: {original_values.mean():.4f}")
        print(f"Медиана: {original_values.median():.4f}")
        print(f"Стд. откл.: {original_values.std():.4f}")
        print(f"Мин: {original_values.min():.4f}")
        print(f"Макс: {original_values.max():.4f}")
        
        print("\nИмпутированные значения:")
        print(f"Количество: {len(imputed_values)}")
        print(f"Среднее: {imputed_values.mean():.4f}")
        print(f"Медиана: {imputed_values.median():.4f}")
        print(f"Стд. откл.: {imputed_values.std():.4f}")
        print(f"Мин: {imputed_values.min():.4f}")
        print(f"Макс: {imputed_values.max():.4f}")
        
        return {
            'original_stats': {
                'count': len(original_values),
                'mean': original_values.mean(),
                'median': original_values.median(),
                'std': original_values.std(),
                'min': original_values.min(),
                'max': original_values.max()
            },
            'imputed_stats': {
                'count': len(imputed_values),
                'mean': imputed_values.mean(),
                'median': imputed_values.median(),
                'std': imputed_values.std(),
                'min': imputed_values.min(),
                'max': imputed_values.max()
            }
        }