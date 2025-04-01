import pandas as pd
import numpy as np
import polars as pl

from sklearn.impute import SimpleImputer

from sklearn.model_selection import (KFold, LeaveOneOut, StratifiedKFold, 
                                   TimeSeriesSplit, GroupKFold, RepeatedKFold,
                                   train_test_split, RandomizedSearchCV)

from sklearn.metrics import mean_squared_error, accuracy_score, make_scorer

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier, 
                            GradientBoostingRegressor, GradientBoostingClassifier)

from sklearn.svm import SVR, SVC

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

import json
import joblib
from datetime import datetime
import time
import os
import traceback

traceback.print_exc()


class AutoModelSelector:
    def __init__(self, task_type='regression', n_trials=10, top_k=3):
        self.task_type = task_type
        self.n_trials = n_trials
        self.top_k = top_k
        self.models = self._get_models()
        self.best_models = []  # Список лучших моделей
        self.numerical_features = None
        self.categorical_features = None
        self.preprocessor = None
        self.label_encoder = LabelEncoder()
        self.feature_importances = None


    def _convert_to_pandas(self, X):
        """Конвертирует Polars DataFrame в Pandas DataFrame"""
        if isinstance(X, pl.DataFrame):
            return X.to_pandas()
        return X
    

    def _prepare_data(self, X, y):
        """Обработка данных (можно добавить дополнительную логику)"""
        # Если требуется кодирование целевой переменной для классификации
        if self.task_type == 'classification':
            y = self.label_encoder.fit_transform(y)
        return X, y
    

    def _create_preprocessor(self):
        """Создает пайплайн для предобработки данных"""
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        return preprocessor


    def _get_models(self):
        if self.task_type == 'regression':
            return {
                'linear': (LinearRegression(), {}),
                'ridge': (Ridge(), {
                    'alpha': [0.1, 1.0, 10.0]
                }),
                'lasso': (Lasso(), {
                    'alpha': [0.1, 1.0, 10.0]
                }),
                'rf': (RandomForestRegressor(), {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20]
                }),
                'gbm': (GradientBoostingRegressor(), {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1]
                }),
                'xgb': (XGBRegressor(), {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1]
                }),
                'lgbm': (LGBMRegressor(), {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1]
                })
            }
        
        else:
            return {
                'logistic': (LogisticRegression(max_iter=1000), {
                    'C': [0.1, 1.0, 10.0]
                }),
                'rf': (RandomForestClassifier(), {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20]
                }),
                'gbm': (GradientBoostingClassifier(), {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1]
                }),
                'xgb': (XGBClassifier(), {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1]
                }),
                'lgbm': (LGBMClassifier(), {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1]
                })
            }


    def _identify_feature_types(self, X):
        X = self._convert_to_pandas(X)
        if isinstance(X, pd.DataFrame):
            self.numerical_features = X.select_dtypes(
                include=['int64', 'float64']).columns.tolist()
            self.categorical_features = X.select_dtypes(
                include=['object', 'category']).columns.tolist()
        else:
            self.numerical_features = list(range(X.shape[1]))
            self.categorical_features = []


    def fit(self, X, y):
        # Конвертируем в pandas если необходимо
        X = self._convert_to_pandas(X)
        if isinstance(y, pl.Series):
            y = y.to_pandas()

        self._identify_feature_types(X)
        self.preprocessor = self._create_preprocessor()
        X, y = self._prepare_data(X, y)
        
        print("\nНачало процесса выбора модели:")
        print("=" * 50)
        
        results = []
        
        for name, (model, params) in self.models.items():
            print(f"\nОптимизация модели: {name}")
            print("-" * 30)
            
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', model)
            ])
            
            best_model, best_score = self._optimize_hyperparameters(
                pipeline, 
                {f'classifier__{k}': v for k, v in params.items()},
                X, y
            )
            
            results.append({
                'name': name,
                'score': best_score,
                'model': best_model,
                'timestamp': datetime.now(),
                'feature_importance': self._get_feature_importances(best_model.named_steps['classifier'])
            })
            
            print(f"Лучший скор: {best_score:.4f}")
        
        # Сортируем результаты и сохраняем top_k моделей
        results.sort(key=lambda x: x['score'], reverse=True)
        self.best_models = results[:self.top_k]
        
        self._print_results(results)
        return self


    def _print_results(self, results):
        print("\n" + "=" * 50)
        print("Итоговые результаты:")
        print("=" * 50)
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Модель: {result['name']}")
            print(f"   Скор: {result['score']:.4f}")
            print(f"   Время обучения: {result['timestamp']}")
            if result['feature_importance'] is not None:
                print("   Важность признаков:")
                features = self.numerical_features + self.categorical_features
                for feat, imp in zip(features, result['feature_importance']):
                    print(f"   - {feat}: {imp:.4f}")


    def save_models(self, base_filename):
        """Сохраняет топ моделей в отдельные файлы"""
        saved_models = []
        for i, model_data in enumerate(self.best_models):
            filename = f"{base_filename}_model_{i+1}.joblib"
            model_info = {
                'model': model_data['model'],
                'name': model_data['name'],
                'score': model_data['score'],
                'numerical_features': self.numerical_features,
                'categorical_features': self.categorical_features,
                'feature_importance': model_data['feature_importance'],
                'timestamp': model_data['timestamp'],
                'label_encoder': self.label_encoder,
                'task_type': self.task_type
            }
            joblib.dump(model_info, filename)
            saved_models.append({
                'filename': filename,
                'name': model_data['name'],
                'score': model_data['score']
            })
            
        # Сохраняем метаданные в JSON
        meta_filename = f"{base_filename}_metadata.json"
        with open(meta_filename, 'w') as f:
            json.dump(saved_models, f, indent=4, default=str)


    @classmethod
    def load_model(cls, filename):
        """Загружает сохраненную модель"""
        model_info = joblib.load(filename)

        instance = cls(task_type=model_info['task_type'])

        instance.best_models = [{
            'model': model_info['model'],
            'name': model_info['name'],
            'score': model_info['score'],
            'feature_importance': model_info['feature_importance'],
            'timestamp': model_info['timestamp']
        }]

        instance.numerical_features = model_info['numerical_features']
        instance.categorical_features = model_info['categorical_features']
        instance.label_encoder = model_info['label_encoder']

        return instance


    def predict(self, X, model_index=0):
        """
        Делает предсказания используя определенную модель из списка лучших
        model_index: индекс модели (0 - лучшая модель)
        """
        if not self.best_models:
            raise ValueError("No fitted models available")
        if model_index >= len(self.best_models):
            raise ValueError(f"Model index {model_index} is out of range")
        
        X = self._convert_to_pandas(X)
        if isinstance(X, pd.DataFrame):
            required_columns = self.numerical_features + self.categorical_features
            X = X[required_columns]
        
        model = self.best_models[model_index]['model']
        predictions = model.predict(X)
        
        if self.task_type == 'classification':
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions


def validate_data(df):
    """Проверяет корректность данных"""
    # Проверка наличия необходимых колонок
    required_columns = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                       'Conductivity', 'Organic_carbon', 'Trihalomethanes',
                       'Turbidity', 'Potability']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Отсутствуют колонки: {missing_columns}")
    
    # Проверка типов данных
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_columns) != len(df.columns):
        raise ValueError("Все колонки должны быть числовыми")
    
    # Проверка целевой переменной
    if not set(df['Potability'].unique()).issubset({0, 1}):
        raise ValueError("Potability должен содержать только 0 и 1")
    
    return True


def create_water_potability_dataset(n_samples=1000):
    """
    Создает датасет о качестве воды
    """
    np.random.seed(42)
    
    data = {
        'ph': np.random.uniform(0, 14, n_samples),
        'Hardness': np.random.uniform(47, 323, n_samples),
        'Solids': np.random.uniform(320, 61227, n_samples),
        'Chloramines': np.random.uniform(0.3, 13, n_samples),
        'Sulfate': np.random.uniform(129, 481, n_samples),
        'Conductivity': np.random.uniform(181, 753, n_samples),
        'Organic_carbon': np.random.uniform(2.2, 28.3, n_samples),
        'Trihalomethanes': np.random.uniform(0, 124, n_samples),
        'Turbidity': np.random.uniform(1.45, 6.74, n_samples),
    }
    
    # Создаем целевую переменную на основе правил
    ph_ok = (data['ph'] >= 6.5) & (data['ph'] <= 8.5)
    hardness_ok = data['Hardness'] <= 180
    solids_ok = data['Solids'] <= 500
    chloramines_ok = data['Chloramines'] <= 4
    sulfate_ok = data['Sulfate'] <= 250
    conductivity_ok = data['Conductivity'] <= 400
    
    # Вода считается питьевой, если соответствует большинству критериев
    conditions = np.array([ph_ok, hardness_ok, solids_ok, chloramines_ok, sulfate_ok, conductivity_ok])
    data['Potability'] = (conditions.sum(axis=0) >= 4).astype(int)
    
    return data


def save_dataset(format='both'):
    """
    Сохраняет датасет в указанном формате
    format: 'csv', 'parquet' или 'both'
    """
    data = create_water_potability_dataset()
    
    if format in ['csv', 'both']:
        pd.DataFrame(data).to_csv('water_potability.csv', index=False)
        print("Датасет сохранен в CSV формате")
    
    if format in ['parquet', 'both']:
        pd.DataFrame(data).to_parquet('water_potability.parquet')
        print("Датасет сохранен в Parquet формате")
    
    return data


def print_dataset_info(data):
    """
    Выводит информацию о датасете
    """
    df = pd.DataFrame(data)
    
    print("\nИнформация о датасете:")
    print("=" * 50)
    print(f"Размер датасета: {df.shape}")
    print("\nОписательная статистика:")
    print(df.describe())
    print("\nКоличество питьевой/непитьевой воды:")
    print(df['Potability'].value_counts())
    
    # Проверяем корреляции
    correlations = df.corr()['Potability'].sort_values(ascending=False)
    print("\nКорреляции с целевой переменной:")
    print(correlations)


def main():
    try:
        
        if not os.path.exists('water_potability.csv'):
            print("Создание нового датасета...")
            data = create_water_potability_dataset()
            pd.DataFrame(data).to_csv('water_potability.csv', index=False)
        
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")
    # Создаем тестовые данные
    # Загрузка данных
    file_path = 'water_potability.csv'

    if not os.path.exists(file_path):
        print(f"Ошибка: Файл '{file_path}' не найден!")
        return
    try:

        df = pd.read_csv('water_potability.csv')

        if validate_data(df):
            print("Данные прошли валидацию")

        if 'Potability' not in df.columns:
            raise ValueError("В датасете отсутствует колонка 'Potability'")
        
        y = df['Potability'].values.ravel()
        X = df.drop('Potability', axis=1)  # Преобразуем признаки в pandas DataFrame
        
        print("\nФорма X:", X.shape)
        print("Форма y:", y.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Создание и обучение модели
        selector = AutoModelSelector(task_type='classification', n_trials=5, top_k=3)
        selector.fit(X_train, y_train)
        predictions = selector.predict(X_test)

        # Сохранение моделей
        selector.save_models('best_models')

        # Предсказания с использованием разных моделей
        predictions_best = selector.predict(X_test)  # Лучшая модель
        predictions_second = selector.predict(X_test, model_index=1)  # Вторая лучшая модель

    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")
        print("\nStack trace:")
     

start = time.time()
print(f"{start:.4f}")
main()
print(f"{time.time()-start:.4f}")