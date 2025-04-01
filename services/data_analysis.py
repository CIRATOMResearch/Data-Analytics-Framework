import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64



class DataAnalyzer:
    def __init__(self, df):
        self.df = df
        
        
    def get_basic_metrics(self, column):
        numeric_data = pd.to_numeric(self.df[column], errors='coerce')
        return {
            'mean': numeric_data.mean(),
            'median': numeric_data.median(),
            'mode': numeric_data.mode().iloc[0] if not numeric_data.mode().empty else None,
            'std': numeric_data.std(),
            'var': numeric_data.var(),
            'min': numeric_data.min(),
            'max': numeric_data.max(),
            'count': numeric_data.count()
        }
    

    def get_dispersion_metrics(self, column):
        numeric_data = pd.to_numeric(self.df[column], errors='coerce')
        return {
            'variance': numeric_data.var(),
            'std_dev': numeric_data.std(),
            'range': numeric_data.max() - numeric_data.min(),
            'iqr': numeric_data.quantile(0.75) - numeric_data.quantile(0.25),
            'mad': np.abs(numeric_data - numeric_data.mean()).mean()
        }
    

    def get_quartiles(self, column):
        numeric_data = pd.to_numeric(self.df[column], errors='coerce')
        return {
            'q1': numeric_data.quantile(0.25),
            'q2': numeric_data.quantile(0.50),
            'q3': numeric_data.quantile(0.75),
            'p10': numeric_data.quantile(0.10),
            'p90': numeric_data.quantile(0.90)
        }
    

    def get_distribution_metrics(self, column):
        numeric_data = pd.to_numeric(self.df[column], errors='coerce')
        return {
            'skewness': numeric_data.skew(),
            'kurtosis': numeric_data.kurtosis(),
            'is_normal': stats.normaltest(numeric_data.dropna())[1] > 0.05
        }


    def plot_distribution(self, column):
        """
        Creates an extended visualization of distribution
        """
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        plt.subplots_adjust(hspace=0.3)
        
        numeric_data = pd.to_numeric(self.df[column], errors='coerce').dropna()
        
        # 1. Histogram with KDE
        sns.histplot(data=numeric_data, stat='density', kde=True, ax=ax1, color='#10a37f')
        ax1.axvline(numeric_data.mean(), color='red', linestyle='dashed', linewidth=2, label='Mean')
        ax1.axvline(numeric_data.median(), color='green', linestyle='dashed', linewidth=2, label='Median')
        ax1.set_title(f'Distribution of {column}')
        ax1.set_xlabel('Values')
        ax1.set_ylabel('Density')
        ax1.legend()

        # 2. Box plot with violin plot
        sns.violinplot(y=numeric_data, ax=ax2, color='#10a37f', inner='box')
        ax2.set_title('Box plot with Violin plot')
        
        # Save plot
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
        buffer.seek(0)
        plt.close()
        
        # Encode in base64
        image = base64.b64encode(buffer.getvalue()).decode()
        
        # Calculate extended statistics
        statistics_dict = {
            'basic': {
                'mean': float(numeric_data.mean()),
                'median': float(numeric_data.median()),
                'std': float(numeric_data.std()),
                'skew': float(numeric_data.skew()),
                'kurtosis': float(numeric_data.kurtosis())
            },
            'percentiles': {
                'p1': float(np.percentile(numeric_data, 1)),
                'p5': float(np.percentile(numeric_data, 5)),
                'p25': float(np.percentile(numeric_data, 25)),
                'p75': float(np.percentile(numeric_data, 75)),
                'p95': float(np.percentile(numeric_data, 95)),
                'p99': float(np.percentile(numeric_data, 99))
            },
            'normality_tests': {
                'shapiro': stats.shapiro(numeric_data)[:2],
                'normaltest': stats.normaltest(numeric_data)[:2],
                'kstest': stats.kstest(numeric_data, 'norm')[:2]
            },
            'distribution_metrics': {
                'iqr': float(stats.iqr(numeric_data)),
                'range': float(numeric_data.max() - numeric_data.min()),
                'variance': float(numeric_data.var()),
                'mad': float(stats.median_abs_deviation(numeric_data)),
                'mode': float(stats.mode(numeric_data)[0])
            }
        }
        
        return {
            'plot': image,
            'statistics': statistics_dict
        }