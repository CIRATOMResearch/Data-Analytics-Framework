import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO


def create_analysis_visualizations(factor_analysis, cluster_analysis, correlation_analysis, regression_analysis):
    def create_plot(plot_type, data):
        plt.figure(figsize=(10, 6))
        
        if plot_type == 'factor':
            sns.scatterplot(data=data['components'], palette='viridis')
            plt.title('Factor Analysis Components')
            
        elif plot_type == 'cluster':
            sns.scatterplot(data=data['clustered_data'], hue='cluster', palette='deep')
            plt.title('Cluster Analysis')
            
        elif plot_type == 'correlation':
            sns.heatmap(data['correlation_matrix'], annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix')
            
        elif plot_type == 'regression':
            if 'actual_vs_predicted' in data:
                sns.regplot(x='actual', y='predicted', data=data['actual_vs_predicted'])
                plt.title('Actual vs Predicted Values')
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
        buffer.seek(0)
        plt.close()
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return {
        'factor_plot': create_plot('factor', factor_analysis),
        'cluster_plot': create_plot('cluster', cluster_analysis),
        'correlation_plot': create_plot('correlation', correlation_analysis),
        'regression_plot': create_plot('regression', regression_analysis)
    } 