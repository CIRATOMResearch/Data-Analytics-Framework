from fastapi import APIRouter, Request, Form, HTTPException, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import numpy as np
import polars as pl
import pandas as pd
from typing import List, Optional
import json
import base64
from io import BytesIO
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from fpdf import FPDF

from services.missing_values_polars import MissingValuesAnalyzer, MissingValuesImputer
from services.standart_or_norm import AutoStandartization
from services.emissions import AutoDataEmissions
from services.preview import StatisticalAnalysis
from services.data_analysis import DataAnalyzer
from services.visualization import create_analysis_visualizations
from services.encoder import AutoCategoricalEncoders

from core.state import get_current_df, set_current_df


router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/stats/metrics", response_class=HTMLResponse)
async def stats_metrics(request: Request):
    current_df = get_current_df()
    if current_df is None:
        return templates.TemplateResponse(
            "stats/metrics.html",
            {"request": request, "error": "Нет загруженных данных"}
        )
    
    analyzer = DataAnalyzer(current_df)
    numeric_columns = current_df.select_dtypes(include=[float, int]).columns
    metrics = {col: analyzer.get_basic_metrics(col) for col in numeric_columns}
    
    return templates.TemplateResponse(
        "stats/metrics.html",
        {
            "request": request,
            "columns": numeric_columns,
            "metrics": metrics
        }
    )


@router.get("/stats/dispersion", response_class=HTMLResponse)
async def stats_dispersion(request: Request):
    current_df = get_current_df()
    if current_df is None:
        return templates.TemplateResponse(
            "stats/dispersion.html",
            {"request": request, "error": "Нет загруженных данных"}
        )
    
    analyzer = DataAnalyzer(current_df)
    numeric_columns = current_df.select_dtypes(include=[float, int]).columns
    metrics = {col: analyzer.get_dispersion_metrics(col) for col in numeric_columns}
    
    return templates.TemplateResponse(
        "stats/dispersion.html",
        {
            "request": request,
            "columns": numeric_columns,
            "metrics": metrics
        }
    )


@router.get("/stats/quartiles", response_class=HTMLResponse)
async def stats_quartiles(request: Request):
    current_df = get_current_df()
    if current_df is None:
        return templates.TemplateResponse(
            "stats/quartiles.html",
            {"request": request, "error": "Нет загруженных данных"}
        )
    
    analyzer = DataAnalyzer(current_df)
    numeric_columns = current_df.select_dtypes(include=[np.number]).columns
    quartiles = {col: analyzer.get_quartiles(col) for col in numeric_columns}
    
    return templates.TemplateResponse(
        "stats/quartiles.html",
        {
            "request": request,
            "columns": numeric_columns,
            "quartiles": quartiles
        }
    )


@router.get("/stats/skewness", response_class=HTMLResponse)
async def stats_skewness(request: Request):
    current_df = get_current_df()
    if current_df is None:
        return templates.TemplateResponse(
            "stats/skewness.html",
            {"request": request, "error": "Нет загруженных данных"}
        )
    
    analyzer = DataAnalyzer(current_df)
    numeric_columns = current_df.select_dtypes(include=[np.number]).columns
    distribution_metrics = {col: analyzer.get_distribution_metrics(col) for col in numeric_columns}
    
    return templates.TemplateResponse(
        "stats/skewness.html",
        {
            "request": request,
            "columns": numeric_columns,
            "distribution_metrics": distribution_metrics
        }
    )


@router.get("/stats/analysis", response_class=HTMLResponse)
async def stats_analysis(request: Request):
    current_df = get_current_df()
    if current_df is None:
        return templates.TemplateResponse(
            "stats/analysis.html",
            {"request": request, "error": "Нет загруженных данных"}
        )
    
    try:
        analyzer = DataAnalyzer(current_df)
        
        # Пример анализа данных
        factor_analysis = analyzer.perform_factor_analysis()
        cluster_analysis = analyzer.perform_cluster_analysis()
        regression_analysis = analyzer.perform_regression_analysis()
        correlation_analysis = analyzer.perform_correlation_analysis()
        
        # Создание визуализаций
        visualizations = create_analysis_visualizations(
            factor_analysis,
            cluster_analysis,
            correlation_analysis,
            regression_analysis
        )
        
        return templates.TemplateResponse(
            "stats/analysis.html",
            {
                "request": request,
                "visualizations": visualizations,
                "factor_analysis": factor_analysis,
                "cluster_analysis": cluster_analysis,
                "regression_analysis": regression_analysis,
                "correlation_analysis": correlation_analysis
            }
        )
    
    except Exception as e:
        return templates.TemplateResponse(
            "stats/analysis.html",
            {"request": request, "error": f"Ошибка при анализе данных: {str(e)}"}
        )


@router.get("/stats/missing_values", response_class=HTMLResponse)
async def missing_values_page(request: Request):
    current_df = get_current_df()
    if current_df is None:
        return templates.TemplateResponse(
            "stats/missing_values.html",
            {"request": request, "error": "Нет загруженных данных"}
        )
    
    analyzer = MissingValuesAnalyzer(current_df)
    analysis = analyzer.analyze()
    
    missing_analysis = {}
    for column, stats in analysis['recommendations'].items():
        missing_analysis[column] = {
            'count': analysis['basic_stats']['columns_with_missing'].get(column, 0),
            'percentage': stats.get('missing_percentage', 0),
            'recommended_method': stats.get('suggested_methods', [{'method': 'none'}])[0]['method']
        }
    
    return templates.TemplateResponse(
        "stats/missing_values.html",
        {
            "request": request,
            "columns": current_df.columns,
            "missing_counts": current_df.isnull().sum(),
            "missing_analysis": missing_analysis
        }
    )


@router.post("/stats/missing_values")
async def handle_missing_values(
    request: Request,
    column: str = Form(...),
    method: str = Form(...),
    value: Optional[str] = Form(None)
):
    global current_df
    
    try:
        imputer = MissingValuesImputer(current_df)
        
        if value and method == 'constant':
            value = float(value)
            imputer.df = imputer.df.with_columns(pl.col(column).fill_null(value))
        else:
            imputer.df = imputer.impute(method, [column])
        
        current_df = imputer.df.to_pandas()
        
        analyzer = MissingValuesAnalyzer(current_df)
        analysis = analyzer.analyze()
        
        missing_analysis = {}
        for col, stats in analysis['recommendations'].items():
            missing_analysis[col] = {
                'count': analysis['basic_stats']['columns_with_missing'].get(col, 0),
                'percentage': stats.get('missing_percentage', 0),
                'recommended_method': stats.get('suggested_methods', [{'method': 'none'}])[0]['method']
            }
        
        table_html = current_df.head(10).to_html(
            classes='table table-striped table-hover',
            index=False
        )
        
        return templates.TemplateResponse(
            "stats/missing_values.html",
            {
                "request": request,
                "columns": current_df.columns,
                "missing_counts": current_df.isnull().sum(),
                "missing_analysis": missing_analysis,
                "table_html": table_html,
                "success_message": f"Пропуски в колонке {column} успешно заполнены методом {method}"
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "stats/missing_values.html",
            {
                "request": request,
                "error": f"Ошибка при заполнении пропусков: {str(e)}"
            }
        )


@router.post("/stats/auto_fill_missing")
async def auto_fill_missing(request: Request):
    current_df = get_current_df()  # Получаем DataFrame из глобального состояния
    
    if current_df is None:
        return JSONResponse(content={"error": "Нет загруженных данных"})
    
    try:
        imputer = MissingValuesImputer(current_df)
        filled_df = imputer.auto_impute()
        
        set_current_df(filled_df.to_pandas())  # Сохраняем обработанный DataFrame
        
        analyzer = MissingValuesAnalyzer(filled_df)
        analysis = analyzer.analyze()
        
        missing_analysis = {}
        for column, stats in analysis['recommendations'].items():
            missing_analysis[column] = {
                'count': analysis['basic_stats']['columns_with_missing'][column],
                'percentage': stats['missing_percentage'],
                'recommended_method': stats['suggested_methods'][0]['method'] if stats['suggested_methods'] else 'none'
            }
        
        # Конвертируем в pandas DataFrame для вызова to_html
        pandas_filled_df = filled_df.to_pandas()
        table_html = pandas_filled_df.head(10).to_html(
            classes='table table-striped table-hover table-bordered',
            index=False,
            border=0,
            justify='left',
            escape=False,
            table_id='data-table'
        )
        
        return templates.TemplateResponse(
            "stats/missing_values.html",
            {
                "request": request,
                "columns": filled_df.columns,
                "missing_counts": filled_df.null_count(),
                "missing_analysis": missing_analysis,
                "table_html": table_html,
                "success_message": "Пропуски успешно заполнены автоматически"
            }
        )
    
    except Exception as e:
        return templates.TemplateResponse(
            "stats/missing_values.html",
            {
                "request": request,
                "error": f"Ошибка при автозаполнении: {str(e)}"
            }
        )


@router.get("/stats/normalization", response_class=HTMLResponse)
async def stats_normalization(request: Request):
    current_df = get_current_df()
    if current_df is None:
        return templates.TemplateResponse(
            "stats/normalization.html",
            {"request": request, "error": "Нет загруженных данных"}
        )
    
    try:
        df = current_df.to_pandas() if isinstance(current_df, pl.DataFrame) else current_df
        
        auto_norm = AutoStandartization(df)
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        auto_methods = auto_norm.auto_selection_method()
        normal_dist = auto_norm.examination_on_normal_distribution()
        outliers_df = auto_norm.get_outliers_three_sigma()
        
        outliers = {}
        for row in outliers_df.iter_rows(named=True):
            outliers[row['column']] = [
                row['mean'],
                row['std'],
                row['n_outliers'],
                row['outlier_percentage']
            ]
        
        return templates.TemplateResponse(
            "stats/normalization.html",
            {
                "request": request,
                "numeric_columns": numeric_columns,
                "auto_methods": auto_methods,
                "normal_dist": normal_dist,
                "outliers": outliers
            }
        )
    
    except Exception as e:
        print(f"Error in stats_normalization: {str(e)}")
        return templates.TemplateResponse(
            "stats/normalization.html",
            {"request": request, "error": f"Ошибка при анализе данных: {str(e)}"}
        )


@router.post("/stats/normalize")
async def normalize_data(
    request: Request,
    method: str = Form(...),
    columns: Optional[List[str]] = Form(None)
):
    current_df = get_current_df()
    if current_df is None:
        return JSONResponse(content={"error": "Нет загруженных данных"}, status_code=400)
    
    try:
        if not isinstance(current_df, pl.DataFrame):
            current_df = pl.from_pandas(current_df)
        
        normalizer = AutoStandartization(current_df)
        
        if method == "auto":
            normalized_df = normalizer.normalize(method='auto')
        else:
            if not columns:
                return JSONResponse(content={"error": "Не указаны колонки для нормализации"}, status_code=400)
            normalized_df = normalizer.normalize(columns=columns, method=method)
        
        set_current_df(normalized_df.to_pandas())
        
        return JSONResponse(content={"success": True})
    
    except Exception as e:
        print(f"Error in normalize_data: {str(e)}")
        return JSONResponse(content={"error": f"Ошибка при нормализации данных: {str(e)}"}, status_code=400)


@router.get("/stats/emissions", response_class=HTMLResponse)
async def emissions_page(request: Request):
    try:
        df = get_current_df()
        if df is None:
            return templates.TemplateResponse(
                "stats/emissions.html",
                {"request": request, "error": "Нет загруженных данных"}
            )
        
        # Преобразуем в polars DataFrame если это pandas DataFrame
        if not isinstance(df, pl.DataFrame):
            df = pl.from_pandas(df)
        
        # Получаем числовые столбцы
        numeric_cols = [col for col in df.columns 
                       if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
        
        return templates.TemplateResponse(
            "stats/emissions.html",
            {
                "request": request,
                "columns": numeric_cols
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "stats/emissions.html",
            {"request": request, "error": f"Ошибка: {str(e)}"}
        )


@router.post("/stats/emissions")
async def handle_emissions(
    request: Request,
    processing_method: str = Form(...),
    column: Optional[str] = Form(None),
    method: Optional[str] = Form(None),
    handling_method: Optional[str] = Form(None)
):
    try:
        df = get_current_df()
        if df is None:
            raise HTTPException(status_code=400, detail="Нет загруженных данных")

        # Преобразуем в polars DataFrame если это pandas DataFrame
        if not isinstance(df, pl.DataFrame):
            df = pl.from_pandas(df)

        emissions = AutoDataEmissions(df)
        
        if processing_method == 'auto':
            # Получаем числовые столбцы используя polars
            numeric_cols = [col for col in df.columns 
                          if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
            
            if len(numeric_cols) == 0:
                raise HTTPException(status_code=400, detail="Нет числовых столбцов для обработки")
            
            first_col = numeric_cols[0]
            before_data = emissions.get_visualization_data(first_col)
            
            # Автоматическая обработка выбросов
            results = emissions.auto_detect_outliers()
            processed_df = df.clone()
            
            for col, info in results.items():
                if info['method'] in ['iqr', 'zscore', 'isolation_forest', 'dbscan', 'lof']:
                    try:
                        processed_df = AutoDataEmissions(processed_df).remove_outliers(col, info['method'])
                    except Exception as e:
                        print(f"Ошибка при обработке выбросов в столбце {col}: {str(e)}")
                        continue
            
            # Обновляем DataFrame в сессии
            set_current_df(processed_df)
            
            # Получаем визуализацию после обработки
            after_data = AutoDataEmissions(processed_df).get_visualization_data(first_col)
            
            return JSONResponse(content={
                "success": True,
                "before_data": before_data,
                "after_data": after_data,
                "message": "Выбросы успешно обработаны автоматически"
            })
        
        else:
            if not column or not method or not handling_method:
                raise HTTPException(status_code=400, detail="Не указаны необходимые параметры")
            
            # Получаем визуализации до обработки
            before_data = emissions.get_visualization_data(column)
            
            if handling_method == 'remove':
                processed_df = emissions.remove_outliers(column, method)
            else:
                processed_df = emissions.cap_outliers(column, method)
            
            # Обновляем DataFrame в сессии
            set_current_df(processed_df)
            
            # Получаем визуализации после обработки
            after_data = AutoDataEmissions(processed_df).get_visualization_data(column)
            
            return JSONResponse(content={
                "success": True,
                "before_data": before_data,
                "after_data": after_data,
                "message": "Выбросы успешно обработаны"
            })
    
    except Exception as e:
        print(f"Ошибка при обработке выбросов: {str(e)}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


@router.get("/export/{format}")
async def export_data(format: str, request: Request):
    try:
        df = get_current_df()
        if df is None:
            raise HTTPException(status_code=400, detail="Нет данных для экспорта")
        
        if format == "csv":
            output = BytesIO()
            df.write_csv(output)
            output.seek(0)
            
            return StreamingResponse(
                output,
                media_type="text/csv",
                headers={
                    "Content-Disposition": "attachment; filename=exported_data.csv"
                }
            )
            
        elif format == "excel":
            output = BytesIO()
            df.write_excel(output)
            output.seek(0)
            
            return StreamingResponse(
                output,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={
                    "Content-Disposition": "attachment; filename=exported_data.xlsx"
                }
            )
            
        elif format == "json":
            return JSONResponse(
                content=df.to_dicts(),
                headers={
                    "Content-Disposition": "attachment; filename=exported_data.json"
                }
            )
            
        elif format == "pdf":
            # Создаем PDF отчет
            buffer = BytesIO()
            pdf = FPDF()  # Используем FPDF вместо PDF
            pdf.add_page()
            
            # Добавляем базовую информацию
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Отчет по анализу данных", ln=True, align="C")
            
            # Добавляем статистику
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Количество строк: {len(df)}", ln=True)
            pdf.cell(0, 10, f"Количество столбцов: {len(df.columns)}", ln=True)
            
            # Добавляем таблицу с данными
            pdf.add_page()
            pdf.set_font("Arial", "B", 12)
            
            # Записываем PDF в буфер
            pdf.output(buffer)
            buffer.seek(0)
            
            return StreamingResponse(
                buffer,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": "attachment; filename=data_report.pdf"
                }
            )
        
        else:
            raise HTTPException(status_code=400, detail="Неподдерживаемый формат экспорта")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/encoding", response_class=HTMLResponse)
async def encoding_page(request: Request):
    current_df = get_current_df()
    if current_df is None:
        return templates.TemplateResponse(
            "stats/encoding.html",
            {"request": request, "error": "Нет загруженных данных"}
        )
    
    try:
        # Преобразуем в pandas DataFrame если это polars DataFrame
        if isinstance(current_df, pl.DataFrame):
            current_df = current_df.to_pandas()
        
        # Получаем категориальные столбцы (строковые и категориальные типы)
        categorical_columns = current_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not categorical_columns:
            return templates.TemplateResponse(
                "stats/encoding.html",
                {
                    "request": request,
                    "error": "В наборе данных нет категориальных столбцов для кодирования"
                }
            )
        
        return templates.TemplateResponse(
            "stats/encoding.html",
            {
                "request": request,
                "categorical_columns": categorical_columns
            }
        )
    except Exception as e:
        print(f"Error in encoding_page: {str(e)}")
        return templates.TemplateResponse(
            "stats/encoding.html",
            {"request": request, "error": f"Ошибка при анализе данных: {str(e)}"}
        )


@router.post("/stats/encoding")
async def handle_encoding(
    request: Request,
    method: str = Form(...),
    columns: list = Form(...)
):
    current_df = get_current_df()
    if current_df is None:
        return templates.TemplateResponse(
            "stats/encoding.html",
            {"request": request, "error": "Нет загруженных данных"}
        )
    
    encoder = AutoCategoricalEncoders(current_df)
    
    if method == "auto":
        encoded_df = encoder.autoselection_method()
    else:
        encoding_decisions = {col: {"method": method} for col in columns}
        encoded_df = encoder.apply_encodings(encoding_decisions)
    
    set_current_df(encoded_df)
    
    return templates.TemplateResponse(
        "stats/encoding.html",
        {
            "request": request,
            "success_message": "Кодирование выполнено успешно",
            "categorical_columns": columns
        }
    )


@router.get("/stats/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    current_df = get_current_df()
    if current_df is None:
        return templates.TemplateResponse(
            "stats/dashboard.html",
            {"request": request, "error": "Нет загруженных данных"}
        )
    
    try:
        # Преобразуем в pandas DataFrame если это polars DataFrame
        if isinstance(current_df, pl.DataFrame):
            current_df = current_df.to_pandas()
        
        # Получаем числовые и категориальные столбцы
        numeric_columns = current_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = current_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Анализ пропущенных значений
        missing_data = pd.DataFrame({
            'Количество пропусков': current_df.isnull().sum(),
            'Процент пропусков': (current_df.isnull().sum() / len(current_df)) * 100
        }).round(2)
        
        # Создаем тепловую карту пропущенных значений
        missing_heatmap = px.imshow(
            current_df.isnull().values,
            labels=dict(x="Столбцы", y="Записи", color="Пропущенные значения"),
            title="Тепловая карта пропущенных значений"
        )
        missing_heatmap.update_layout(height=400)
        missing_heatmap_html = pio.to_html(missing_heatmap, full_html=False)
        
        # Анализ выбросов для числовых столбцов
        outliers_data = {}
        box_plots = []
        for col in numeric_columns:
            Q1 = current_df[col].quantile(0.25)
            Q3 = current_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = current_df[(current_df[col] < lower_bound) | (current_df[col] > upper_bound)][col]
            
            outliers_data[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(current_df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            # Box plot для каждого столбца
            box_fig = px.box(current_df, y=col, title=f"Box Plot: {col}")
            box_fig.update_layout(height=400)
            box_plots.append(pio.to_html(box_fig, full_html=False))
        
        # Корреляционная матрица
        corr_matrix = current_df[numeric_columns].corr()
        heatmap = px.imshow(
            corr_matrix,
            labels=dict(x="Столбцы", y="Столбцы", color="Корреляция"),
            title="Корреляционная матрица"
        )
        heatmap.update_layout(height=600)
        heatmap_html = pio.to_html(heatmap, full_html=False)
        
        # Гистограммы распределения для числовых столбцов
        dist_plots = []
        for col in numeric_columns:
            hist_fig = px.histogram(
                current_df, 
                x=col,
                title=f"Распределение: {col}",
                marginal="box"
            )
            hist_fig.update_layout(height=400)
            dist_plots.append(pio.to_html(hist_fig, full_html=False))
        
        # Scatter matrix для числовых столбцов
        if len(numeric_columns) > 1:
            scatter_matrix = px.scatter_matrix(
                current_df[numeric_columns[:4]],  # Берем первые 4 столбца для наглядности
                title="Матрица рассеяния"
            )
            scatter_matrix.update_layout(height=800)
            scatter_matrix_html = pio.to_html(scatter_matrix, full_html=False)
        else:
            scatter_matrix_html = None
        
        # Оценка качества данных
        data_quality = {
            'total_rows': len(current_df),
            'total_columns': len(current_df.columns),
            'missing_cells': current_df.isnull().sum().sum(),
            'missing_percentage': (current_df.isnull().sum().sum() / (len(current_df) * len(current_df.columns))) * 100,
            'duplicated_rows': current_df.duplicated().sum(),
            'numeric_columns': len(numeric_columns),
            'categorical_columns': len(categorical_columns),
            'memory_usage': current_df.memory_usage(deep=True).sum() / 1024 / 1024  # в МБ
        }
        
        return templates.TemplateResponse(
            "stats/dashboard.html",
            {
                "request": request,
                "columns": current_df.columns.tolist(),
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "missing_data": missing_data.to_dict('index'),
                "missing_heatmap": missing_heatmap_html,
                "outliers_data": outliers_data,
                "box_plots": box_plots,
                "heatmap": heatmap_html,
                "dist_plots": dist_plots,
                "scatter_matrix": scatter_matrix_html,
                "data_quality": data_quality
            }
        )
    
    except Exception as e:
        print(f"Error in dashboard_page: {str(e)}")
        return templates.TemplateResponse(
            "stats/dashboard.html",
            {"request": request, "error": f"Ошибка при создании дашборда: {str(e)}"}
        )