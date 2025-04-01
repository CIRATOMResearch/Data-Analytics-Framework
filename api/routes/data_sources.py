from fastapi import APIRouter, Request, Form, UploadFile, File, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
import pandas as pd
import os
from typing import Optional

from database.mysql_connector import MySQLConnector
from database.postgresql_connector import PostgreSQLConnector
from database.mongodb_connector import MongoDBConnector
from database.cassandra_connector import CassandraConnector

from services.data_conversion import convert_to_csv
from services.preview import StatisticalAnalysis as Preview
from core.state import set_current_df

from models.schemas import DataUpload



# fo195883115@gmail.com
# egorvah012345

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# Глобальная переменная для хранения данных
current_df = None
UPLOAD_DIR = "uploads"


@router.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    try:
        # Сохранение файла
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Попробуем прочитать CSV-файл
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                raise HTTPException(status_code=400, detail="Загруженный файл пуст или некорректен")
        except pd.errors.EmptyDataError:
            raise HTTPException(status_code=400, detail="Нет данных для парсинга")
        except pd.errors.ParserError:
            raise HTTPException(status_code=400, detail="Ошибка парсинга csv файла")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка обработки файла: {str(e)}")
        
        # Загрузка DataFrame
        set_current_df(df)  # Сохраняем DataFrame глобально
        
        # Создание HTML таблицы
        table_html = df.head(10).to_html(
            classes='table table-striped table-hover table-bordered',
            index=False,
            border=0,
            justify='left',
            escape=False,
            table_id='data-table'
        )
        
        return JSONResponse(
            content={
                "success": True,
                "table_html": table_html,
                "message": "Файл успешно загружен"
            }
        )
    
    except HTTPException as http_exc:
        return JSONResponse(content={"error": http_exc.detail}, status_code=http_exc.status_code)
    except Exception as e:
        return JSONResponse(content={"error": f"Ошибка: {str(e)}"}, status_code=500)


@router.post("/connect/mysql")
async def connect_mysql(
    request: Request,
    host: str = Form(...),
    user: str = Form(...),
    password: str = Form(...),
    database: str = Form(...),
    query: str = Form(...)
):
    global current_df
    
    try:
        connector = MySQLConnector(host, user, password, database)
        if not connector.connect():
            return templates.TemplateResponse("data_sources.html", {"request": request, "error": "Ошибка подключения к MySQL"})
        
        df = connector.execute_query(query)
        connector.close()
        
        if df is None:
            return templates.TemplateResponse("data_sources.html", {"request": request, "error": "Ошибка выполнения запроса"})
        
        current_df = df
        
        # Создаем HTML страницу
        table_html = df.head(10).to_html(
            classes='table table-striped table-hover table-bordered',
            index=False,
            border=0,
            justify='left',
            escape=False,
            table_id='data-table'
        )
        
        preview = Preview(df)
        analysis_results = preview.pv()
        
        return templates.TemplateResponse(
            "data_sources.html",
            {
                "request": request,
                "table_html": table_html,
                "analysis_results": analysis_results,
                "success_message": "Данные успешно загружены из MySQL"
            }
        )
    
    except Exception as e:
        return templates.TemplateResponse("data_sources.html", {"request": request, "error": f"Ошибка: {str(e)}"})


@router.post("/connect/postgresql")
async def connect_postgresql(
    request: Request,
    host: str = Form(...),
    port: int = Form(...),
    user: str = Form(...),
    password: str = Form(...),
    database: str = Form(...),
    query: str = Form(...)
):
    global current_df
    
    try:
        connector = PostgreSQLConnector(host, user, password, database, port)
        if not connector.connect():
            return templates.TemplateResponse("data_sources.html", {"request": request, "error": "Ошибка подключения к PostgreSQL"})
        
        df = connector.execute_query(query)
        connector.close()
        
        if df is None:
            return templates.TemplateResponse("data_sources.html", {"request": request, "error": "Ошибка выполнения запроса"})
        
        current_df = df
        
        # Create HTML table
        table_html = df.head(10).to_html(
            classes='table table-striped table-hover table-bordered',
            index=False,
            border=0,
            justify='left',
            escape=False,
            table_id='data-table'
        )
        
        preview = Preview(df)
        analysis_results = preview.pv()
        
        return templates.TemplateResponse(
            "data_sources.html",
            {
                "request": request,
                "table_html": table_html,
                "analysis_results": analysis_results,
                "success_message": "Данные успешно загружены из PostgreSQL"
            }
        )
    
    except Exception as e:
        return templates.TemplateResponse("data_sources.html", {"request": request, "error": f"Ошибка: {str(e)}"})


@router.post("/connect/mongodb")
async def connect_mongodb(
    request: Request,
    host: str = Form(...),
    port: int = Form(...),
    username: str = Form(None),
    password: str = Form(None),
    database: str = Form(...),
    collection: str = Form(...)
):
    global current_df
    
    try:
        connector = MongoDBConnector(host, port, username, password)
        if not connector.connect():
            return templates.TemplateResponse("data_sources.html", {"request": request, "error": "Ошибка подключения к MongoDB"})
        
        df = connector.get_collection_data(database, collection)
        connector.close()
        
        if df is None:
            return templates.TemplateResponse("data_sources.html", {"request": request, "error": "Ошибка получения данных"})
        
        current_df = df
        
        # Create HTML table
        table_html = df.head(10).to_html(
            classes='table table-striped table-hover table-bordered',
            index=False,
            border=0,
            justify='left',
            escape=False,
            table_id='data-table'
        )
        
        preview = Preview(df)
        analysis_results = preview.pv()
        
        return templates.TemplateResponse(
            "data_sources.html",
            {
                "request": request,
                "table_html": table_html,
                "analysis_results": analysis_results,
                "success_message": "Данные успешно загружены из MongoDB"
            }
        )
    
    except Exception as e:
        return templates.TemplateResponse("data_sources.html", {"request": request, "error": f"Ошибка: {str(e)}"})


@router.post("/connect/cassandra")
async def connect_cassandra(
    request: Request,
    host: str = Form(...),
    port: int = Form(...),
    username: str = Form(None),
    password: str = Form(None),
    keyspace: str = Form(...),
    query: str = Form(...)
):
    global current_df
    
    try:
        connector = CassandraConnector([host], port, username, password)
        if not connector.connect():
            return templates.TemplateResponse("data_sources.html", {"request": request, "error": "Ошибка подключения к Cassandra"})
        
        df = connector.execute_query(keyspace, query)
        connector.close()
        
        if df is None:
            return templates.TemplateResponse("data_sources.html", {"request": request, "error": "Ошибка выполнения запроса"})
        
        current_df = df
        
        # Create HTML table
        table_html = df.head(10).to_html(
            classes='table table-striped table-hover table-bordered',
            index=False,
            border=0,
            justify='left',
            escape=False,
            table_id='data-table'
        )
        
        preview = Preview(df)
        analysis_results = preview.pv()
        
        return templates.TemplateResponse(
            "data_sources.html",
            {
                "request": request,
                "table_html": table_html,
                "analysis_results": analysis_results,
                "success_message": "Данные успешно загружены из Cassandra"
            }
        )
    
    except Exception as e:
        return templates.TemplateResponse("data_sources.html", {"request": request, "error": f"Ошибка: {str(e)}"})


@router.get("/data-sources", response_class=HTMLResponse)
async def data_sources_page(request: Request):
    return templates.TemplateResponse(
        "data_sources.html",
        {"request": request}
    )


@router.get("/home", response_class=HTMLResponse)
async def home_redirect():
    return RedirectResponse(url="/")