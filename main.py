from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from config import settings

from api.routes import stats, data_sources


import uvicorn

# Инициализация приложения на Fastapi
app = FastAPI(
    title=settings.app_name,
    description="Data Analysis and Preprocessing Platform",
    version="1.0.0"
)

# Ключи
app.add_middleware(
    SessionMiddleware, 
    secret_key="your-super-secret-key"
)

# Конфигурация статических файлов
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Включить маршрутизаторы
app.include_router(data_sources.router, prefix="", tags=["Data Sources"])
app.include_router(stats.router, prefix="", tags=["Statistics"])

# Корень
@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(
        "stats/home.html",
        {"request": request}
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True ,workers=4)