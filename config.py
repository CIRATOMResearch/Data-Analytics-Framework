from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "ATOM Analytics"
    secret_key: str
    database_url: str

    class Config:
        env_file = ".env"

settings = Settings()