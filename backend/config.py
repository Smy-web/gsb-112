from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    PYANNOTE_AUTH_TOKEN: str = ""
    SMTP_HOST: str = "smtp.example.com"
    SMTP_PORT: int = 465
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    DATABASE_URL: str = "sqlite:///./merit_record.db"
    TEMP_AUDIO_DIR: str = "./temp_audio"
    OUTPUT_DIR: str = "./output"
    APP_NAME: str = "寺庙修缮募捐功德纪要系统"
    VERSION: str = "1.0.0"

    class Config:
        env_file = ".env"


settings = Settings()
