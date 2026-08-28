from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Research Assistant"
    app_version: str = "0.1.0"
    llm_provider: str = "local"
    llm_model: str = ""
    vector_db_path: str = "./chroma_db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()