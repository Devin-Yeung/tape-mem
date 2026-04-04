from pydantic_settings import BaseSettings, SettingsConfigDict


class Env(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False)

    openai_compatible_base_url: str
    openai_compatible_api_key: str
    llm_model: str
