from dotenv import load_dotenv
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    USERNAME: str = Field(..., env='USERNAME')
    PASSWORD: str = Field(..., env='PASSWORD')
    HOST: str = Field('0.0.0.0', env='HOST')
    PORT: int = Field(7331, env='PORT')
    LOG_LEVEL: str = Field('DEBUG', env='LOG_LEVEL')


load_dotenv()
settings = Settings()
