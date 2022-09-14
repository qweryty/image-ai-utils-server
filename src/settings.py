import os.path
from typing import Optional

from diffusers.utils import DIFFUSERS_CACHE
from dotenv import load_dotenv
from pydantic import BaseSettings, Field, validator

from utils import resolve_path


class Settings(BaseSettings):
    USERNAME: str = Field(..., env='IMAGE_AI_UTILS_USERNAME')
    PASSWORD: str = Field(..., env='IMAGE_AI_UTILS_PASSWORD')
    HOST: str = Field('0.0.0.0', env='HOST')
    PORT: int = Field(7331, env='PORT')
    LOG_LEVEL: str = Field('DEBUG', env='LOG_LEVEL')
    FILE_LOG_LEVEL: str = Field('ERROR', env='FILE_LOG_LEVEL')
    DIFFUSERS_CACHE_PATH: str = Field(DIFFUSERS_CACHE, env='DIFFUSERS_CACHE_PATH')
    USE_OPTIMIZED_MODE: bool = Field(True, env='USE_OPTIMIZED_MODE')

    # TODO make abspath from current file
    @validator('DIFFUSERS_CACHE_PATH')
    def make_abspath(cls, path: Optional[str]) -> Optional[str]:
        if path is None or os.path.isabs(path):
            return path

        return resolve_path(path)


load_dotenv()
settings = Settings()
