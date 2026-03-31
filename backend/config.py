from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Redis
    REDIS_URL: str = "redis://redis:6379/0"

    # Storage
    STORAGE_BACKEND: str = "local"          # "local" or "s3"
    LOCAL_STORAGE_PATH: str = "/storage"

    # S3 (optional)
    S3_BUCKET: str = ""
    S3_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""

    # File size limits
    MAX_IMAGE_SIZE_MB: int = 50
    MAX_VIDEO_SIZE_MB: int = 500

    # AI model paths
    REALESRGAN_MODEL_PATH: str = "/models/realesrgan/RealESRGAN_x4plus.pth"
    U2NET_MODEL_PATH: str = "/models/u2net/u2net.pth"

    # Database
    DATABASE_URL: str = "sqlite:////storage/jobs.db"

    # Video safety cap
    MAX_VIDEO_FRAMES: int = 300

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
