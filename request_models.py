from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field


class ESRGANModel(str, Enum):
    # General
    GENERAL_X4_V3 = 'general_x4_v3'
    X4_PLUS = 'x4_plus'
    X2_PLUS = 'x2_plus'
    ESRNET_X4_PLUS = 'x4_plus'
    OFFICIAL_X4 = 'official_x4'

    # Anime/Illustrations
    X4_PLUS_ANIME_6B = 'x4_plus_anime_6b'

    # Anime video
    ANIME_VIDEO_V3 = 'anime_video_v3'


class ScalingMode(str, Enum):
    SHRINK = 'shrink'
    GROW = 'grow'


class ImageFormat(str, Enum):
    PNG = 'PNG'
    JPEG = 'JPEG'
    BMP = 'BMP'


class WebSocketResponseStatus(str, Enum):
    FINISHED = 'finished'
    PROGRESS = 'progress'


MIN_SEED = -0x8000_0000_0000_0000
MAX_SEED = 0xffff_ffff_ffff_ffff


class BaseDiffusionRequest(BaseModel):
    prompt: str = Field(...)
    num_variants: int = Field(4, gt=0)
    output_format: ImageFormat = ImageFormat.PNG
    num_inference_steps: int = Field(50, gt=0)
    guidance_scale: float = Field(7.5)
    seed: Optional[int] = Field(None, ge=MIN_SEED, le=MAX_SEED)
    batch_size: int = Field(7, gt=0)
    try_smaller_batch_on_fail: bool = True
    scaling_mode: ScalingMode = ScalingMode.GROW


class TextToImageRequest(BaseDiffusionRequest):
    aspect_ratio: float = Field(1., gt=0)  # width/height


class ImageToImageRequest(BaseDiffusionRequest):
    source_image: bytes
    strength: float = Field(0.8, ge=0, le=1)


class InpaintingRequest(ImageToImageRequest):
    mask: Optional[bytes] = None


class GoBigRequest(BaseDiffusionRequest):
    image: bytes
    use_real_esrgan: bool = True
    init_strength: float = Field(.5, ge=0, le=1)
    target_width: int = Field(..., gt=0)
    target_height: int = Field(..., gt=0)


class UpscaleRequest(BaseModel):
    image: bytes
    model: ESRGANModel = ESRGANModel.GENERAL_X4_V3
    target_width: int = Field(..., gt=0)
    target_height: int = Field(..., gt=0)


class ImageArrayResponse(BaseModel):
    images: List[bytes]


class UpscaleResponse(BaseModel):
    image: bytes
