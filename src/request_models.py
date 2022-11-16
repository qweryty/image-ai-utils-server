"""
Tools for converting web socket requests into interpretable values.

Classes:
    BaseDiffusionRequest - for performing diffusion; meant to be sub-classed
    BaseImageGenerationRequest - for text-guided diffusion;
        meant to be sub-classed
    TextToImageRequest - for text-to-image diffusion
    ImageToImageRequest - for text-guided image-to-image diffusion
    InpaintingRequest - for text-guided mask infill diffusion
    GoBigRequest - for scaling up an image with diffusion
    MakeTilableRequest - for converting an image's edges to be tilable
    UpscaleRequest - for scaling up an image with RealESRGAN
    FaceRestorationRequest - for fixing faces with GFPGAN
    ImageArrayResponse - for returning multiple validated images
    MakeTilableResponse - for returning validated mask
    ImageResponse - for returning a single validated image
"""


# pylint: disable=no-self-argument
# `cls` is important here

# pylint: disable=too-few-public-methods

from typing import Optional, List

from PIL import Image
from pydantic import BaseModel, Field, validator

from consts import (
    ImageFormat,
    MIN_SEED,
    MAX_SEED,
    ScalingMode,
    ESRGANModel,
    GFPGANModel,
)
from utils import image_to_base64url


class BaseDiffusionRequest(BaseModel):
    prompt: str = Field(...)
    output_format: ImageFormat = ImageFormat.PNG
    num_inference_steps: int = Field(50, gt=0)
    guidance_scale: float = Field(6.0)
    seed: Optional[int] = Field(None, ge=MIN_SEED, le=MAX_SEED)
    batch_size: int = Field(6, gt=0)
    try_smaller_batch_on_fail: bool = True


class BaseImageGenerationRequest(BaseDiffusionRequest):
    num_variants: int = Field(4, gt=0)
    scaling_mode: ScalingMode = ScalingMode.GROW


class TextToImageRequest(BaseImageGenerationRequest):
    aspect_ratio: float = Field(1.0, gt=0)  # width/height


class ImageToImageRequest(BaseImageGenerationRequest):
    source_image: bytes
    strength: float = Field(0.8, ge=0, le=1)


class InpaintingRequest(ImageToImageRequest):
    mask: Optional[bytes] = None


class GoBigRequest(BaseDiffusionRequest):
    image: bytes
    use_real_esrgan: bool = True
    esrgan_model: ESRGANModel = ESRGANModel.GENERAL_X4_V3
    maximize: bool = True
    strength: float = Field(0.3, ge=0, le=1)
    target_width: int = Field(..., gt=0)
    target_height: int = Field(..., gt=0)
    overlap: int = Field(64, gt=0, lt=512)


class MakeTilableRequest(BaseImageGenerationRequest):
    source_image: bytes
    border_width: int = Field(50, gt=0, lt=256)
    border_softness: float = Field(0.5, ge=0, le=1)
    strength: float = Field(0.8, ge=0, le=1)


class UpscaleRequest(BaseModel):
    image: bytes
    model: ESRGANModel = ESRGANModel.GENERAL_X4_V3
    target_width: int = Field(..., gt=0)
    target_height: int = Field(..., gt=0)
    maximize: bool = True


class FaceRestorationRequest(BaseModel):
    image: bytes
    model_type: GFPGANModel
    use_real_esrgan: bool = True
    bg_tile: int = 400
    upscale: int = 2
    aligned: bool = False
    only_center_face: bool = False


class ImageArrayResponse(BaseModel):
    images: List[bytes]

    @validator("images", pre=True)
    def images_to_bytes(cls, val_images: List) -> List:
        return [
            image_to_base64url(image) if isinstance(image, Image.Image) else image
            for image in val_images
        ]


class MakeTilableResponse(ImageArrayResponse):
    mask: bytes

    @validator("mask", pre=True)
    def mask_to_bytes(cls, val_mask):
        if isinstance(val_mask, Image.Image):
            val_mask = image_to_base64url(val_mask)
        return val_mask


class ImageResponse(BaseModel):
    image: bytes

    @validator("image", pre=True)
    def image_to_bytes(cls, v_image):
        if isinstance(v_image, Image.Image):
            v_image = image_to_base64url(v_image)
        return v_image
