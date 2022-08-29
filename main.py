import mimetypes
from base64 import b64encode
from enum import Enum
from io import BytesIO
from typing import Optional, List

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from torch import autocast

from universal_pipeline import StableDiffusionUniversalPipeline

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)

# TODO settings
MAX_PIXELS_PER_BATCH = 512 * 512 * 7  # better way to determine batch size?
MIN_SEED = -0x8000_0000_0000_0000
MAX_SEED = 0xffff_ffff_ffff_ffff

# TODO universal pipeline to optimize memory
pipeline = StableDiffusionUniversalPipeline.from_pretrained(
    'CompVis/stable-diffusion-v1-4',
    revision='fp16',
    torch_dtype=torch.bfloat16,
    use_auth_token=True
).to('cuda')


def dummy_checker(images, **kwargs): return images, False


pipeline.safety_checker = dummy_checker  # Disabling safety


class ImageFormat(str, Enum):
    PNG = 'PNG'
    JPEG = 'JPEG'
    BMP = 'BMP'


class BaseDiffusionRequest(BaseModel):
    prompt: str = Field(...)
    output_format: ImageFormat = ImageFormat.PNG
    num_inference_steps: int = Field(50, gt=0)
    guidance_scale: float = Field(7.5)
    seed: Optional[int] = Field(None, ge=MIN_SEED, le=MAX_SEED)


class TextToImageRequest(BaseDiffusionRequest):
    aspect_ratio: float = Field(1., gt=0)  # width/height
    num_variants: int = Field(6, gt=0)


class ImageToImageRequest(BaseDiffusionRequest):
    source_image: bytes
    strength: int = Field(0.8, ge=0, le=1)
    mask: Optional[bytes] = None
    num_variants: int = Field(6, gt=1)


class GoBigRequest(BaseDiffusionRequest):
    image: bytes
    use_real_esrgan: bool = True
    init_strength: float = Field(.5, ge=0, le=1)
    target_width: int = Field(..., gt=0)
    target_height: int = Field(..., gt=0)
    # TODO factor


class UpscaleRequest(BaseModel):
    image: bytes
    target_width: int = Field(..., gt=0)
    target_height: int = Field(..., gt=0)
    # TODO factor


class ImageArrayResponse(BaseModel):
    images: List[bytes]


# TODO task queue and web sockets?
#  (or can set up an external scheduler and use this as internal endpoint)
@app.post('/text_to_image')
def text_to_image(request: TextToImageRequest) -> ImageArrayResponse:
    if request.aspect_ratio > 1:
        height = 512
        width = int(height * request.aspect_ratio)
        width -= width % 64
    else:
        width = 512
        height = int(width / request.aspect_ratio)
        height -= height % 64

    if request.seed is not None:
        generator = torch.Generator('cuda').manual_seed(request.seed)
    else:
        generator = None

    total_pixels = width * height * request.num_variants
    if total_pixels > MAX_PIXELS_PER_BATCH:
        batch_size = MAX_PIXELS_PER_BATCH // (width * height)
        num_batches = request.num_variants // batch_size

        prompts = [request.prompt] * batch_size
    else:
        num_batches = 0
        batch_size = 0
        prompts = []

    last_batch_size = request.num_variants - batch_size * num_batches
    images = []

    with autocast('cuda'):
        for i in range(num_batches):
            new_images = pipeline(
                prompts,
                num_inference_steps=request.num_inference_steps,
                generator=generator,
                guidance_scale=request.guidance_scale,
                height=height,
                width=width,
            )['sample']
            images.extend(new_images)

        if last_batch_size:
            new_images = pipeline(
                [request.prompt] * last_batch_size,
                num_inference_steps=request.num_inference_steps,
                generator=generator,
                guidance_scale=request.guidance_scale,
                height=height,
                width=width,
            )['sample']
            images.extend(new_images)

    encoded_images = []
    data_string = f'data:{mimetypes.types_map[f".{request.output_format.lower()}"]};base64,' \
        .encode()
    for image in images:
        buffer = BytesIO()
        image.save(buffer, format=request.output_format)
        encoded_images.append(data_string + b64encode(buffer.getvalue()))

    return ImageArrayResponse(images=encoded_images)


@app.post('/image_to_image')
def image_to_image() -> ImageArrayResponse:
    return ImageArrayResponse(images=[])


@app.post('/gobig')
def gobig(request: GoBigRequest):
    pass


@app.post('/upscale')
def upscale(request: UpscaleRequest):
    pass


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
