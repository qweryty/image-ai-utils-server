import logging
import mimetypes
from base64 import b64encode, b64decode
from enum import Enum
from io import BytesIO
from typing import Optional, List

import torch
import uvicorn
from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from torch import autocast

from universal_pipeline import StableDiffusionUniversalPipeline, preprocess

logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)

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
    num_variants: int = Field(6, gt=0)
    output_format: ImageFormat = ImageFormat.PNG
    num_inference_steps: int = Field(50, gt=0)
    guidance_scale: float = Field(7.5)
    seed: Optional[int] = Field(None, ge=MIN_SEED, le=MAX_SEED)
    batch_size: int = Field(7, gt=0)  # TODO determine automatically
    try_smaller_batch_on_fail: bool = True


class TextToImageRequest(BaseDiffusionRequest):
    aspect_ratio: float = Field(1., gt=0)  # width/height


class ImageToImageRequest(BaseDiffusionRequest):
    source_image: bytes
    strength: float = Field(0.8, ge=0, le=1)
    mask: Optional[bytes] = None


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


def do_diffusion(request: BaseDiffusionRequest, diffusion_method, **kwargs) -> ImageArrayResponse:
    if request.seed is not None:
        generator = torch.Generator('cuda').manual_seed(request.seed)
    else:
        generator = None

    with autocast('cuda'):
        batch_size = request.batch_size
        while True:
            try:
                num_batches = request.num_variants // batch_size
                prompts = [request.prompt] * batch_size
                last_batch_size = request.num_variants - batch_size * num_batches
                images = []
                for i in range(num_batches):
                    new_images = diffusion_method(
                        prompt=prompts,
                        num_inference_steps=request.num_inference_steps,
                        generator=generator,
                        guidance_scale=request.guidance_scale,
                        **kwargs
                    )['sample']
                    images.extend(new_images)

                if last_batch_size:
                    new_images = diffusion_method(
                        prompt=[request.prompt] * last_batch_size,
                        num_inference_steps=request.num_inference_steps,
                        generator=generator,
                        guidance_scale=request.guidance_scale,
                        **kwargs
                    )['sample']
                    images.extend(new_images)
                break
            except RuntimeError as e:
                if request.try_smaller_batch_on_fail and batch_size > 1:
                    logger.warning(f'Batch size {batch_size} was too large, trying smaller')
                    batch_size -= 1
                else:
                    raise e

    encoded_images = []
    data_string = f'data:{mimetypes.types_map[f".{request.output_format.lower()}"]};base64,' \
        .encode()
    for image in images:
        buffer = BytesIO()
        image.save(buffer, format=request.output_format)
        encoded_images.append(data_string + b64encode(buffer.getvalue()))

    return ImageArrayResponse(images=encoded_images)

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
    return do_diffusion(
        request, pipeline.__call__, height=height, width=width
    )


# TODO use common utils
def base64url_to_image(source: bytes) -> Image.Image:
    _, data = source.split(b',')
    return Image.open(BytesIO(b64decode(data)))


@app.post('/image_to_image')
def image_to_image(request: ImageToImageRequest) -> ImageArrayResponse:
    source_image = base64url_to_image(request.source_image)
    aspect_ratio = source_image.width / source_image.height
    if aspect_ratio > 1:
        height = 512
        width = int(height * aspect_ratio)
        width -= width % 64
    else:
        width = 512
        height = int(width / aspect_ratio)
        height -= height % 64

    source_image = source_image.resize((width, height))
    with autocast('cuda'):
        preprocessed_source_image = preprocess(source_image).to(pipeline.device)
    return do_diffusion(
        request,
        pipeline.image_to_image,
        init_image=preprocessed_source_image,
        strength=request.strength,
    )


@app.post('/gobig')
def gobig(request: GoBigRequest):
    pass


@app.post('/upscale')
def upscale(request: UpscaleRequest):
    pass


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
