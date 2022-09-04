from settings import settings  # noqa
import logging
import mimetypes
from base64 import b64encode
from enum import Enum
from io import BytesIO
from typing import Optional, List, Tuple

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi import status
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
from torch import autocast

import esrgan_upscaler
from logging_settings import LOGGING
from universal_pipeline import StableDiffusionUniversalPipeline, preprocess, preprocess_mask
from utils import base64url_to_image, image_to_base64url

logger = logging.getLogger(__name__)
security = HTTPBasic()


async def authorize(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != settings.USERNAME or credentials.password != settings.PASSWORD:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


app = FastAPI(dependencies=[Depends(authorize)])
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
    target_width: int = Field(..., gt=0)
    target_height: int = Field(..., gt=0)


class ImageArrayResponse(BaseModel):
    images: List[bytes]


class UpscaleResponse(BaseModel):
    image: bytes


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


def size_from_aspect_ratio(aspect_ratio: float) -> Tuple[int, int]:
    if aspect_ratio > 1:
        height = 512
        width = int(height * aspect_ratio)
        width -= width % 64
    else:
        width = 512
        height = int(width / aspect_ratio)
        height -= height % 64

    return width, height


# TODO task queue and web sockets?
#  (or can set up an external scheduler and use this as internal endpoint)
@app.post('/text_to_image')
def text_to_image(request: TextToImageRequest) -> ImageArrayResponse:
    width, height = size_from_aspect_ratio(request.aspect_ratio)
    return do_diffusion(
        request, pipeline.__call__, height=height, width=width
    )


@app.post('/image_to_image')
async def image_to_image(request: ImageToImageRequest) -> ImageArrayResponse:
    source_image = base64url_to_image(request.source_image)
    aspect_ratio = source_image.width / source_image.height
    size = size_from_aspect_ratio(aspect_ratio)

    source_image = source_image.resize(size)

    with autocast('cuda'):
        preprocessed_source_image = preprocess(source_image).to(pipeline.device)
        preprocessed_alpha = None
        if source_image.mode == 'RGBA':
            preprocessed_alpha = 1 - preprocess_mask(
                source_image.getchannel('A')
            ).to(pipeline.device)

        if preprocessed_alpha is not None and not preprocessed_alpha.any():
            preprocessed_alpha = None

    return do_diffusion(
        request,
        pipeline.image_to_image,
        init_image=preprocessed_source_image,
        strength=request.strength,
        alpha=preprocessed_alpha,
    )


@app.post('/inpainting')
async def inpainting(request: InpaintingRequest) -> ImageArrayResponse:
    source_image = base64url_to_image(request.source_image)
    aspect_ratio = source_image.width / source_image.height
    size = size_from_aspect_ratio(aspect_ratio)

    source_image = source_image.resize(size)
    mask = None
    if request.mask:
        mask = base64url_to_image(request.mask).resize(size)

    with autocast('cuda'):
        preprocessed_source_image = preprocess(source_image).to(pipeline.device)
        preprocessed_mask = None
        if mask is not None:
            preprocessed_mask = preprocess_mask(mask).to(pipeline.device)

        if preprocessed_mask is not None and not preprocessed_mask.any():
            preprocessed_mask = None

        preprocessed_alpha = None
        if source_image.mode == 'RGBA':
            preprocessed_alpha = 1 - preprocess_mask(
                source_image.getchannel('A')
            ).to(pipeline.device)

        if preprocessed_alpha is not None and not preprocessed_alpha.any():
            preprocessed_alpha = None

        if preprocessed_alpha is not None:
            if preprocessed_mask is not None:
                preprocessed_mask = torch.max(preprocessed_mask, preprocessed_alpha)
            else:
                preprocessed_mask = preprocessed_alpha

    # TODO return error if mask empty
    return do_diffusion(
        request,
        pipeline.image_to_image,
        init_image=preprocessed_source_image,
        strength=request.strength,
        mask=preprocessed_mask,
        alpha=preprocessed_alpha,
    )


@app.post('/gobig')
async def gobig(request: GoBigRequest) -> UpscaleResponse:
    pass


@app.post('/upscale')
async def upscale(request: UpscaleRequest) -> UpscaleResponse:
    # TODO multiple steps to make bigger than 4x
    return UpscaleResponse(
        image=image_to_base64url(
            esrgan_upscaler.upscale(image=base64url_to_image(request.image))
        )
    )


@app.get('/ping')
async def ping():
    return


if __name__ == '__main__':
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, log_config=LOGGING)
