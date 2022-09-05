import functools

from settings import settings  # noqa
from logging_settings import LOGGING  # noqa

import json
import logging
import mimetypes
from base64 import b64encode
from io import BytesIO
from typing import Callable

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, WebSocket, status, WebSocketDisconnect
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from torch import autocast

import esrgan_upscaler
from request_models import BaseDiffusionRequest, ImageArrayResponse, ImageToImageRequest, \
    TextToImageRequest, GoBigRequest, UpscaleResponse, UpscaleRequest, InpaintingRequest, \
    WebSocketResponseStatus
from universal_pipeline import StableDiffusionUniversalPipeline, preprocess, preprocess_mask
from utils import base64url_to_image, image_to_base64url, size_from_aspect_ratio

logger = logging.getLogger(__name__)
security = HTTPBasic()


async def authorize(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != settings.USERNAME or credentials.password != settings.PASSWORD:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


async def authorize_web_socket(websocket: WebSocket) -> bool:
    credentials = await websocket.receive_json()
    if credentials.get('username') != settings.USERNAME or \
            credentials.get('password') != settings.PASSWORD:
        await websocket.close(status.WS_1008_POLICY_VIOLATION)
        return False

    return True


def websocket_handler(path, app):
    def decorator(handler):
        @functools.wraps(handler)
        async def wrapper(websocket: WebSocket):
            await websocket.accept()
            try:
                if not await authorize_web_socket(websocket):
                    return

                await handler(websocket)
            except WebSocketDisconnect:
                return

        return app.websocket(path)(wrapper)
    return decorator


app = FastAPI(dependencies=[Depends(authorize)])
app.add_middleware(GZipMiddleware, minimum_size=1000)

try:
    pipeline = StableDiffusionUniversalPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4',
        revision='fp16',
        torch_dtype=torch.bfloat16,
        use_auth_token=True,
        cache_dir=settings.DIFFUSERS_CACHE_PATH,
    ).to('cuda')
except Exception as e:
    logger.exception(e)
    raise e


def dummy_checker(images, **kwargs): return images, False


pipeline.safety_checker = dummy_checker  # Disabling safety


async def do_diffusion(
        request: BaseDiffusionRequest,
        diffusion_method: Callable,
        websocket: WebSocket,
        **kwargs
) -> ImageArrayResponse:
    if request.seed is not None:
        generator = torch.Generator('cuda').manual_seed(request.seed)
    else:
        generator = None

    with autocast('cuda'):
        with torch.inference_mode():
            for batch_size in range(min(request.batch_size, request.num_variants), 0, -1):
                try:
                    num_batches = request.num_variants // batch_size
                    prompts = [request.prompt] * batch_size
                    last_batch_size = request.num_variants - batch_size * num_batches
                    images = []

                    for i in range(num_batches):
                        async def progress_callback(batch_step: int, total_batch_steps: int):
                            current_step = i * total_batch_steps + batch_step
                            total_steps = num_batches * total_batch_steps
                            if last_batch_size:
                                total_steps += total_batch_steps
                            progress = current_step / total_steps
                            await websocket.send_json(
                                {'status': WebSocketResponseStatus.PROGRESS, 'progress': progress}
                            )

                        new_images = (await diffusion_method(
                            prompt=prompts,
                            num_inference_steps=request.num_inference_steps,
                            generator=generator,
                            guidance_scale=request.guidance_scale,
                            progress_callback=progress_callback,
                            **kwargs
                        ))['sample']
                        images.extend(new_images)

                    if last_batch_size:
                        async def progress_callback(batch_step: int, total_batch_steps: int):
                            current_step = num_batches * total_batch_steps + batch_step
                            total_steps = (num_batches + 1) * total_batch_steps
                            progress = current_step / total_steps
                            await websocket.send_json(
                                {'status': WebSocketResponseStatus.PROGRESS, 'progress': progress}
                            )

                        new_images = (await diffusion_method(
                            prompt=[request.prompt] * last_batch_size,
                            num_inference_steps=request.num_inference_steps,
                            generator=generator,
                            guidance_scale=request.guidance_scale,
                            progress_callback=progress_callback,
                            **kwargs
                        ))['sample']
                        images.extend(new_images)
                    break
                except RuntimeError as e:
                    if request.try_smaller_batch_on_fail:
                        logger.warning(f'Batch size {batch_size} was too large, trying smaller')
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail=f'Couldn\'t fit {batch_size} images with such aspect ratio into '
                                   f'memory. Try using smaller batch size or enabling '
                                   f'try_smaller_batch_on_fail option'
                        )
            else:
                raise HTTPException(
                    status_code=400, detail='Couldn\'t fit image with such aspect ratio into memory'
                )

    encoded_images = []
    data_string = f'data:{mimetypes.types_map[f".{request.output_format.lower()}"]};base64,' \
        .encode()
    for image in images:
        buffer = BytesIO()
        image.save(buffer, format=request.output_format)
        encoded_images.append(data_string + b64encode(buffer.getvalue()))

    return ImageArrayResponse(images=encoded_images)


# TODO task queue?
#  (or can set up an external scheduler and use this as internal endpoint)
@websocket_handler('/text_to_image', app)
async def text_to_image(websocket: WebSocket):
    request = TextToImageRequest(**(await websocket.receive_json()))
    width, height = size_from_aspect_ratio(request.aspect_ratio)
    response = await do_diffusion(
        request, pipeline.text_to_image, websocket, height=height, width=width
    )
    await websocket.send_json(
        {'status': WebSocketResponseStatus.FINISHED, 'result': json.loads(response.json())}
    )


@websocket_handler('/image_to_image', app)
async def image_to_image(websocket: WebSocket):
    request = ImageToImageRequest(**(await websocket.receive_json()))
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

    response = await do_diffusion(
        request,
        pipeline.image_to_image,
        websocket,
        init_image=preprocessed_source_image,
        strength=request.strength,
        alpha=preprocessed_alpha,
    )
    await websocket.send_json(
        {'status': WebSocketResponseStatus.FINISHED, 'result': json.loads(response.json())}
    )


@websocket_handler('/inpainting', app)
async def inpainting(websocket: WebSocket):

    request = InpaintingRequest(**(await websocket.receive_json()))
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
    response = await do_diffusion(
        request,
        pipeline.image_to_image,
        websocket,
        init_image=preprocessed_source_image,
        strength=request.strength,
        mask=preprocessed_mask,
        alpha=preprocessed_alpha,
    )
    await websocket.send_json(
        {'status': WebSocketResponseStatus.FINISHED, 'result': json.loads(response.json())}
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
