from settings import settings  # noqa
from logging_settings import LOGGING  # noqa

import asyncio
import functools
from json import JSONDecodeError

from exceptions import BatchSizeIsTooLargeException, AspectRatioTooWideException, \
    BaseWebSocketException
import face_restoration
from gobig import do_gobig

import json
import logging
from typing import Callable, List, Dict, Any, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, WebSocket, status, WebSocketDisconnect
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from PIL import ImageChops, Image, ImageDraw
from torch import autocast

import esrgan_upscaler
from request_models import BaseImageGenerationRequest, ImageArrayResponse, ImageToImageRequest, \
    TextToImageRequest, GoBigRequest, ImageResponse, UpscaleRequest, InpaintingRequest, \
    FaceRestorationRequest, MakeTilableRequest, MakeTilableResponse
from universal_pipeline import StableDiffusionUniversalPipeline, preprocess, preprocess_mask
from utils import base64url_to_image, image_to_base64url, size_from_aspect_ratio, download_models

print('Imported')
logger = logging.getLogger(__name__)
security = HTTPBasic()


async def authorize(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != settings.USERNAME or credentials.password != settings.PASSWORD:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


async def authorize_web_socket(websocket: WebSocket) -> bool:
    credentials = await websocket.receive_json()
    if credentials.get('username') != settings.USERNAME or \
            credentials.get('password') != settings.PASSWORD:
        await websocket.close(status.WS_1008_POLICY_VIOLATION, 'Authorization error')
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
            except BaseWebSocketException as e:
                await websocket.close(status.WS_1008_POLICY_VIOLATION, e.message)
            except JSONDecodeError:
                await websocket.close(
                    status.WS_1008_POLICY_VIOLATION,
                    'Server received message that is not in json format'
                )
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
    if settings.USE_OPTIMIZED_MODE:
        pipeline.enable_attention_slicing()
except Exception as e:
    print('Caught exception')
    print(e)
    logger.exception(e)
    raise e


async def do_diffusion(
        request: BaseImageGenerationRequest,
        diffusion_method: Callable,
        websocket: WebSocket,
        batched_params: Optional[Dict[str, List[Any]]] = None,
        return_images: bool = False,
        progress_multiplier: float = 1.,
        progress_offset: float = 0.,
        **kwargs
) -> Union[ImageArrayResponse, List[Image.Image]]:
    if request.seed is not None:
        generator = torch.Generator('cuda').manual_seed(request.seed)
    else:
        generator = None

    if batched_params is None:
        batched_params = {}
    with autocast('cuda'):
        with torch.inference_mode():
            for batch_size in range(min(request.batch_size, request.num_variants), 0, -1):
                try:
                    num_batches = request.num_variants // batch_size
                    prompts = [request.prompt] * batch_size
                    last_batch_size = request.num_variants - batch_size * num_batches
                    images = []

                    for i in range(num_batches):
                        batch_kwargs = {}
                        for key, value in batched_params.items():
                            batch_kwargs[key] = value[i * batch_size: (i + 1) * batch_size]

                        async def progress_callback(batch_step: int, total_batch_steps: int):
                            current_step = i * total_batch_steps + batch_step
                            total_steps = num_batches * total_batch_steps
                            if last_batch_size:
                                total_steps += total_batch_steps
                            progress = progress_multiplier * current_step / total_steps + \
                                       progress_offset
                            await websocket.send_json(
                                {'status': WebSocketResponseStatus.PROGRESS, 'progress': progress}
                            )
                            # https://github.com/aaugustin/websockets/issues/867
                            # Probably will be fixed if/when diffusers will implement asynchronous
                            # pipeline
                            # https://github.com/huggingface/diffusers/issues/374
                            await asyncio.sleep(0)

                        new_images = (await diffusion_method(
                            prompt=prompts,
                            num_inference_steps=request.num_inference_steps,
                            generator=generator,
                            guidance_scale=request.guidance_scale,
                            progress_callback=progress_callback,
                            **batch_kwargs,
                            **kwargs
                        ))
                        images.extend(new_images)

                    if last_batch_size:
                        batch_kwargs = {}
                        for key, value in batched_params.items():
                            batch_kwargs[key] = value[-last_batch_size:]

                        async def progress_callback(batch_step: int, total_batch_steps: int):
                            current_step = num_batches * total_batch_steps + batch_step
                            total_steps = (num_batches + 1) * total_batch_steps
                            progress = progress_multiplier * current_step / total_steps + \
                                       progress_offset
                            await websocket.send_json(
                                {'status': WebSocketResponseStatus.PROGRESS, 'progress': progress}
                            )

                        new_images = (await diffusion_method(
                            prompt=[request.prompt] * last_batch_size,
                            num_inference_steps=request.num_inference_steps,
                            generator=generator,
                            guidance_scale=request.guidance_scale,
                            progress_callback=progress_callback,
                            **batch_kwargs,
                            **kwargs
                        ))
                        images.extend(new_images)
                    break
                except RuntimeError as e:
                    if request.try_smaller_batch_on_fail:
                        logger.warning(f'Batch size {batch_size} was too large, trying smaller')
                    else:
                        raise BatchSizeIsTooLargeException(batch_size)
            else:
                raise AspectRatioTooWideException

    if return_images:
        return images
    else:
        return ImageArrayResponse(images=images)


# TODO task queue?
#  (or can set up an external scheduler and use this as internal endpoint)
@websocket_handler('/text_to_image', app)
async def text_to_image(websocket: WebSocket):
    request = TextToImageRequest(**(await websocket.receive_json()))
    width, height = size_from_aspect_ratio(request.aspect_ratio, request.scaling_mode)
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
    size = size_from_aspect_ratio(aspect_ratio, request.scaling_mode)

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
    size = size_from_aspect_ratio(aspect_ratio, request.scaling_mode)

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


@websocket_handler('/gobig', app)
async def gobig(websocket: WebSocket):
    request = GoBigRequest(**(await websocket.receive_json()))

    if request.seed is not None:
        generator = torch.Generator('cuda').manual_seed(request.seed)
    else:
        generator = None

    async def progress_callback(progress: float):
        await websocket.send_json(
            {'status': WebSocketResponseStatus.PROGRESS, 'progress': progress}
        )

    upscaled = await do_gobig(
        input_image=base64url_to_image(request.image),
        prompt=request.prompt,
        maximize=request.maximize,
        target_width=request.target_width,
        target_height=request.target_height,
        overlap=request.overlap,
        use_real_esrgan=request.use_real_esrgan,
        esrgan_model=request.esrgan_model,
        pipeline=pipeline,
        strength=request.strength,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        generator=generator,
        progress_callback=progress_callback
    )
    response = ImageResponse(image=image_to_base64url(upscaled))
    await websocket.send_json(
        {'status': WebSocketResponseStatus.FINISHED, 'result': json.loads(response.json())}
    )


@app.post('/upscale')
async def upscale(request: UpscaleRequest) -> ImageResponse:
    try:
        source_image = base64url_to_image(request.image)
        while source_image.width < request.target_width or \
                source_image.height < request.target_height:
            source_image = esrgan_upscaler.upscale(image=source_image, model_type=request.model)

        if not request.maximize:
            source_image = source_image.resize((request.target_width, request.target_height))

        return ImageResponse(image=image_to_base64url(source_image))
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail='Scaling factor or image is too large'
        )


@app.post('/restore_face')
async def restore_face(request: FaceRestorationRequest) -> ImageResponse:
    if request.model_type == GFPGANModel.V1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='GFPGAN v1 model is not supported'
        )
    return ImageResponse(
        image=face_restoration.restore_face(
            image=base64url_to_image(request.image),
            model_type=request.model_type,
            use_real_esrgan=request.use_real_esrgan,
            bg_tile=request.bg_tile,
            upscale=request.upscale,
            aligned=request.aligned,
            only_center_face=request.only_center_face
        )
    )


@websocket_handler('/make_tilable', app)
async def make_tilable(websocket: WebSocket):
    request = MakeTilableRequest(**(await websocket.receive_json()))
    source_image = base64url_to_image(request.source_image)
    aspect_ratio = source_image.width / source_image.height
    size = size_from_aspect_ratio(aspect_ratio, request.scaling_mode)
    horizontal_offset_image = ImageChops.offset(source_image.resize(size), int(size[0] / 2), 0)

    # Horizontal offset
    with autocast('cuda'):
        preprocessed_horizontal_offset_image = preprocess(
            horizontal_offset_image
        ).to(pipeline.device)
        preprocessed_alpha = None
        if horizontal_offset_image.mode == 'RGBA':
            preprocessed_alpha = 1 - preprocess_mask(
                horizontal_offset_image.getchannel('A')
            ).to(pipeline.device)

        if preprocessed_alpha is not None and not preprocessed_alpha.any():
            preprocessed_alpha = None

        gradient_width = request.border_width * request.border_softness
        if int(gradient_width) != 0:
            gradient_step = 255 / gradient_width
        else:
            gradient_step = 255

        horizontal_mask = Image.new('L', size, color=0x00)
        start_gradient_x = size[0] / 2 - request.border_width
        horizontal_draw = ImageDraw.Draw(horizontal_mask)
        for i in range(int(gradient_width)):
            fill_color = min(int(i * gradient_step), 255)
            x = int(start_gradient_x + i)
            width = (request.border_width - i) * 2
            horizontal_draw.rectangle(((x, 0), (x + width, size[1])), fill=fill_color)
        x = int(start_gradient_x + gradient_width)
        width = (request.border_width - gradient_width) * 2
        horizontal_draw.rectangle(((x, 0), (x + width, size[1])), fill=255)
        horizontal_preprocessed_mask = preprocess_mask(horizontal_mask).to(pipeline.device)

    horizontal_offset_result = await do_diffusion(
        request,
        pipeline.image_to_image,
        websocket,
        return_images=True,
        progress_multiplier=1/3,
        init_image=preprocessed_horizontal_offset_image,
        mask=horizontal_preprocessed_mask,
        strength=request.strength,
        alpha=preprocessed_alpha,
    )
    '''horizontal_offset_result = [
        Image.composite(image, horizontal_offset_image, horizontal_mask)
        for image in horizontal_offset_result
    ]'''

    # Vertical offset
    with autocast('cuda'):
        preprocessed_vertical_offset_images = []
        for image in horizontal_offset_result:
            vertical_offset_image = ImageChops.offset(image, 0, int(size[1] / 2))
            preprocessed_vertical_offset_images.append(
                preprocess(vertical_offset_image).to(pipeline.device)
            )

        vertical_mask = Image.new('L', size, color=0x00)
        start_gradient_y = size[1] / 2 - request.border_width
        vertical_draw = ImageDraw.Draw(vertical_mask)
        for i in range(int(gradient_width)):
            fill_color = min(int(i * gradient_step), 255)
            y = int(start_gradient_y + i)
            height = (request.border_width - i) * 2
            vertical_draw.rectangle(((0, y), (size[0], y + height)), fill=fill_color)

        y = int(start_gradient_y + gradient_width)
        height = (request.border_width - gradient_width) * 2
        vertical_draw.rectangle(((0, y), (size[0], y + height)), fill=255)
        vertical_preprocessed_mask = preprocess_mask(vertical_mask).to(pipeline.device)

    vertical_offset_result = await do_diffusion(
        request,
        pipeline.image_to_image,
        websocket,
        return_images=True,
        progress_multiplier=1/3,
        progress_offset=1/3,
        batched_params={'init_image': preprocessed_vertical_offset_images},
        strength=request.strength,
        mask=vertical_preprocessed_mask,
    )

    # Center
    with autocast('cuda'):
        preprocessed_center_offset_images = []
        for image in vertical_offset_result:
            center_offset_image = ImageChops.offset(image, -int(size[0] / 2), 0)
            preprocessed_center_offset_images.append(
                preprocess(center_offset_image).to(pipeline.device)
            )

        center_mask = Image.new('L', size, color=0x00)
        center_draw = ImageDraw.Draw(center_mask)
        for i in range(int(gradient_width)):
            fill_color = min(int(i * gradient_step), 255)
            y = int(start_gradient_y + i)
            x = int(start_gradient_x + i)
            offset = (request.border_width - i) * 2
            center_draw.rectangle(((x, y), (x + offset, y + offset)), fill=fill_color)

        y = int(start_gradient_y + gradient_width)
        x = int(start_gradient_x + gradient_width)
        offset = (request.border_width - gradient_width) * 2
        center_draw.rectangle(((x, y), (x + offset, y + offset)), fill=255)
        center_preprocessed_mask = preprocess_mask(center_mask).to(pipeline.device)

    images = await do_diffusion(
        request,
        pipeline.image_to_image,
        websocket,
        return_images=True,
        progress_multiplier=1 / 3,
        progress_offset=2 / 3,
        batched_params={'init_image': preprocessed_center_offset_images},
        strength=request.strength,
        mask=center_preprocessed_mask,
    )

    response = MakeTilableResponse(
        images=[
            ImageChops.offset(image, 0, -int(size[1] / 2)) for image in images
        ],
        mask=ImageChops.lighter(
            ImageChops.offset(horizontal_mask, -int(size[0] / 2), 0),
            ImageChops.offset(vertical_mask, 0, -int(size[1] / 2)),
        )
    )
    await websocket.send_json(
        {'status': WebSocketResponseStatus.FINISHED, 'result': json.loads(response.json())}
    )


@app.get('/ping')
async def ping():
    return


async def setup():
    await download_models(face_restoration.GFPGAN_URLS + esrgan_upscaler.ESRGAN_URLS)


if __name__ == '__main__':
    asyncio.run(setup())
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, log_config=LOGGING)
