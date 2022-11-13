"""
Primary server tools.

Functions:
    authorize - authorize user's credentials
    authorize_web_socket - authorize user's credentials on the web socket
    websocket_handler - decorator to pass functions to web socket
    text_to_image - diffusion from text
    image_to_image - diffusion from text and image
    inpainting - diffusion on a masked image
    gobig - scale up an image and diffuse details
    upscale - scale up an image
    restore_face - fix faces
    make_tilable - diffusion on image edges to be tilable
    ping - measure latency
    setup - download models

Variables:
    LOGGER (logging.Logger) - logging
    SECURITY (fastapi.security.HTTPBasic) - security credentials
    APP (FastAPI) - server application
    PIPELINE (StablePipe) - diffusion model pipeline
"""

import asyncio
import functools
from json import JSONDecodeError
import json
import logging
from typing import Callable

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    WebSocket,
    status,
    WebSocketDisconnect,
)
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from PIL import Image, ImageChops, ImageDraw
from torch import autocast
import uvicorn

from batcher import do_diffusion
from consts import WebSocketResponseStatus, GFPGANModel
import esrgan_upscaler
from exceptions import BaseWebSocketException
import face_restoration
from gobig import do_gobig
from logging_settings import LOGGING
from pipeline import StablePipe
from request_models import (
    ImageToImageRequest,
    TextToImageRequest,
    GoBigRequest,
    ImageResponse,
    UpscaleRequest,
    InpaintingRequest,
    FaceRestorationRequest,
    MakeTilableRequest,
    MakeTilableResponse,
)
from settings import SETTINGS
from utils import (
    base64url_to_image,
    image_to_base64url,
    size_from_aspect_ratio,
    download_models,
)

LOGGER = logging.getLogger(__name__)
SECURITY = HTTPBasic()


async def authorize(credentials: HTTPBasicCredentials = Depends(SECURITY)):
    """
    Authorize user's credentials.

    Optional Arguments:
        credentials (HTTPBasicCredentials) - login credentials
            (default: Depends(security))

    Raises:
        HTTPException - if credentials don't match
    """
    if (
        credentials.username != SETTINGS.USERNAME
        or credentials.password != SETTINGS.PASSWORD
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


async def authorize_web_socket(websocket: WebSocket) -> bool:
    """
    Authorize user's credentials on the web socket.

    Arguments:
        websocket (WebSocket) - web socket to check

    Returns:
        authorized (bool) - `True` if authorized, else `False`
    """
    credentials = await websocket.receive_json()
    if (
        credentials.get("username") != SETTINGS.USERNAME
        or credentials.get("password") != SETTINGS.PASSWORD
    ):
        await websocket.close(
            status.WS_1008_POLICY_VIOLATION, "Authorization error"
        )
        return False

    return True


def websocket_handler(path: str, app: FastAPI) -> Callable:
    """
    Decorate functions to be handled by the web socket.

    Arguments:
        path (str) - web socket path corresponding to function call
        app (FastAPI) - web socket app

    Returns:
        decorator (Callable) - handled by web socket
    """

    def decorator(handler):
        @functools.wraps(handler)
        async def wrapper(websocket: WebSocket):
            await websocket.accept()
            try:
                if not await authorize_web_socket(websocket):
                    return

                await handler(websocket)
            except BaseWebSocketException as exc:
                await websocket.close(
                    status.WS_1008_POLICY_VIOLATION, exc.message
                )
            except JSONDecodeError:
                await websocket.close(
                    status.WS_1008_POLICY_VIOLATION,
                    "Server received message that is not in json format",
                )
            except WebSocketDisconnect:
                return

        return app.websocket(path)(wrapper)

    return decorator


APP = FastAPI(dependencies=[Depends(authorize)])
APP.add_middleware(GZipMiddleware, minimum_size=1000)

try:
    PIPELINE = StablePipe(
        cache_dir=SETTINGS.DIFFUSERS_CACHE_PATH,
        optimized=SETTINGS.USE_OPTIMIZED_MODE,
    )
    if SETTINGS.USE_OPTIMIZED_MODE:
        PIPELINE.enable_attention_slicing()
except Exception as exception:
    LOGGER.exception(exception)
    raise exception


# TODO task queue?
#  (or can set up an external scheduler and use this as internal endpoint)
@websocket_handler("/text_to_image", APP)
async def text_to_image(websocket: WebSocket):
    """Perform diffusion from a text prompt."""
    request = TextToImageRequest(**(await websocket.receive_json()))
    width, height = size_from_aspect_ratio(
        request.aspect_ratio, request.scaling_mode
    )
    response = await do_diffusion(
        request, PIPELINE.text_to_image, websocket, height=height, width=width
    )
    await websocket.send_json(
        {
            "status": WebSocketResponseStatus.FINISHED,
            "result": json.loads(response.json()),
        }
    )


@websocket_handler("/image_to_image", APP)
async def image_to_image(websocket: WebSocket):
    """Perform diffusion from a text prompt and starting image."""
    request = ImageToImageRequest(**(await websocket.receive_json()))
    source_image = base64url_to_image(request.source_image)
    aspect_ratio = source_image.width / source_image.height
    size = size_from_aspect_ratio(aspect_ratio, request.scaling_mode)
    response = await do_diffusion(
        request,
        PIPELINE.image_to_image,
        websocket,
        init_image=source_image.resize(size),
        strength=request.strength,
    )
    await websocket.send_json(
        {
            "status": WebSocketResponseStatus.FINISHED,
            "result": json.loads(response.json()),
        }
    )


@websocket_handler("/inpainting", APP)
async def inpainting(websocket: WebSocket):
    """Perform diffusion from a text prompt on masked areas of an image."""
    request = InpaintingRequest(**(await websocket.receive_json()))
    source_image = base64url_to_image(request.source_image)
    aspect_ratio = source_image.width / source_image.height
    size = size_from_aspect_ratio(aspect_ratio, request.scaling_mode)
    mask = None
    if request.mask:
        mask = base64url_to_image(request.mask).resize(size)
    # TODO return error if mask empty
    response = await do_diffusion(
        request,
        PIPELINE.image_to_image,
        websocket,
        init_image=source_image.resize(size).convert("RGB"),
        strength=request.strength,
        mask=mask,
    )
    await websocket.send_json(
        {
            "status": WebSocketResponseStatus.FINISHED,
            "result": json.loads(response.json()),
        }
    )


@websocket_handler("/gobig", APP)
async def gobig(websocket: WebSocket):
    """Scale up and perform piecewise diffusion on an image."""
    request = GoBigRequest(**(await websocket.receive_json()))
    upscaled = await do_gobig(
        request=request,
        websocket=websocket,
        input_image=base64url_to_image(request.image),
        pipeline=PIPELINE,
    )
    response = ImageResponse(image=image_to_base64url(upscaled))
    await websocket.send_json(
        {
            "status": WebSocketResponseStatus.FINISHED,
            "result": json.loads(response.json()),
        }
    )


@APP.post("/upscale")
async def upscale(request: UpscaleRequest) -> ImageResponse:
    """Scale up an image using RealESRGAN."""
    try:
        source_image = base64url_to_image(request.image)
        while (
            source_image.width < request.target_width
            or source_image.height < request.target_height
        ):
            source_image = esrgan_upscaler.upscale(
                image=source_image, model_type=request.model
            )

        if not request.maximize:
            source_image = source_image.resize(
                (request.target_width, request.target_height)
            )

        return ImageResponse(image=image_to_base64url(source_image))
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Scaling factor or image is too large",
        ) from exc


@APP.post("/restore_face")
async def restore_face(request: FaceRestorationRequest) -> ImageResponse:
    """Fix faces using GFPGAN."""
    if request.model_type == GFPGANModel.V1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GFPGAN v1 model is not supported",
        )
    return ImageResponse(
        image=face_restoration.restore_face(
            image=base64url_to_image(request.image),
            model_type=request.model_type,
            use_real_esrgan=request.use_real_esrgan,
            bg_tile=request.bg_tile,
            upscale=request.upscale,
            aligned=request.aligned,
            only_center_face=request.only_center_face,
        )
    )


@websocket_handler("/make_tilable", APP)
async def make_tilable(websocket: WebSocket):
    """Perform diffusion on image edges to make it tilable."""
    # pylint: disable=invalid-name
    request = MakeTilableRequest(**(await websocket.receive_json()))
    source_image = base64url_to_image(request.source_image)
    aspect_ratio = source_image.width / source_image.height
    size = size_from_aspect_ratio(aspect_ratio, request.scaling_mode)
    horizontal_offset_image = ImageChops.offset(
        source_image.resize(size), int(size[0] / 2), 0
    )
    request.num_variants = 1  # TODO: actually respect user's choice
    # Horizontal offset
    with autocast("cuda"):
        gradient_width = request.border_width * request.border_softness
        if int(gradient_width) != 0:
            gradient_step = 255 / gradient_width
        else:
            gradient_step = 255

        horizontal_mask = Image.new("L", size, color=0x00)
        start_gradient_x = size[0] / 2 - request.border_width
        horizontal_draw = ImageDraw.Draw(horizontal_mask)
        for i in range(int(gradient_width)):
            fill_color = min(int(i * gradient_step), 255)
            x = int(start_gradient_x + i)
            width = (request.border_width - i) * 2
            horizontal_draw.rectangle(
                ((x, 0), (x + width, size[1])), fill=fill_color
            )
        x = int(start_gradient_x + gradient_width)
        width = (request.border_width - gradient_width) * 2
        horizontal_draw.rectangle(((x, 0), (x + width, size[1])), fill=255)

    horizontal_offset_result = await do_diffusion(
        request,
        PIPELINE.image_to_image,
        websocket,
        return_images=True,
        progress_multiplier=1 / 3,
        init_image=horizontal_offset_image,
        mask=horizontal_mask,
        strength=request.strength,
    )

    # Vertical offset
    with autocast("cuda"):
        #  vertical_offset_images = []
        #  for image in horizontal_offset_result:
        #      vertical_offset_image = ImageChops.offset(
        #          image, 0, int(size[1] / 2)
        #      )
        #      vertical_offset_images.append(
        #          vertical_offset_image
        #      )
        vertical_offset_image = ImageChops.offset(
            horizontal_offset_result[0], 0, int(size[1] / 2)
        )

        vertical_mask = Image.new("L", size, color=0x00)
        start_gradient_y = size[1] / 2 - request.border_width
        vertical_draw = ImageDraw.Draw(vertical_mask)
        for i in range(int(gradient_width)):
            fill_color = min(int(i * gradient_step), 255)
            y = int(start_gradient_y + i)
            height = (request.border_width - i) * 2
            vertical_draw.rectangle(
                ((0, y), (size[0], y + height)), fill=fill_color
            )

        y = int(start_gradient_y + gradient_width)
        height = (request.border_width - gradient_width) * 2
        vertical_draw.rectangle(((0, y), (size[0], y + height)), fill=255)

    vertical_offset_result = await do_diffusion(
        request,
        PIPELINE.image_to_image,
        websocket,
        return_images=True,
        progress_multiplier=1 / 3,
        progress_offset=1 / 3,
        #  batched_params={"init_image": vertical_offset_images},
        init_image=vertical_offset_image,
        strength=request.strength,
        mask=vertical_mask,
    )

    # Center
    with autocast("cuda"):
        #  center_offset_images = []
        #  for image in vertical_offset_result:
        #      center_offset_image = ImageChops.offset(
        #          image, -int(size[0] / 2), 0
        #      )
        #      center_offset_images.append(center_offset_image)
        center_offset_image = ImageChops.offset(
            vertical_offset_result[0], -int(size[0] / 2), 0
        )

        center_mask = Image.new("L", size, color=0x00)
        center_draw = ImageDraw.Draw(center_mask)
        for i in range(int(gradient_width)):
            fill_color = min(int(i * gradient_step), 255)
            y = int(start_gradient_y + i)
            x = int(start_gradient_x + i)
            offset = (request.border_width - i) * 2
            center_draw.rectangle(
                ((x, y), (x + offset, y + offset)), fill=fill_color
            )

        y = int(start_gradient_y + gradient_width)
        x = int(start_gradient_x + gradient_width)
        offset = (request.border_width - gradient_width) * 2
        center_draw.rectangle(((x, y), (x + offset, y + offset)), fill=255)

    images = await do_diffusion(
        request,
        PIPELINE.image_to_image,
        websocket,
        return_images=True,
        progress_multiplier=1 / 3,
        progress_offset=2 / 3,
        #  batched_params={"init_image": preprocessed_center_offset_images},
        init_image=center_offset_image,
        strength=request.strength,
        mask=center_mask,
    )

    response = MakeTilableResponse(
        images=[
            ImageChops.offset(image, 0, -int(size[1] / 2)) for image in images
        ],
        mask=ImageChops.lighter(
            ImageChops.offset(horizontal_mask, -int(size[0] / 2), 0),
            ImageChops.offset(vertical_mask, 0, -int(size[1] / 2)),
        ),
    )
    await websocket.send_json(
        {
            "status": WebSocketResponseStatus.FINISHED,
            "result": json.loads(response.json()),
        }
    )


@APP.get("/ping")
async def ping():
    """Do nothing, to measure ping speed."""
    return


async def setup():
    """Download GFPGAN and RealESRGAN models."""
    await download_models(
        face_restoration.GFPGAN_URLS + esrgan_upscaler.ESRGAN_URLS
    )


if __name__ == "__main__":
    asyncio.run(setup())
    uvicorn.run(
        APP, host=SETTINGS.HOST, port=SETTINGS.PORT, log_config=LOGGING
    )
