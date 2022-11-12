"""
Utility functions for server.

Functions:
    image_to_base64url - convert image to bytes
    base64url_to_image - convert bytes back to image
    size_from_aspect_ratio - pixel proportions from ratio
    resolve_path - absolute path
    download_models - download model weights
"""


import asyncio
import logging
import mimetypes
import os
from base64 import b64decode, b64encode
from io import BytesIO
from typing import Tuple, List

import aiofiles
import httpx
from PIL import Image
# TODO use common utils
from tqdm import tqdm

from consts import ImageFormat, ScalingMode

LOGGER = logging.getLogger(__name__)


def image_to_base64url(
    image: Image.Image, output_format: ImageFormat = ImageFormat.PNG
) -> bytes:
    """
    Convert image to bytes string.

    Arguments:
        image (Image.Image) - image to convert

    Optional Arguments:
        output_format (ImageFormat) - filetype for output
            (default: ImageFormat.PNG)

    Returns:
        bytes (bytes) - encoded image
    """
    # pylint: disable=line-too-long
    data_string = f'data:{mimetypes.types_map[f".{output_format.lower()}"]};base64,'.encode()
    buffer = BytesIO()
    image.save(buffer, format=output_format)
    return data_string + b64encode(buffer.getvalue())


def base64url_to_image(source: bytes) -> Image.Image:
    """
    Convert a bytes string back to an image.

    Arguments:
        source (bytes) - as encoded with `image_to_base64url`

    Returns:
        image (Image.Image) - decoded image
    """
    _, data = source.split(b",")
    return Image.open(BytesIO(b64decode(data)))


def size_from_aspect_ratio(
    aspect_ratio: float, scaling_mode: ScalingMode
) -> Tuple[int, int]:
    """
    Return a standard size from an aspect ratio.

    Arguments:
        aspect_ratio (float) - ratio of width:height
        scaling_mode (ScalingMode) - if `ScalingMode.GROW` then smallest side
            will be 512, else largest side will be 512

    Returns:
        width (int) - pixels wide
        height (int) - pixels tall
    """
    if (scaling_mode == ScalingMode.GROW) == (aspect_ratio > 1):
        height = 512
        width = int(height * aspect_ratio)
        width -= width % 64
    else:
        width = 512
        height = int(width / aspect_ratio)
        height -= height % 64
    return width, height


def resolve_path(path: str) -> str:
    """Get absolute path from relative path."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


async def download_models(models: List[str]):
    """
    Asynchronously download models.

    Arguments:
        models (List[str]) - urls pointing to model weights
    """
    models_dir = resolve_path("models")
    os.makedirs(models_dir, exist_ok=True)

    async def download(url: str, position: int):
        file_name = url.split("/")[-1]
        file_path = os.path.join(models_dir, file_name)
        if os.path.exists(file_path):
            return

        LOGGER.info(f"Downloading {file_path} {position}")
        async with aiofiles.open(file_path, mode="wb") as outfile:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "GET", url, follow_redirects=True
                ) as response:
                    response.raise_for_status()
                    total = int(response.headers["Content-Length"])
                    with tqdm(
                        desc=f"Downloading {file_name}",
                        total=total,
                        unit_scale=True,
                        unit_divisor=1024,
                        unit="B",
                        position=position,
                    ) as progress:
                        num_bytes_downloaded = response.num_bytes_downloaded
                        async for chunk in response.aiter_bytes():
                            await outfile.write(chunk)
                            progress.update(
                                response.num_bytes_downloaded
                                - num_bytes_downloaded
                            )
                            num_bytes_downloaded = (
                                response.num_bytes_downloaded
                            )

    await asyncio.gather(
        *[download(model, i) for i, model in enumerate(models)]
    )
