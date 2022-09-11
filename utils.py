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

from consts import ScalingMode

logger = logging.getLogger(__name__)


def image_to_base64url(image: Image.Image, output_format: str = 'PNG') -> bytes:
    data_string = f'data:{mimetypes.types_map[f".{output_format.lower()}"]};base64,'.encode()
    buffer = BytesIO()
    image.save(buffer, format=output_format)
    return data_string + b64encode(buffer.getvalue())


def base64url_to_image(source: bytes) -> Image.Image:
    _, data = source.split(b',')
    return Image.open(BytesIO(b64decode(data)))


def size_from_aspect_ratio(aspect_ratio: float, scaling_mode: ScalingMode) -> Tuple[int, int]:
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
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


async def download_models(models: List[str]):
    async def download(url: str, position: int):
        file_name = url.split('/')[-1]
        file_path = os.path.join(resolve_path('models'), file_name)
        if os.path.exists(file_path):
            return

        logger.info(f'Downloading {file_path} {position}')
        async with aiofiles.open(file_path, mode='wb') as f:
            async with httpx.AsyncClient() as client:
                async with client.stream('GET', url, follow_redirects=True) as response:
                    response.raise_for_status()
                    total = int(response.headers['Content-Length'])
                    with tqdm(
                            desc=f'Downloading {file_name}',
                            total=total,
                            unit_scale=True,
                            unit_divisor=1024,
                            unit='B',
                            position=position
                    ) as progress:
                        num_bytes_downloaded = response.num_bytes_downloaded
                        async for chunk in response.aiter_bytes():
                            await f.write(chunk)
                            progress.update(response.num_bytes_downloaded - num_bytes_downloaded)
                            num_bytes_downloaded = response.num_bytes_downloaded

    await asyncio.gather(*[download(model, i) for i, model in enumerate(models)])
