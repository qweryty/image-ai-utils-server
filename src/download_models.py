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
import os
from typing import List

import aiofiles
import httpx
from huggingface_hub import snapshot_download
from tqdm import tqdm

from consts import (
    STABLE_DIFFUSION_MODEL_NAME,
    STABLE_DIFFUSION_INPAINTING_NAME,
    STABLE_DIFFUSION_16B_REVISION,
)
from esrgan_upscaler import ESRGAN_URLS
from face_restoration import GFPGAN_URLS
from settings import SETTINGS
from utils import resolve_path


LOGGER = logging.getLogger(__name__)


async def download_models(models: List[str]):
    """
    Asynchronously download specified models.

    Arguments:
        models (List[str]) - urls pointing to model weights
    """
    models_dir = resolve_path("models")
    os.makedirs(models_dir, exist_ok=True)

    async def download_one(url: str, position: int):
        file_name = url.split("/")[-1]
        file_path = os.path.join(models_dir, file_name)
        if os.path.exists(file_path):
            return

        LOGGER.info(f"Downloading {file_path} {position}")
        async with aiofiles.open(file_path, mode="wb") as outfile:
            async with httpx.AsyncClient() as client:
                async with client.stream("GET", url, follow_redirects=True) as response:
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
                                response.num_bytes_downloaded - num_bytes_downloaded
                            )
                            num_bytes_downloaded = response.num_bytes_downloaded

    await asyncio.gather(*[download_one(model, i) for i, model in enumerate(models)])


async def download_stable_diffusion(model_name: str, revision: str):
    """Download specified huggingface model and revision."""
    snapshot_download(
        model_name,
        revision=revision,
        cache_dir=SETTINGS.DIFFUSERS_CACHE_PATH,
        use_auth_token=True,
    )


async def download():
    """Download all models, asynchronously."""
    revision = STABLE_DIFFUSION_16B_REVISION if SETTINGS.USE_OPTIMIZED_MODE else None
    # TODO check if model exists and hash
    await asyncio.gather(
        download_models(GFPGAN_URLS + ESRGAN_URLS),
        download_stable_diffusion(STABLE_DIFFUSION_MODEL_NAME, revision),
        download_stable_diffusion(STABLE_DIFFUSION_INPAINTING_NAME, revision),
    )


if __name__ == "__main__":
    asyncio.run(download())
