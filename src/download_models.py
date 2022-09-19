import asyncio
import logging
import os
from typing import List

import aiofiles
import httpx
from huggingface_hub import snapshot_download
from tqdm import tqdm

from consts import STABLE_DIFFUSION_MODEL_NAME, STABLE_DIFFUSION_REVISION
from esrgan_upscaler import ESRGAN_URLS
from face_restoration import GFPGAN_URLS
from settings import settings
from utils import resolve_path

logger = logging.getLogger(__name__)


async def download_models(models: List[str]):
    models_dir = resolve_path('models')
    os.makedirs(models_dir, exist_ok=True)

    async def download(url: str, position: int):
        file_name = url.split('/')[-1]
        file_path = os.path.join(models_dir, file_name)
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


async def download_stable_diffusion(model_name: str, revision: str):
    snapshot_download(
        model_name, revision=revision, cache_dir=settings.DIFFUSERS_CACHE_PATH, use_auth_token=True
    )


async def download():
    await asyncio.gather(
        download_models(GFPGAN_URLS + ESRGAN_URLS),
        download_stable_diffusion(STABLE_DIFFUSION_MODEL_NAME, STABLE_DIFFUSION_REVISION)
    )


if __name__ == '__main__':
    asyncio.run(download())
