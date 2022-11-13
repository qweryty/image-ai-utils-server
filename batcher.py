"""
Tools for performing diffusion with on-the-fly batch sizing.

Classes:
    Batcher - determine batch size on the fly

Functions:
    do_diffusion - wrap `Batcher`

Variables:
    LOGGER (logging.Logger) - logging
"""


import asyncio
import logging
from math import ceil
from typing import Callable, List, Union

from fastapi import WebSocket
from PIL import Image
import torch
from torch import autocast

from consts import WebSocketResponseStatus
from exceptions import (
    BatchSizeIsTooLargeException,
    AspectRatioTooWideException,
)
from request_models import BaseDiffusionRequest, ImageArrayResponse


LOGGER = logging.getLogger(__name__)


class Batcher:
    """
    Perform diffusion with on-the-fly batch sizing.

    Instance Attributes:
        request (BaseDiffusionRequest) - diffusion request from web socket
        num_batches (int) - starts by optimizing `request.batch_size` and
            adjusts during `do_diffusion`
        websocket (WebSocket) - web socket
        progress_multiplier (float) - scale progress by this factor,
            set when calling `do_diffusion` (default: 1.0)
        progress_offset (float) - offset progress by this factor,
            set when calling `do_diffusion` (default: 0.0)

    Read-Only Instance Attributes:
        batch_size (int) - number of images per batch
        last_batch_size (int) - number of images in the last batch

    Methods:
        do_diffusion - perform diffusion with on-the-fly batch size
        call_pipe - perform diffusion for each batch with a set batch size
        diffusion_method - perform diffusion on a single batch
        progress_callback - send progress back to web socket
    """

    def __init__(
        self,
        request: BaseDiffusionRequest,
        diffusion_method: Callable,
        websocket: WebSocket,
    ):
        """
        Initialize Batcher.

        Arguments:
            request (BaseDiffusionRequest) - diffusion request from web socket
            diffusion_method (Callable) - method used to perform the diffusion
            websocket (WebSocket) - web socket
        """
        self.request = request
        self.diffusion_method = diffusion_method
        self.websocket = websocket

        self.num_batches = 1
        self._count = 0
        self.progress_multiplier = 1.0
        self.progress_offset = 0.0

    @property
    def batch_size(self):
        """Get batch size."""
        return ceil(self.request.num_variants / self.num_batches)

    @property
    def last_batch_size(self):
        """Get batch size of the last batch."""
        return self.request.num_variants % self.batch_size or self.batch_size

    @property
    def _next_batch_size_to_try(self):
        """If this batch fails, return the next batch size."""
        return ceil(self.request.num_variants / (self.num_batches + 1))

    async def progress_callback(self, batch_step: int, total_batch_steps: int):
        """
        Send progress back to websocket.

        Arguments:
            batch_step (int) - timestep of the current batch
            total_batch_steps (int) - number of steps in the current batch
        """
        current_step = (self._count * total_batch_steps) + batch_step
        total_steps = self.num_batches * total_batch_steps
        progress = (
            self.progress_multiplier * current_step / total_steps
            + self.progress_offset
        )
        if self.websocket is not None:
            await self.websocket.send_json(
                {
                    "status": WebSocketResponseStatus.PROGRESS,
                    "progress": progress,
                }
            )
        # https://github.com/aaugustin/websockets/issues/867
        # Probably will be fixed if/when diffusers will
        # implement asynchronous pipeline.
        # https://github.com/huggingface/diffusers/issues/374
        await asyncio.sleep(0)

    async def call_pipe(self, **kwargs):
        """
        Call the Diffusion pipeline.

        All kwargs are passed to `self.diffusion_method`.

        Returns:
            images (List[Image.Image]) - images from diffusion
        """
        with autocast("cuda"):
            with torch.inference_mode():
                images = []
                for count in range(self.num_batches):
                    self._count = count
                    prompts = [self.request.prompt] * (
                        self.batch_size
                        if count + 1 < self.num_batches
                        else self.last_batch_size
                    )
                    new_images = await self.diffusion_method(
                        prompt=prompts,
                        num_inference_steps=self.request.num_inference_steps,
                        guidance_scale=self.request.guidance_scale,
                        progress_callback=self.progress_callback,
                        **kwargs,
                    )
                    images.extend(new_images)
                return images

    async def do_diffusion(
        self,
        return_images: bool = False,
        **kwargs,
    ) -> Union[ImageArrayResponse, List[Image.Image]]:
        """
        Perform diffusion with batch size calculated on the fly.

        Optional Arguments:
            return_images (bool) - return list rather than ImageArrayResponse
                (default: `False`)
            progress_multiplier (float) - scale progress by this factor
                (default: 1.0)
            progress_offset (float) - offset progress by this amount
                (default: 0.0)

        Additional kwargs are passed to `diffusion_method`.

        Returns:
            images (List[Image.Image] if `return_images`
                else ImageArrayResponse)

        Raises:
            BatchSizeIsTooLargeException - if batch size > 1 and a CUDA memory
                error occurs
            AspectRatioTooWideException - if batch size == 1 and a CUDA memory
                error occurs
        """
        if self.request.seed is not None:
            generator = torch.Generator("cuda").manual_seed(self.request.seed)
        else:
            generator = None

        self.progress_multiplier = kwargs.pop("progress_multiplier", 1.0)
        self.progress_offset = kwargs.pop("progress_offset", 0.0)
        self._count = 0
        self.num_batches = ceil(self.request.num_variants / self.request.batch_size)

        while self.num_batches <= self.request.num_variants:
            try:
                images = await self.call_pipe(generator=generator, **kwargs)
                break
            except RuntimeError as exc:
                if self.request.try_smaller_batch_on_fail:
                    batch_size = self.batch_size
                    while self.batch_size == self._next_batch_size_to_try:
                        self.num_batches += 1
                    self.num_batches += 1
                    LOGGER.warning(
                        f"Batch size {batch_size} was too large, "
                        f"trying {self.batch_size}."
                    )
                else:
                    raise BatchSizeIsTooLargeException(
                        self.batch_size
                    ) from exc
        else:
            raise AspectRatioTooWideException

        if return_images:
            return images
        return ImageArrayResponse(images=images)


async def do_diffusion(
    request: BaseDiffusionRequest,
    diffusion_method: Callable,
    websocket: WebSocket,
    **kwargs,
) -> Union[ImageArrayResponse, List[Image.Image]]:
    """
    Wrap `Batcher`.

    See `Batcher.__init__` and `Batcher.do_diffusion` for usage.
    """
    return await Batcher(
        request=request, diffusion_method=diffusion_method, websocket=websocket
    ).do_diffusion(**kwargs)
