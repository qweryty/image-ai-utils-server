"""
Provide a wrapper around diffusers' stable diffusion pipelines.

Classes:
    StablePipe - wrapper for both img2img and inpaint pipelines
"""


import gc
import inspect
import re
from typing import Awaitable, Callable, List, Optional, Union

from PIL import Image
import torch
from torchvision import transforms
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
)


# ----------------------------- This gets messy. -----------------------------
# Why this messy solution was chosen:
#
# Using `exec` is decidedly un-pythonic, and prefaces the use of other messy
# tools like wildcard imports, modifying global variables, and monkey patching.
#
# The choice was made to pursue this, however, with the following
# considerations in mind:
#   - Having a progress bar in-app is really useful
#   - Waiting to do more inference until the progress bar connects is silly;
#     this should be done asynchronously or not at all
#   - While sub-classing the `diffusers` pipelines would be the most pythonic
#     solution, the `__call__` methods are very long, and change a lot with
#     each version release. Sub-classing would involve copy-pasting large
#     swaths of code, which would need to be updated by hand every time
#     `diffusers` updates. This is not forward-looking, nor very pythonic.
#   - Using `re` to _very carefully_ patch in `async` and `await` seems like a
#     reasonable compromise.
#
# If anyone reading this knows of a way to accomplish the same thing, but in a
# less-horrifying manner, please submit a pull request!
# ----------------------------------------------------------------------------

# pylint: disable=wildcard-import, line-too-long
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import *
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import *


def _deindent(source_string: str) -> str:
    """Remove leading indent from source code."""
    lines = source_string.split("\n")
    indent = len(lines[0]) - len(lines[0].lstrip())
    return "\n".join(line[indent:] for line in lines)


def _async_patch(pipeline_class, timesteps_name: str = "timesteps"):
    """
    Patch, in place, pipeline's `__call__` method.

    Makes `__call__` asynchronous.
    `await`s `callback` function.
    Changes arguments passed to `callback` function.
    """
    # pylint: disable=undefined-variable, global-variable-undefined, exec-used
    global __call__
    source = _deindent(inspect.getsource(pipeline_class.__call__))
    source = re.sub(
        "def __call__",
        "async def __call__",
        source,
    )
    source = re.sub(
        r"callback\(i, t, latents\)",
        f"await callback(i + 1, len({timesteps_name}))",
        source,
    )
    exec(source, globals())
    pipeline_class.__call__ = __call__
    del __call__


_async_patch(StableDiffusionImg2ImgPipeline)
_async_patch(StableDiffusionInpaintPipeline, "timesteps_tensor")
# -------------------------------- End mess. --------------------------------


class StablePipe:
    """
    Wrapper around diffusers' stable diffusion pipelines.

    Easily swap `StableDiffusionImg2ImgPipeline` and
    `StableDiffusionInpaintPipeline` between CPU and GPU. Intelligently choose
    which model is needed for which task.

    Instance Attributes:
        device (`torch.device`, read-only) - which device the active model uses
        mode (str) - which model is active; one of "img2img" or "inpaint"

    Methods:
        enable_attention_slicing - better memory performance
        disable_attention_slicing - better speed performance
        text_to_image - text prompt to image
        image_to_image - image and text prompt to image
    """

    def __init__(self, **kwargs):
        """
        Initialize pipeline.

        All kwargs are passed to `StableDiffusionImg2ImgPipeline` and
        `StableDiffusionInpaintPipeline`.
        """
        self._init_models(**kwargs)

    def _init_models(
        self,
        version="5",
        inpaint_version="runwayml/stable-diffusion-inpainting",
        **kwargs,
    ):
        try:
            model_name = (
                f"CompVis/stable-diffusion-v1-{version}"
                if 0 < int(version) <= 3
                else f"runwayml/stable-diffusion-v1-{version}"
            )
        except ValueError:
            model_name = version
        self._mode = "img2img"
        self._pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_name,
            use_auth_token=True,
            safety_checker=None,
            **kwargs,
        ).to("cuda")
        self._pipe.set_progress_bar_config(dynamic_ncols=True)
        self._inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            inpaint_version,
            use_auth_token=True,
            safety_checker=None,
            **kwargs,
        )
        self._inpaint_pipe.set_progress_bar_config(dynamic_ncols=True)

    def enable_attention_slicing(self):
        """Enable attention slicing for better memory management."""
        self._pipe.enable_attention_slicing()
        self._inpaint_pipe.enable_attention_slicing()

    def disable_attention_slicing(self):
        """Disable attention slicing for better speed."""
        self._pipe.disable_attention_slicing()
        self._inpaint_pipe.disable_attention_slicing()

    @torch.no_grad()
    def _init_image(self, height: int, width: int) -> Image.Image:
        latents = torch.randn(
            (1, self._pipe.unet.in_channels, height // 8, width // 8),
            device="cuda",
            dtype=torch.float16,
        )
        latents = 1 / 0.18215 * latents
        with torch.autocast("cuda"):
            init_tensor = self._pipe.vae.decode(latents)
        init_tensor = (init_tensor["sample"] / 2 + 0.5).clamp(0, 1)
        return transforms.ToPILImage()(init_tensor[0])

    # TODO: combine these to not repeat code
    async def text_to_image(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        progress_callback: Optional[Callable[[int, int], Awaitable]] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """
        Convert a text prompt to an image.

        Arguments:
            prompt (Union[str, List[str]]) - prompts for diffusing
            height (int, optional) - image height, default 512
            width (int, optional) - image width, default 512

        Additional kwargs are passed to `StableDiffusionImg2ImgPipeline`.

        Returns:
            (List[Image.Image]) - generated image(s)
        """
        kwargs["num_inference_steps"] = kwargs.pop("num_inference_steps", 50)
        kwargs["guidance_scale"] = kwargs.pop("guidance_scale", 6.0)
        self.mode = "img2img"
        init_image = self._init_image(height, width)
        result = await self._pipe(
            prompt,
            init_image=init_image
            if isinstance(init_image, Image.Image)
            else init_image,
            strength=1.0,
            callback=progress_callback,
            **kwargs,
        )
        gc.collect()
        torch.cuda.empty_cache()
        return result["images"]

    async def image_to_image(
        self,
        prompt: Union[str, List[str]],
        init_image: Union[Image.Image, List[Image.Image]],
        mask: Optional[Image.Image] = None,
        progress_callback: Optional[Callable[[int, int], Awaitable]] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """
        Generate a new image, using another image as a starting point.

        Encompasses both traditional img2img and inpainting.

        Arguments:
            prompt (Union[str, List[str]]) - prompt to guide image generation
            init_image (Union[Image.Image, List[Image.Image]]) - starting image
            mask (Image.Image, optional) - inpainting mask, default: None

        Additional kwargs are passed to `StableDiffusionImg2ImgPipeline` or
        `StableDiffusionInpaintPipeline`.

        Returns:
            (List[Image.Image]) - generated image(s)
        """
        kwargs["strength"] = kwargs.pop("strength", 0.8)
        kwargs["num_inference_steps"] = kwargs.pop("num_inference_steps", 50)
        kwargs["guidance_scale"] = kwargs.pop("guidance_scale", 6.0)
        if mask is None:
            self.mode = "img2img"
            result = await self._pipe(
                prompt=prompt,
                init_image=init_image.convert("RGB")
                if isinstance(init_image, Image.Image)
                else init_image,
                callback=progress_callback,
                **kwargs,
            )
        else:
            self.mode = "inpaint"
            result = await self._inpaint_pipe(
                prompt=prompt,
                image=init_image.convert("RGB"),
                mask_image=mask,
                height=init_image.height,
                width=init_image.width,
                callback=progress_callback,
                **kwargs,
            )
        gc.collect()
        torch.cuda.empty_cache()
        return result["images"]

    @property
    def device(self):
        """Get the current device."""
        if self.mode == "img2img":
            return self._pipe.device
        return self._inpaint_pipe.device

    @property
    def mode(self):
        """Get/set the current mode, i.e., which model is on the GPU."""
        return self._mode

    @mode.setter
    def mode(self, new_mode: str):
        if self._mode == new_mode:
            return

        self._mode = new_mode
        if new_mode == "img2img":
            self._inpaint_pipe.to("cpu")
            self._pipe.to("cuda")
        elif new_mode == "inpaint":
            self._pipe.to("cpu")
            self._inpaint_pipe.to("cuda")
        else:
            raise ValueError(
                '`mode` should be one of "img2img" or "inpaint", '
                f'but got "{new_mode}"'
            )

        gc.collect()
        torch.cuda.empty_cache()
