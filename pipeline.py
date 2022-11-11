"""
Provide a wrapper around diffusers' stable diffusion pipelines.

Classes:
    StablePipe - wrapper for both img2img and inpaint pipelines
"""


import gc
from typing import Union, List, Optional  # , Awaitable, Callable

from PIL import Image
import torch
from torchvision import transforms

from diffusers.pipelines.stable_diffusion import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
)


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
        self._inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            inpaint_version,
            use_auth_token=True,
            safety_checker=None,
            **kwargs,
        )

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
    # TODO: use progress_callback
    async def text_to_image(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        #  progress_callback: Optional[Callable[[int, int], Awaitable]] = None,
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
        result = self._pipe(
            prompt,
            init_image=init_image
            if isinstance(init_image, Image.Image)
            else init_image,
            strength=1.0,
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
        #  progress_callback: Optional[Callable[[int, int], Awaitable]] = None,
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
            result = self._pipe(
                prompt=prompt,
                init_image=init_image.convert("RGB")
                if isinstance(init_image, Image.Image)
                else init_image,
                **kwargs,
            )
        else:
            self.mode = "inpaint"
            result = self._inpaint_pipe(
                prompt=prompt,
                image=init_image.convert("RGB"),
                mask_image=mask,
                height=init_image.height,
                width=init_image.width,
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
