# pylint: disable=no-name-in-module, no-member, import-error

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
    def __init__(self, **kwargs):
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
        self._pipe.enable_attention_slicing()
        self._inpaint_pipe.enable_attention_slicing()

    def disable_attention_slicing(self):
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
    ):
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
                image=init_image.convert("RGB")
                if isinstance(init_image, Image.Image)
                else init_image,
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
        if self.mode == "img2img":
            return self._pipe.device
        return self._inpaint_pipe.device

    @property
    def mode(self):
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
