"""Tools for intelligently uscaling images.

Functions:
    do_gobig - upscale an image using RealESRGAN and Stable Diffusion

Some contents of this file are copyright (c) 2022 Jeffrey Quesnelle and
fall under the MIT License:

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


# pylint: disable=invalid-name

# https://github.com/lowfuel/progrock-stable
from typing import Tuple, Optional, List, Callable, Awaitable

import torch
from PIL import Image, ImageDraw
from PIL.Image import Resampling
from torch import autocast

from esrgan_upscaler import upscale
from request_models import ESRGANModel
from pipeline import StablePipe


# Alternative method composites a grid of images at the positions provided
def _grid_merge(
    source: Image.Image, slices: List[Tuple[Image.Image, int, int]]
):
    source = source.convert("RGBA")
    for (
        image_slice,
        posx,
        posy,
    ) in slices:  # go in reverse to get proper stacking
        source.alpha_composite(image_slice, (posx, posy))
    return source


def _grid_coords(
    target: Tuple[int, int], original: Tuple[int, int], overlap: int
):
    # Generate a list of coordinate tuples for our sections, in order of how
    # they'll be rendered; target should be the size for the gobig result,
    # original is the size of each chunk being rendered.
    center = []
    target_x, target_y = target
    center_x = int(target_x / 2)
    center_y = int(target_y / 2)
    original_x, original_y = original
    x = center_x - int(original_x / 2)
    y = center_y - int(original_y / 2)
    center.append((x, y))  # center chunk
    uy = y  # up
    uy_list = []
    dy = y  # down
    dy_list = []
    lx = x  # left
    lx_list = []
    rx = x  # right
    rx_list = []
    while uy > 0:  # center row vertical up
        uy = uy - original_y + overlap
        uy_list.append((lx, uy))
    while (dy + original_y) <= target_y:  # center row vertical down
        dy = dy + original_y - overlap
        dy_list.append((rx, dy))
    while lx > 0:
        lx = lx - original_x + overlap
        lx_list.append((lx, y))
        uy = y
        while uy > 0:
            uy = uy - original_y + overlap
            uy_list.append((lx, uy))
        dy = y
        while (dy + original_y) <= target_y:
            dy = dy + original_y - overlap
            dy_list.append((lx, dy))
    while (rx + original_x) <= target_x:
        rx = rx + original_x - overlap
        rx_list.append((rx, y))
        uy = y
        while uy > 0:
            uy = uy - original_y + overlap
            uy_list.append((rx, uy))
        dy = y
        while (dy + original_y) <= target_y:
            dy = dy + original_y - overlap
            dy_list.append((rx, dy))
    # Calculate a new size that will fill the canvas, which will be optionally
    # used in _grid_slice and go_big.
    last_coordx, last_coordy = dy_list[-1:][0]
    render_edgey = (
        last_coordy + original_y
    )  # outer bottom edge of the render canvas
    render_edgex = (
        last_coordx + original_x
    )  # outer side edge of the render canvas
    scalarx = render_edgex / target_x
    scalary = render_edgey / target_y
    if scalarx <= scalary:
        new_edgex = int(target_x * scalarx)
        new_edgey = int(target_y * scalarx)
    else:
        new_edgex = int(target_x * scalary)
        new_edgey = int(target_y * scalary)
    # Now put all the chunks into one master list of coordinates (essentially
    # reverse of how we calculated them so that the central slices will be on
    # top).
    result = []
    for coords in dy_list[::-1]:
        result.append(coords)
    for coords in uy_list[::-1]:
        result.append(coords)
    for coords in rx_list[::-1]:
        result.append(coords)
    for coords in lx_list[::-1]:
        result.append(coords)
    result.append(center[0])
    return result, (new_edgex, new_edgey)


# Chop our source into a grid of images that each equal the size of the
# original render
def _grid_slice(
    source: Image.Image, overlap: int, og_size: Tuple[int, int]
) -> Tuple[List[Tuple[Image.Image, int, int]], int]:
    # pylint: disable=invalid-name
    width, height = og_size  # size of the slices to be rendered
    coordinates, new_size = _grid_coords(source.size, og_size, overlap)
    slices = []
    for coordinate in coordinates:
        x, y = coordinate
        slices.append(((source.crop((x, y, x + width, y + height))), x, y))
    return slices, new_size


# Upscale our image before inference.
def _prescale(
    input_image: Image.Image, esrgan_model: ESRGANModel, **kwargs
) -> Tuple[Image.Image, List[Tuple[Image.Image, int, int]], dict]:
    target_size = (
        kwargs.pop("target_width", 1024),
        kwargs.pop("target_height", 1024),
    )
    resampling_mode = kwargs.pop("resampling_mode", Resampling.LANCZOS)
    overlap = kwargs.get("overlap", 128)
    slice_size = kwargs.pop("slice_size")

    if kwargs.pop("use_real_esrgan", True):
        input_image = upscale(input_image, esrgan_model)
    target_image = input_image.resize(target_size, resampling_mode)
    slices, new_canvas_size = _grid_slice(target_image, overlap, slice_size)
    if kwargs.pop("maximize", True):
        # Increase our final image size to use up blank space.
        target_image = input_image.resize(new_canvas_size, resampling_mode)
        slices, new_canvas_size = _grid_slice(
            target_image, overlap, slice_size
        )
    input_image.close()
    return target_image, slices, kwargs


@torch.no_grad()
async def _gobig_inference(
    prompt: str,
    pipeline: StablePipe,
    slices: List[Tuple[Image.Image, int, int]],
    progress_callback: Optional[Callable[[float], Awaitable]] = None,
    **kwargs
):
    better_slices = []
    count = 0
    if progress_callback is not None:
        async def chunk_progress_callback(
            batch_step: int, total_batch_steps: int
        ):
            current_step = count * total_batch_steps + batch_step
            total_steps = len(slices) * total_batch_steps
            progress = current_step / total_steps
            await progress_callback(progress)

    else:
        chunk_progress_callback = None

    with autocast("cuda"):
        with torch.inference_mode():
            # TODO run in batches
            for count, (chunk, coord_x, coord_y) in enumerate(slices):
                result_slice = (
                    await pipeline.image_to_image(
                        prompt=prompt,
                        init_image=chunk,
                        progress_callback=chunk_progress_callback,
                        **kwargs,
                    )
                )[0]
                # result_slice.copy?
                better_slices.append((result_slice, coord_x, coord_y))

    return better_slices


# Composite slices onto a target image.
def _splice(
    image: Image.Image,
    slices: List[Tuple[Image.Image, int, int]],
    overlap: int,
    slice_size: Tuple[int, int],
) -> Image.Image:
    # Create an alpha channel for compositing the slices.
    alpha = Image.new("L", slice_size, color=0xFF)
    alpha_gradient = ImageDraw.Draw(alpha)
    # We want the alpha gradient to be half the size of the overlap,
    # otherwise we always see some of the original background underneath.
    alpha_overlap = int(overlap / 2)
    for i in range(overlap):
        shape = ((slice_size[0] - i, slice_size[1] - i), (i, i))
        fill = min(int(i * (255 / alpha_overlap)), 255)
        alpha_gradient.rectangle(shape, fill=fill)
    # Now composite the slices together.
    finished_slices = []
    for better_slice, x, y in slices:
        better_slice.putalpha(alpha)
        finished_slices.append((better_slice, x, y))
    return _grid_merge(image, finished_slices)


async def do_gobig(
    input_image: Image.Image,
    prompt: str,
    esrgan_model: ESRGANModel,
    pipeline: StablePipe,
    progress_callback: Optional[Callable[[float], Awaitable]] = None,
    **kwargs,
) -> Image.Image:
    """
    Perform high-resolution upscaling with RealESRGAN and Stable Diffusion.

    Arguments:
        input_image (Image.Image) - image to upscale
        prompt (str) - prompt to guide Stable Diffusion
        esrgan_model (ESRGANModel) - RealESRGAN model to use
        pipeline (StablePipe) - Stable Diffusion pipeline for rendering

    Optional Arguments:
        target_width (int) - final desired width (default: 1024)
        target_height (int) - final desired height (default: 1024)
        overlap (int) - fuzz amount that Stable Diffusion chunks will overlap
            (default: 128)
        maximize (bool) - increase final image size to use up blank space
            (default: True)
        use_real_esrgan (bool) - upscale with RealESRGAN rather than PIL
            (default: True)
        resampling_mode (Resampling) - method for scaling images when using PIL
            (default: Resampling.LANCZOS)
        strength (float) - strength for Stable Diffusion render (default: 0.3)
        num_inference_steps (Optional[int]) - number of diffusion iterations
            (default: 50)
        guidance_scale (Optional[float]) - guidance scale for Diffusion render
            (default: 7.5)
        progress_callback (Optional[Callable[[float], Awaitable]]) - send
            progress information to web socket (default: None)

    Additional arguments are passed to `pipeline` at render time.

    Returns:
        ret_value (Image.Image) - describe ret_value
    """
    # Get our render size for each slice, and our target size.
    slice_size = (512, 512)
    target_image, slices, kwargs = _prescale(
        input_image, esrgan_model, slice_size=slice_size, **kwargs
    )
    overlap = kwargs.pop("overlap", 128)
    kwargs["strength"] = kwargs.pop("strength", 0.3)
    kwargs["num_inference_steps"] = kwargs.pop("num_inference_steps", 50)
    kwargs["guidance_scale"] = kwargs.pop("guidance_scale", 7.5)

    # Now we trigger a do_run for each slice.
    better_slices = await _gobig_inference(
        prompt, pipeline, slices, progress_callback, **kwargs
    )
    return _splice(target_image, better_slices, overlap, slice_size)
