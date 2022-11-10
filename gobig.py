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
def grid_merge(
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


def grid_coords(
    target: Tuple[int, int], slice_size: Tuple[int, int], overlap: int
):
    # generate a list of coordinate tuples for our sections, in order of how they'll be rendered
    # target should be the size for the gobig result, original is the size of each chunk being
    # rendered
    center = []
    target_x, target_y = target
    center_x = int(target_x / 2)
    center_y = int(target_y / 2)
    slice_x, slice_y = slice_size
    x = center_x - int(slice_x / 2)
    y = center_y - int(slice_y / 2)
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
        uy = uy - slice_y + overlap
        uy_list.append((lx, uy))
    while (dy + slice_y) <= target_y:  # center row vertical down
        dy = dy + slice_y - overlap
        dy_list.append((rx, dy))
    while lx > 0:
        lx = lx - slice_x + overlap
        lx_list.append((lx, y))
        uy = y
        while uy > 0:
            uy = uy - slice_y + overlap
            uy_list.append((lx, uy))
        dy = y
        while (dy + slice_y) <= target_y:
            dy = dy + slice_y - overlap
            dy_list.append((lx, dy))
    while (rx + slice_x) <= target_x:
        rx = rx + slice_x - overlap
        rx_list.append((rx, y))
        uy = y
        while uy > 0:
            uy = uy - slice_y + overlap
            uy_list.append((rx, uy))
        dy = y
        while (dy + slice_y) <= target_y:
            dy = dy + slice_y - overlap
            dy_list.append((rx, dy))
    # calculate a new size that will fill the canvas, which will be optionally used in grid_slice and go_big
    last_coordx, last_coordy = dy_list[-1:][0]
    render_edgey = (
        last_coordy + slice_y
    )  # outer bottom edge of the render canvas
    render_edgex = (
        last_coordx + slice_x
    )  # outer side edge of the render canvas
    scalarx = render_edgex / target_x
    scalary = render_edgey / target_y
    if scalarx <= scalary:
        new_edgex = int(target_x * scalarx)
        new_edgey = int(target_y * scalarx)
    else:
        new_edgex = int(target_x * scalary)
        new_edgey = int(target_y * scalary)
    # now put all the chunks into one master list of coordinates (essentially reverse of how we calculated them so that the central slices will be on top)
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


def grid_slice(source: Image.Image, overlap: int, slice_size: Tuple[int, int]):
    width, height = slice_size
    coordinates, new_size = grid_coords(source.size, slice_size, overlap)
    slices = []
    for coordinate in coordinates:
        x, y = coordinate
        slices.append(((source.crop((x, y, x + width, y + height))), x, y))
    return slices, new_size


async def do_gobig(
    input_image: Image.Image,
    prompt: str,
    maximize: bool,
    target_width: int,
    target_height: int,
    overlap: int,
    use_real_esrgan: bool,
    esrgan_model: ESRGANModel,
    pipeline: StablePipe,
    resampling_mode: Resampling = Resampling.LANCZOS,
    strength: float = 0.8,
    num_inference_steps: Optional[int] = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    progress_callback: Optional[Callable[[float], Awaitable]] = None,
) -> Image.Image:
    # get our render size for each slice, and our target size
    slice_width = slice_height = 512
    if use_real_esrgan:
        input_image = upscale(input_image, esrgan_model)
    target_image = input_image.resize(
        (target_width, target_height), resampling_mode
    )
    slices, new_canvas_size = grid_slice(
        target_image, overlap, (slice_width, slice_height)
    )
    if maximize:
        # increase our final image size to use up blank space
        target_image = input_image.resize(new_canvas_size, resampling_mode)
        slices, new_canvas_size = grid_slice(
            target_image, overlap, (slice_width, slice_height)
        )
    input_image.close()
    # now we trigger a do_run for each slice
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
                        strength=strength,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        progress_callback=chunk_progress_callback,
                    )
                )[0]
                # result_slice.copy?
                better_slices.append((result_slice, coord_x, coord_y))

    # create an alpha channel for compositing the slices
    alpha = Image.new("L", (slice_width, slice_height), color=0xFF)
    alpha_gradient = ImageDraw.Draw(alpha)
    # we want the alpha gradient to be half the size of the overlap,
    # otherwise we always see some of the original background underneath
    alpha_overlap = int(overlap / 2)
    for i in range(overlap):
        shape = ((slice_width - i, slice_height - i), (i, i))
        fill = min(int(i * (255 / alpha_overlap)), 255)
        alpha_gradient.rectangle(shape, fill=fill)
    # now composite the slices together
    finished_slices = []
    for better_slice, x, y in better_slices:
        better_slice.putalpha(alpha)
        finished_slices.append((better_slice, x, y))
    final_output = grid_merge(target_image, finished_slices)

    return final_output
