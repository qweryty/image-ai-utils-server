import logging

import numpy as np
from PIL import Image
from gfpgan import GFPGANer

from esrgan_upscaler import get_upsampler
from request_models import ESRGANModel, GFPGANModel
from utils import resolve_path

logger = logging.getLogger(__name__)


GFPGAN_URLS = [
    'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
    'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth',
    'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
]

MODEL_PATHS = {
    # path, channel_multiplier, arch
    GFPGANModel.V1_3: ['models/GFPGANv1.3.pth', 2, 'clean'],
    GFPGANModel.V1_2: ['models/GFPGANCleanv1-NoCE-C2.pth', 2, 'clean'],
    GFPGANModel.V1: ['models/GFPGANv1.pth', 1, 'original'],
}

for key, value in MODEL_PATHS.items():
    MODEL_PATHS[key][0] = resolve_path(value[0])


def restore_face(
        image: Image.Image,
        model_type: GFPGANModel = GFPGANModel.V1_3,
        use_real_esrgan: bool = True,
        bg_tile: int = 400,
        upscale: int = 2,
        aligned: bool = False,
        only_center_face: bool = False
):
    # ------------------------ set up background upsampler ------------------------
    if use_real_esrgan:
        bg_upsampler = get_upsampler(model_type=ESRGANModel.X2_PLUS, tile=bg_tile)
    else:
        bg_upsampler = None

    # ------------------------ set up GFPGAN restorer ------------------------
    # determine model paths
    model_path, channel_multiplier, arch = MODEL_PATHS[model_type]

    restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler
    )

    # ------------------------ restore ------------------------
    input_img = np.asarray(image.convert('RGB'))

    # restore faces and background if necessary
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        input_img,
        has_aligned=aligned,
        only_center_face=only_center_face,
        paste_back=True
    )

    return Image.fromarray(restored_img)
