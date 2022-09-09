import os.path
from enum import Enum

import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

from request_models import ESRGANModel
from utils import resolve_path


MODEL_PATHS = {
    ESRGANModel.GENERAL_X4_V3: 'models/realesr-general-x4v3.pth',
    ESRGANModel.X4_PLUS: 'models/RealESRGAN_x4plus.pth',
    ESRGANModel.X2_PLUS: 'models/RealESRGAN_x2plus.pth',
    ESRGANModel.ESRNET_X4_PLUS: 'models/RealESRNet_x4plus.pth',
    ESRGANModel.OFFICIAL_X4: 'models/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth',
    ESRGANModel.ANIME_VIDEO_V3: 'models/realesr-animevideov3.pth',
    ESRGANModel.X4_PLUS_ANIME_6B: 'models/RealESRGAN_x4plus_anime_6B.pth',
}

for key, value in MODEL_PATHS.items():
    MODEL_PATHS[key] = resolve_path(value)


def upscale(
        image: Image.Image,
        model_type: ESRGANModel = ESRGANModel.GENERAL_X4_V3,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 0,
        half: bool = True,
        outscale: float = 4
):
    # x4 RRDBNet model
    if model_type in [ESRGANModel.X4_PLUS, ESRGANModel.ESRNET_X4_PLUS, ESRGANModel.OFFICIAL_X4]:
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
        )
        netscale = 4
    elif model_type in [ESRGANModel.X4_PLUS_ANIME_6B]:  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4
        )
        netscale = 4
    elif model_type in [ESRGANModel.X2_PLUS]:  # x2 RRDBNet model
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2
        )
        netscale = 2
    elif model_type in [ESRGANModel.ANIME_VIDEO_V3]:  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'
        )
        netscale = 4
    elif model_type in [ESRGANModel.GENERAL_X4_V3]:
        model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu'
        )
        netscale = 4
    else:
        raise ValueError('Incorrect model')  # TODO custom exception

    # determine model paths
    model_path = MODEL_PATHS[model_type]
    # restorer
    # TODO use gpu(available in newer version)
    # TODO store upsampler
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not half,
    )

    numpy_image = np.asarray(image)
    output, _ = upsampler.enhance(numpy_image, outscale=outscale)
    return Image.fromarray(np.uint8(output))
