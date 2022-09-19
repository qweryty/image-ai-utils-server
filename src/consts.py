from enum import Enum


class ESRGANModel(str, Enum):
    # General
    GENERAL_X4_V3 = 'general_x4_v3'
    X4_PLUS = 'x4_plus'
    X2_PLUS = 'x2_plus'
    ESRNET_X4_PLUS = 'x4_plus'
    OFFICIAL_X4 = 'official_x4'

    # Anime/Illustrations
    X4_PLUS_ANIME_6B = 'x4_plus_anime_6b'

    # Anime video
    ANIME_VIDEO_V3 = 'anime_video_v3'


class GFPGANModel(str, Enum):
    V1_3 = 'V1.3'
    V1_2 = 'V1.2'
    V1 = 'V1'


class ScalingMode(str, Enum):
    SHRINK = 'shrink'
    GROW = 'grow'


class ImageFormat(str, Enum):
    PNG = 'PNG'
    JPEG = 'JPEG'
    BMP = 'BMP'


class WebSocketResponseStatus(str, Enum):
    FINISHED = 'finished'
    PROGRESS = 'progress'


MIN_SEED = -0x8000_0000_0000_0000
MAX_SEED = 0xffff_ffff_ffff_ffff

STABLE_DIFFUSION_MODEL_NAME = 'CompVis/stable-diffusion-v1-4'
STABLE_DIFFUSION_REVISION = 'fp16'
