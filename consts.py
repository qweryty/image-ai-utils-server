"""
Server constants.

Classes:
    ESRGANModel - upscaling constants
    GFPGANModel - face-fixing constants
    ScalingMode - shrink/grow constants
    ImageFormat - file format constants
    WebSocketResponseStatus - web socket status constants

Variables:
    MIN_SEED (int) - minimum allowed random seed
    MAX_SEED (int) - maximum allowed random seed
"""


from enum import Enum


class ESRGANModel(str, Enum):
    """Model names for upscaling."""

    # General
    GENERAL_X4_V3 = "general_x4_v3"
    X4_PLUS = "x4_plus"
    X2_PLUS = "x2_plus"
    ESRNET_X4_PLUS = "x4_plus"
    OFFICIAL_X4 = "official_x4"

    # Anime/Illustrations
    X4_PLUS_ANIME_6B = "x4_plus_anime_6b"

    # Anime video
    ANIME_VIDEO_V3 = "anime_video_v3"


class GFPGANModel(str, Enum):
    """Model names for fixing faces."""

    V1_3 = "V1.3"
    V1_2 = "V1.2"
    V1 = "V1"


class ScalingMode(str, Enum):
    """Constants regarding scaling modes."""

    SHRINK = "shrink"
    GROW = "grow"


class ImageFormat(str, Enum):
    """Possible image file formats."""

    PNG = "PNG"
    JPEG = "JPEG"
    BMP = "BMP"


class WebSocketResponseStatus(str, Enum):
    """Possible web socket statuses."""

    FINISHED = "finished"
    PROGRESS = "progress"


MIN_SEED = -0x8000_0000_0000_0000
MAX_SEED = 0xFFFF_FFFF_FFFF_FFFF
