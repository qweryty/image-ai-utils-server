import logging
import mimetypes
import os
from base64 import b64decode, b64encode
from io import BytesIO
from typing import Tuple

from PIL import Image

from consts import ScalingMode

# TODO use common utils

logger = logging.getLogger(__name__)


def image_to_base64url(image: Image.Image, output_format: str = 'PNG') -> bytes:
    data_string = f'data:{mimetypes.types_map[f".{output_format.lower()}"]};base64,'.encode()
    buffer = BytesIO()
    image.save(buffer, format=output_format)
    return data_string + b64encode(buffer.getvalue())


def base64url_to_image(source: bytes) -> Image.Image:
    _, data = source.split(b',')
    return Image.open(BytesIO(b64decode(data)))


def size_from_aspect_ratio(aspect_ratio: float, scaling_mode: ScalingMode) -> Tuple[int, int]:
    if (scaling_mode == ScalingMode.GROW) == (aspect_ratio > 1):
        height = 512
        width = int(height * aspect_ratio)
        width -= width % 64
    else:
        width = 512
        height = int(width / aspect_ratio)
        height -= height % 64
    return width, height


def resolve_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))
