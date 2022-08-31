import mimetypes
from base64 import b64decode, b64encode
from io import BytesIO

from PIL import Image


# TODO use common utils
def image_to_base64url(image: Image.Image, output_format: str = 'PNG') -> bytes:
    data_string = f'data:{mimetypes.types_map[f".{output_format.lower()}"]};base64,'.encode()
    buffer = BytesIO()
    image.save(buffer, format=output_format)
    return data_string + b64encode(buffer.getvalue())


def base64url_to_image(source: bytes) -> Image.Image:
    _, data = source.split(b',')
    return Image.open(BytesIO(b64decode(data)))
