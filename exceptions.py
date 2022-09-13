from fastapi import HTTPException, status


class BaseWebSocketException(Exception):
    message = 'Unexpected exception, check server logs for details'


class BatchSizeIsTooLargeException(BaseWebSocketException):
    def __init__(self, batch_size):
        self.message = f'Couldn\'t fit {batch_size} images with such aspect ratio into ' \
                       f'memory. Try using smaller batch size or enabling ' \
                       f'try_smaller_batch_on_fail option'


class AspectRatioTooWideException(BaseWebSocketException):
    message = 'Couldn\'t fit image with such aspect ratio into memory, ' \
              'try using another scaling mode'


class CouldntFixFaceException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Couldn\'t fix faces. GFPGANer returned None'
        )
