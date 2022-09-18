import sys

import requests
from huggingface_hub import model_info
from requests import HTTPError

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Validator requires token as first argument')
        sys.exit(1)

    token = sys.argv[1]
    try:
        model_info('CompVis/stable-diffusion-v1-4', revision='fp16', token=token)
    except HTTPError as e:
        status_code = e.response.status_code
        if status_code == 401:
            print('Token is invalid')
        elif status_code == 403:
            print('Couldn\'t access model, try accepting terms of service')
        else:
            print(f'Server returned unknown status code: {status_code}, try again later')
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f'Couldn\'t connect to server, check your internet connection or try again later')
        sys.exit(1)
    except Exception as e:
        print(f'Unknown error: {e}')
        sys.exit(1)

    print('Success')
