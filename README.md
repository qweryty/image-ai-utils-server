# Image AI Utils Server
## Installation
### Requirements
- NVIDIA GPU with at least 10gb of VRAM is highly recomended
- You need to create your Hugging Face token [here](https://huggingface.co/docs/hub/security-tokens)
and accept terms of service [here](https://huggingface.co/CompVis/stable-diffusion-v1-4)

### Windows
- Download and extract `image_ai_utils_windows.7z` from the [releases](https://github.com/qweryty/image-ai-utils-server/releases) page
- Replace `<your-token>` in `.env` file with your Hugging Face token(see [.env section](#env-file-fields-description) for details)
- Run main.exe
- During the first run it will download stable diffusion models to directory, specified in .env file

### Linux
Python version 3.9 should be installed. Newer versions are not supported yet and older versions 
may have some unexpected problems.

```shell
python3.9 -m venv .venv
source .venv/bin/activate
pip install poetry
poetry install
mkdir models
curl -L https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -o models
curl -L https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth -o models
curl -L https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -o models
curl -L https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -o models
cp .env.example .env
```

Replace `<your-token>` in `.env` file with your Hugging Face token(see [.env section](#env-file-fields-description) for details)

Run with `python main.py`

During the first run it will download stable diffusion models to directory, specified in .env file

### Docker
TODO

## `.env` File Fields Description
- `IMAGE_AI_UTILS_USERNAME` - Username which the plugin uses to access the server (you don't need to change this field for local installation)
- `IMAGE_AI_UTILS_PASSWORD` - password which the plugin uses to access the server (you don't need to change this field for local installation)
- `HOST` - URL or IP addres of the server; one server can serve multiple URLs or IPs, `0.0.0.0` will  (you don't need to change this field for local installation)
- `PORT` - server port (you don't need to change this field for local installation, unless it conflicts with some other service)
- `PYTORCH_CUDA_ALLOC_CONF` - see https://pytorch.org/docs/stable/notes/cuda.html#memory-management
- `DIFFUSERS_CACHE_PATH` - the path where downloaded stable diffusion models will be stored
- `HUGGING_FACE_HUB_TOKEN` - token required to download stable diffusion models

## Common Problems
### main.exe closes shortly after startup
You can look into `messages.log` file, it will contain all the errors encountered during the run of the program. If you still can't solve your problem, please [report an issue](https://github.com/qweryty/image-ai-utils-server/issues/new) and attach this file in the comment

### `RuntimeError: CUDA out of memory`
For now only GPUs with 10gb or more VRAM are prooven to be working. It is possible to decrease memory requirements further
optimizations are possible, but for now I'm waiting for [this PR](https://github.com/huggingface/diffusers/pull/366) to be 
merged and new version of diffusers library repo to be released
