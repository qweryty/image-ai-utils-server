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
curl -L https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -o models
curl -L https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -o models
curl -L https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth -o models
curl -L https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth -o models
curl -L https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -o models
curl -L https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth -o models
cp .env.example .env
```

Replace `<your-token>` in `.env` file with your Hugging Face token(see [.env section](#env-file-fields-description) for details)

Run with `python main.py`

During the first run it will download stable diffusion models to directory, specified in .env file

### Docker
TODO

## `.env` File Fields Description
- `IMAGE_AI_UTILS_USERNAME` - username which the plugin uses to access the server (you don't need to change this field for local installation)
- `IMAGE_AI_UTILS_PASSWORD` - password which the plugin uses to access the server (you don't need to change this field for local installation)
- `HOST` - URL or IP addres of the server; one server can serve multiple URLs or IPs, `0.0.0.0` will  (you don't need to change this field for local installation)
- `PORT` - server port (you don't need to change this field for local installation, unless it conflicts with some other service)
- `PYTORCH_CUDA_ALLOC_CONF` - see https://pytorch.org/docs/stable/notes/cuda.html#memory-management
- `DIFFUSERS_CACHE_PATH` - the path where downloaded stable diffusion models will be stored
- `HUGGING_FACE_HUB_TOKEN` - token required to download stable diffusion models
- `USE_OPTIMIZED_MODE` - when enabled, stable diffusion will consume less VRAM with the cost of speed

## Common Problems
### main.exe closes shortly after startup
You can look into `messages.log` file, it will contain all the errors encountered during the run of the program. If you still can't solve your problem, please [report an issue](https://github.com/qweryty/image-ai-utils-server/issues/new) and attach this file in the comment

### `RuntimeError: CUDA out of memory`
Try enabling `USE_OPTIMIZED_MODE` in .env file.

If that didn't help and you have less than 4GB of VRAM, you are probably out of luck and need better hardware.

Another option would be to rent a VPS with GPU and running your server there.

### `OSError: Windows requires Developer Mode to be activated`
During the first run `diffusers` library needs to create some symlinks which requires developer mode or admin rights. 
If you don't want to activate developer mode, right mouse click on main.exe and choose "Run as administrator", 
you only need to do it once, next time it will work without extra privileges.
