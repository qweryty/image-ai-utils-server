# Image AI Utils Server
## Installation
### Requirements
Python version 3.9 should be installed. Newer versions are not supported yet and older versions 
may have some unexpected problems.

You need to create your Hugging Face token [here](https://huggingface.co/docs/hub/security-tokens)
and accept terms of service [here](https://huggingface.co/CompVis/stable-diffusion-v1-4)

### Windows
- Download and extract `image_ai_utils_windows.7z` from the [releases](https://github.com/qweryty/image-ai-utils-server/releases) page
- Edit `.env` file with your credentials and Hugging Face token
- Run main.exe

### Linux
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

Edit `.env` file with your credentials and Hugging Face token.

Run with `python main.py`

### Docker
TODO
