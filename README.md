# Image AI Utils Server
## Installation
### Requirements
Python version 3.9 should be installed. Newer versions are not supported yet and older versions 
may have some unexpected problems.

You need to create your Hugging Face token [here](https://huggingface.co/docs/hub/security-tokens)
and accept terms of service [here](https://huggingface.co/CompVis/stable-diffusion-v1-4)

### Windows
TODO

### Linux
```shell
python3.9 -m venv .venv
source .venv/bin/activate
pip install poetry
poetry install
cp .env.example .env
```

Edit `.env` file with your credentials and Hugging Face token.

Run with `python main.py`

### Docker
TODO