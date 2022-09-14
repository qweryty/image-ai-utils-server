pyinstaller src\main.py --hidden-import colorlog --collect-all huggingface_hub --additional-hooks-dir pyinstaller_hooks -y
Move-Item -Path dist\main -Destination install\packages\top.morozov.image-ai-utils-server\data
copy .env.example install\packages\top.morozov.image-ai-utils-server\data\.env
copy LICENSE.md install\packages\top.morozov.image-ai-utils-server\meta\
cd install\packages\top.morozov.image-ai-utils-server\data\

mkdir basicsr\archs, basicsr\data, basicsr\losses, basicsr\models
mkdir realesrgan\archs, realesrgan\data, realesrgan\losses, realesrgan\models
mkdir gfpgan\archs, gfpgan\data, gfpgan\losses, gfpgan\models

cd ..\..\..\..

.\scripts\qtif\bin\binarycreator.exe -c .\install\config\config.xml -p .\install\packages\ -f .\build\installer.exe