pyinstaller main.spec
pyinstaller validate_token.spec

Move-Item -Path dist\main -Destination install\packages\top.morozov.image_ai_utils_server\data
Move-Item dist\validate_token.exe -Destination install\resources

copy .env.example install\packages\top.morozov.image_ai_utils_server\data\.env
copy LICENSE.md install\packages\top.morozov.image_ai_utils_server\meta\
cd install\packages\top.morozov.image_ai_utils_server\data\

mkdir basicsr\archs, basicsr\data, basicsr\losses, basicsr\models
mkdir realesrgan\archs, realesrgan\data, realesrgan\losses, realesrgan\models
mkdir gfpgan\archs, gfpgan\data, gfpgan\losses, gfpgan\models

cd ..\..\..\..

.\scripts\qtif\bin\binarycreator.exe -c .\install\config\config.xml -p .\install\packages\ -r .\install\resources\resources.qrc -f .\build\installer.exe