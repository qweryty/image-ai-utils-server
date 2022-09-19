# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = ['colorlog']
tmp_ret = collect_all('huggingface_hub')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


block_cipher = None


main_analysis = Analysis(
    ['src\\main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['pyinstaller_hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
main_pyz = PYZ(main_analysis.pure, main_analysis.zipped_data, cipher=block_cipher)

main_exe = EXE(
    main_pyz,
    main_analysis.scripts,
    [],
    exclude_binaries=True,
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)



download_models_analysis = Analysis(
    ['src\\download_models.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['pyinstaller_hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
download_models_pyz = PYZ(
    download_models_analysis.pure, download_models_analysis.zipped_data, cipher=block_cipher
)

download_models_exe = EXE(
    download_models_pyz,
    download_models_analysis.scripts,
    [],
    exclude_binaries=True,
    name='download_models',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    main_exe,
    main_analysis.binaries,
    main_analysis.zipfiles,
    main_analysis.datas,
    download_models_exe,
    download_models_analysis.binaries,
    download_models_analysis.zipfiles,
    download_models_analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='image_ai_utils_server',
)
