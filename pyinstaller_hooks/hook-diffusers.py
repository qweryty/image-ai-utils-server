from PyInstaller.utils.hooks import collect_all
from diffusers.dependency_versions_table import deps


def hook(hook_api):
    packages = deps.keys()
    print('---------------------- IMPORTING DIFFUSERS DEPS ----------------------')
    print(list(packages))

    for package in packages:
        datas, _, hidden_imports = collect_all(package)
        hook_api.add_datas(datas)
        hook_api.add_imports(*hidden_imports)
