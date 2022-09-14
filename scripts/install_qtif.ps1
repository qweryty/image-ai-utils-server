curl -L https://download.qt.io/official_releases/qt-installer-framework/4.4.1/QtInstallerFramework-windows-x64-4.4.1.exe -o scripts
$SCRIPTS_PATH = Resolve-Path .\scripts
$QT_PATH = Join-Path -Path $SCRIPTS_PATH -ChildPath qtif
$QT_PATH = $QT_PATH.replace(':\', ':\\')
.\scripts\QtInstallerFramework-windows-x64-4.4.1.exe --platform minimal --root $QT_PATH --al -c install
