var TEMP_DIRECTORY = null;

function Component() {
    TEMP_DIRECTORY = installer.value('HomeDir') + '/AppData/Local/Temp'
    //installer.setValue('Test', 'Hello')
    //QMessageBox.information('test', 'test', installer.value('Test'))

    if (installer.isInstaller()) {
        copied = installer.performOperation('Copy', ['://validate_token.exe', TEMP_DIRECTORY]);
        installer.addWizardPage(component, 'SettingsWidget', QInstaller.ReadyForInstallation);
        component.loaded.connect(this, Component.prototype.installerLoaded);
    }
}

Component.prototype.installerLoaded = function() {
    var page = gui.pageWidgetByObjectName('DynamicSettingsWidget');
    page.complete = false;
    page.tokenLineEdit.textChanged.connect(this, Component.prototype.tokenChanged)
    page.validatePushButton.clicked.connect(this, Component.prototype.validateToken)
    page.validatePushButton.enabled = false;
}

Component.prototype.tokenChanged = function(text) {
    var page = gui.pageWidgetByObjectName('DynamicSettingsWidget');
    page.complete = false;
    page.validatePushButton.enabled = text != '';
}

Component.prototype.validateToken = function() {
    var page = gui.pageWidgetByObjectName('DynamicSettingsWidget');
    response = installer.execute(TEMP_DIRECTORY + '/validate_token.exe', [page.tokenLineEdit.text]);
    if (response.length !== 2) {
        page.complete = false;
        page.successStatusLabel.text = 'Couldn\'t execute checker';
    }
    let status, message;
    [message, status] = response;
    page.successStatusLabel.text = message;
    if (status != 0) {
        page.complete = false;
        return;
    }

    installer.setValue('HuggingFaceToken', page.tokenLineEdit.text)
    page.complete = true;
}
