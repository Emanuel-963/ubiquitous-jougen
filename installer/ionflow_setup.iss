; ──────────────────────────────────────────────────────────────
; IonFlow Pipeline — Inno Setup Installer Script
; ──────────────────────────────────────────────────────────────
; Prerequisites:
;   1. Build with PyInstaller:  python build_exe.py
;   2. The dist\IonFlow_Pipeline\ folder must exist
;   3. Install Inno Setup 6+:  https://jrsoftware.org/isinfo.php
;   4. Compile:  ISCC.exe installer\ionflow_setup.iss
; ──────────────────────────────────────────────────────────────

#define MyAppName      "IonFlow Pipeline"
#define MyAppVersion   "0.2.0"
#define MyAppPublisher "Emanuel"
#define MyAppURL       "https://github.com/Emanuel-963/ubiquitous-jougen"
#define MyAppExeName   "IonFlow_Pipeline.exe"

; Adjust this path if your dist folder is elsewhere
#define DistDir        "..\dist\IonFlow_Pipeline"

[Setup]
AppId={{8F3A5D72-2E1B-4C6F-9D0A-7B8E1F2C3D4E}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputDir=..\dist\installer
OutputBaseFilename=IonFlow_Pipeline_Setup_{#MyAppVersion}
SetupIconFile=..\data\ionflow.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}
VersionInfoVersion={#MyAppVersion}.0
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription={#MyAppName} installer
VersionInfoProductName={#MyAppName}

[Languages]
Name: "portuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"
Name: "english";    MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Copy the entire PyInstaller dist folder
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}";         Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Desinstalar {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}";   Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
// Show a summary page at the end of the installer
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    MsgBox('IonFlow Pipeline foi instalado com sucesso!' + #13#10 +
           'Acesse pelo menu Iniciar ou atalho na área de trabalho.',
           mbInformation, MB_OK);
end;
