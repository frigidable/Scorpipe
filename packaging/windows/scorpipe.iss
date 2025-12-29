; Inno Setup script for Scorpipe
; Requires Inno Setup 6.

#define AppName "Scorpipe"
#define AppExeName "scorpipe.exe"
#ifndef AppVersion
#error AppVersion is not defined. Pass /DAppVersion=<version> to ISCC.
#endif
#define AppPublisher "Scorpipe"

[Setup]
AppId={{C9705E22-9AD2-4B7F-A0A4-0D3C84F151C4}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputBaseFilename=ScorpioPipe-Setup-x64-{#AppVersion}
OutputDir={#SourcePath}\Output
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
; Copy the PyInstaller folder contents (dist/scorpipe/*) into the installation directory.
Source: "..\..\dist\scorpipe\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent