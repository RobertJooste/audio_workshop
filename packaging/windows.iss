#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

[Setup]
AppId={{B1E3B03C-8E4B-45E3-9D66-9D3A73193A67}
AppName=EPUB Audiobook Maker
AppVersion={#AppVersion}
AppPublisher=Robert Jooste
DefaultDirName={localappdata}\Programs\EPUB Audiobook Maker
DefaultGroupName=EPUB Audiobook Maker
PrivilegesRequired=lowest
OutputDir=..\artifacts
OutputBaseFilename=EPUB-Audiobook-Maker-Setup-windows-x64
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\EPUB Audiobook Maker.exe

[Files]
Source: "..\dist\EPUB Audiobook Maker.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\EPUB Audiobook Maker"; Filename: "{app}\EPUB Audiobook Maker.exe"
Name: "{autodesktop}\EPUB Audiobook Maker"; Filename: "{app}\EPUB Audiobook Maker.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Run]
Filename: "{app}\EPUB Audiobook Maker.exe"; Description: "Launch EPUB Audiobook Maker"; Flags: postinstall nowait skipifsilent
