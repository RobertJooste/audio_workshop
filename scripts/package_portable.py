#!/usr/bin/env python3
"""Package a PyInstaller build as a portable platform download."""

import argparse
import platform
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_NAME = 'EPUB Audiobook Maker'


def _package_windows(architecture: str, artifacts: Path) -> Path:
    executable = ROOT / 'dist' / f'{APP_NAME}.exe'
    if not executable.is_file():
        raise FileNotFoundError(f'Windows executable not found: {executable}')
    destination = artifacts / f'EPUB-Audiobook-Maker-windows-{architecture}.exe'
    shutil.copy2(executable, destination)
    return destination


def _package_macos(architecture: str, artifacts: Path) -> Path:
    app_bundle = ROOT / 'dist' / f'{APP_NAME}.app'
    if not app_bundle.is_dir():
        raise FileNotFoundError(f'macOS app bundle not found: {app_bundle}')
    destination = artifacts / f'EPUB-Audiobook-Maker-macos-{architecture}.zip'
    subprocess.run(
        [
            'ditto',
            '-c',
            '-k',
            '--sequesterRsrc',
            '--keepParent',
            str(app_bundle),
            str(destination),
        ],
        check=True,
    )
    return destination


def _package_linux(architecture: str, artifacts: Path) -> Path:
    executable = ROOT / 'dist' / APP_NAME
    if not executable.is_file():
        raise FileNotFoundError(f'Linux executable not found: {executable}')
    archive_name = f'EPUB-Audiobook-Maker-linux-{architecture}'
    destination = artifacts / f'{archive_name}.tar.gz'
    with tempfile.TemporaryDirectory() as temporary_directory:
        staging = Path(temporary_directory) / archive_name
        staging.mkdir()
        staged_executable = staging / APP_NAME
        shutil.copy2(executable, staged_executable)
        readme = staging / 'README.txt'
        readme.write_text(
            'EPUB Audiobook Maker — portable Linux build\n\n'
            'Run the "EPUB Audiobook Maker" executable in this folder. '
            'If your desktop does not launch executables on double-click, '
            'open a terminal here and run: ./EPUB\\ Audiobook\\ Maker\n\n'
            'The first use downloads the Kokoro model and voices (~192 MB) '
            'and requires an internet connection. Audiobooks are saved in '
            '~/Audiobooks.\n',
            encoding='utf-8',
        )
        with tarfile.open(destination, 'w:gz') as archive:
            archive.add(staging, arcname=archive_name)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform', choices=('windows', 'macos', 'linux'), required=True)
    parser.add_argument('--architecture', choices=('x64', 'arm64'), required=True)
    args = parser.parse_args()

    host_platform = {
        'Windows': 'windows',
        'Darwin': 'macos',
        'Linux': 'linux',
    }.get(platform.system())
    if host_platform != args.platform:
        parser.error(f'Cannot package {args.platform} artifacts on {platform.system()}.')
    if args.platform == 'windows' and args.architecture != 'x64':
        parser.error('Only Windows x64 builds are currently configured.')
    if args.platform == 'linux' and args.architecture != 'x64':
        parser.error('Only Linux x64 builds are currently configured.')

    artifacts = ROOT / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    package = {
        'windows': _package_windows,
        'macos': _package_macos,
        'linux': _package_linux,
    }[args.platform](args.architecture, artifacts)
    print(f'Created portable download: {package}')


if __name__ == '__main__':
    main()
