#!/usr/bin/env python3
"""Build a platform-native, double-clickable desktop application."""

import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', default='0.1.0')
    args = parser.parse_args()
    if not args.version.replace('.', '').replace('-', '').isalnum():
        parser.error('Version may contain only letters, numbers, dots, and hyphens.')

    command = [
        sys.executable,
        '-m',
        'PyInstaller',
        '--noconfirm',
        '--clean',
        '--windowed',
        '--name',
        'EPUB Audiobook Maker',
        '--add-data',
        f'templates{os.pathsep}templates',
    ]
    for package in (
        'kokoro_onnx',
        'onnxruntime',
        'espeakng_loader',
        'phonemizer',
        'sounddevice',
        'soundfile',
        'fitz',
    ):
        command.extend(('--collect-all', package))
    if sys.platform == 'darwin':
        command.extend((
            '--osx-bundle-identifier',
            'com.robertjooste.epubaudiobookmaker',
        ))
    elif sys.platform in {'win32', 'linux'}:
        command.append('--onefile')
    command.append('desktop_app.py')
    subprocess.run(command, check=True)
    print(f'Built desktop app version {args.version}.')


if __name__ == '__main__':
    main()
