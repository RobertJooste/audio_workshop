import os
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import package_portable


class PortablePackagingTests(unittest.TestCase):
    def test_windows_package_is_standalone_executable(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable = root / 'dist' / 'EPUB Audiobook Maker.exe'
            executable.parent.mkdir()
            executable.write_bytes(b'portable executable')
            artifacts = root / 'artifacts'
            artifacts.mkdir()

            with patch.object(package_portable, 'ROOT', root):
                result = package_portable._package_windows('x64', artifacts)

            self.assertEqual(
                result.name,
                'EPUB-Audiobook-Maker-windows-x64.exe',
            )
            self.assertEqual(result.read_bytes(), b'portable executable')

    @unittest.skipIf(os.name == 'nt', 'Windows does not preserve POSIX executable bits.')
    def test_linux_archive_contains_executable_and_launch_instructions(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable = root / 'dist' / 'EPUB Audiobook Maker'
            executable.parent.mkdir()
            executable.write_bytes(b'portable executable')
            executable.chmod(0o755)
            artifacts = root / 'artifacts'
            artifacts.mkdir()

            with patch.object(package_portable, 'ROOT', root):
                result = package_portable._package_linux('x64', artifacts)

            self.assertEqual(
                result.name,
                'EPUB-Audiobook-Maker-linux-x64.tar.gz',
            )
            with tarfile.open(result, 'r:gz') as archive:
                executable_entry = archive.getmember(
                    'EPUB-Audiobook-Maker-linux-x64/EPUB Audiobook Maker'
                )
                readme = archive.extractfile(
                    'EPUB-Audiobook-Maker-linux-x64/README.txt'
                )
                self.assertIsNotNone(readme)
                self.assertIn(b'Run the "EPUB Audiobook Maker" executable', readme.read())
            self.assertTrue(executable_entry.mode & 0o111)
