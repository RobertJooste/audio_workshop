# EPUB Audiobook Maker

Turn an EPUB or PDF into chapter-by-chapter WAV audio using the local Kokoro ONNX text-to-speech model. The project includes a browser-based interface for EPUB books and a command-line reader for EPUB and PDF.

Audio is generated locally by Kokoro. You do not need an API key or a cloud TTS account. The first run downloads the Kokoro model and voice data.
The app uses the Kokoro v1.0 fp16 model and its matching voice bundle (about 192 MB total); English voices include `af_heart`. The app supports American and British English narration.

## Requirements

- Python 3.10–3.13
- Internet access the first time Kokoro model files are needed
- Speakers/headphones for CLI playback; the web app only writes files

## Install

From the project directory, create and activate a virtual environment and install dependencies:

```sh
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
venv\Scripts\Activate.ps1
```

The commands below use the project environment directly, so activating it is optional:

```sh
./venv/bin/python ...
```

## Web app

Start the local web app:

```sh
./venv/bin/python web_app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

1. Select an EPUB and click **Load chapters**.
2. Select the chapters to include. All chapters are selected by default; use **Select all** for the whole book.
3. Choose a narrator voice and click **Start generation**.
4. Follow progress, pause/continue between chunks, or stop and resume the job.
5. Download completed WAV files from the **Generated files** list.

Before starting, you can optionally blend a second bundled voice, adjust reading speed, set pauses after chapter headings and between sentences, and add case-insensitive word or phrase pronunciation replacements. Blended voices are experimental and may sound better or worse depending on the pair, so try a short chapter first. Custom settings are saved in their own output subfolder to keep different versions separate.

The web app only generates files; it does not play audio in the browser. Output is stored in the project's `processed/` directory:

```text
processed/<Book Title>/<Voice Name - Voice ID>/Book <content hash>/Settings <settings hash>/
```

For example:

```text
processed/The Eye of the World/Bella - af_bella/Book 0123456789abcdef/Settings 0123456789ab/
```

Different source books and custom settings are kept separate to avoid reusing the wrong audio. The web app accepts EPUB files up to 100 MB and binds to `127.0.0.1` by default, so it is intended for use on the same computer. Uploaded books and job state are stored locally under `~/.cache/epub-tts-web`; unused previews expire after one hour, stopped or failed jobs expire after one day, and paused jobs are automatically stopped after one day of inactivity. Expired data is cleaned up on subsequent app requests.

## Desktop packages

Tagged GitHub releases build double-clickable installers for Windows x64, macOS Apple Silicon, and macOS Intel. The Windows installer creates a Start menu shortcut (and offers an optional desktop shortcut); the macOS disk image contains the app, which can be dragged into Applications or opened directly. The desktop app opens the local interface in the default browser. Use **Quit app** in the page to stop it cleanly.

The installers do not include the Kokoro model. On first use, the app downloads the model and voices (about 192 MB), so an internet connection is required; later launches use the cached files. Audiobooks are saved under `~/Audiobooks`, while app data and logs are stored in the operating system's per-user data folders. The packages are currently unsigned: Windows may show a SmartScreen warning, and macOS may require approval in Privacy & Security or opening the app with Control-click.

To create a release, push a version tag such as `v0.1.0`. GitHub Actions runs the tests, builds all three platform packages, and attaches them to a GitHub Release. You can also run the workflow manually to build downloadable artifacts without publishing a release.

To build the desktop app locally:

```sh
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements-build.txt
python scripts/build_desktop.py
```

The build produces a macOS `.app` bundle on macOS and a Windows `.exe` on Windows. The installer and disk-image wrappers are created by the release workflow.

## Command-line reader

Read and play an EPUB or PDF:

```sh
./venv/bin/python epub_pdf_tts_reader.py book.epub
./venv/bin/python epub_pdf_tts_reader.py book.pdf
```

Save audio without playing it:

```sh
./venv/bin/python epub_pdf_tts_reader.py book.epub --save audiobook.wav --no-play
```

For books with multiple chapters, WAV files are named with each chapter's number and title. Continue an interrupted CLI export with `--resume`:

```sh
./venv/bin/python epub_pdf_tts_reader.py book.epub --save audiobook.wav --no-play --resume
```

Choose another Kokoro voice, change reading speed, or set the language:

```sh
./venv/bin/python epub_pdf_tts_reader.py book.epub --voice af_nicole --speed 0.9 --lang en-us
./venv/bin/python epub_pdf_tts_reader.py book.epub --voice af_heart --no-play --save audiobook.wav
```

Show the supported English voices available in the installed model:

```sh
./venv/bin/python epub_pdf_tts_reader.py --list-voices
```

The reader supports American and British English voice presets only, with `en-us` or `en-gb` pronunciation settings.

CLI audio playback supports Space to pause or continue in an interactive terminal. The web app's pause/continue controls stop generation at chunk boundaries.

## Output formats and limitations

- Audio export is WAV; MP3 export is not currently implemented.
- The web app supports EPUB input. The CLI supports EPUB and PDF input.
- PDF files need a text layer; scanned image-only PDFs require OCR, which is not included.
- EPUB extraction follows the book's reading order and filters common navigation and footnote content. When the EPUB table of contents links to chapter anchors, those links are used to split named chapters (including character-POV chapters) even when one HTML file contains several chapters or a chapter continues across files.
- A leading EPUB chapter heading is synthesized separately, followed by a one-second pause before the chapter text. The web app lets you adjust this.
- Kokoro is a local synthetic voice model; voice labels describe available presets, not individual human narrators.

## Tests

Run the test suite from the project directory:

```sh
./venv/bin/python -m unittest discover -s tests -v
```

Tests cover EPUB extraction and chapter selection, text chunking, audio export and resume, and web upload, progress controls, and file downloads.
