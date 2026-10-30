#!/usr/bin/env python3
"""EPUB/PDF TTS reader using Kokoro-ONNX (local, free, natural-sounding)."""

import argparse
import hashlib
import os
import posixpath
import re
import select
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

import fitz
import numpy as np
import sounddevice as sd
import soundfile as sf
from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
from kokoro_onnx import Kokoro

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_VOICE = 'af_bella'
HEADING_PAUSE_SECONDS = 1.0
DEFAULT_SENTENCE_PAUSE_SECONDS = 0.25
TTS_PIPELINE_VERSION = 'kokoro-v1.0-fp16-toc-chapters-v4'
SUPPORTED_ENGLISH_LANGUAGES = ('en-us', 'en-gb')

_RELEASE = 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1'
_MODEL_CACHE = Path.home() / '.cache' / 'kokoro-onnx' / 'v1.0'
_MODEL_FILE = 'kokoro-v1.0.fp16.onnx'
_VOICES_FILE = 'voices-v1.0.bin'
_PROGRESS_ROOT = Path.home() / '.cache' / 'epub-tts'


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _download_if_missing(filename: str) -> Path:
    dest = _MODEL_CACHE / filename
    if not dest.exists():
        _MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Downloading {filename}...")
        temporary = dest.with_name(f'{dest.name}.download')
        try:
            urllib.request.urlretrieve(f'{_RELEASE}/{filename}', temporary)
            temporary.replace(dest)
        finally:
            temporary.unlink(missing_ok=True)
    return dest


def load_model() -> Kokoro:
    model_path = _download_if_missing(_MODEL_FILE)
    voices_path = _download_if_missing(_VOICES_FILE)
    print("[INFO] Loading Kokoro model...")
    return Kokoro(str(model_path), str(voices_path))


def english_voices(voices: list[str]) -> list[str]:
    return [voice for voice in voices if voice.startswith(('af_', 'am_', 'bf_', 'bm_'))]


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

@dataclass
class Chapter:
    index: int
    title: str
    text: str
    speech_heading: str | None = None
    speech_body: str | None = None


def extract_chapters_from_pdf(path: str) -> list[Chapter]:
    with fitz.open(path) as doc:
        toc = doc.get_toc()
        top = [(title, page - 1) for level, title, page in toc if level == 1]

        if not top:
            text = "\n".join(page.get_text() for page in doc)
            return [Chapter(index=0, title="Book", text=text)]

        chapters = []
        for i, (title, start) in enumerate(top):
            end = top[i + 1][1] if i + 1 < len(top) else len(doc)
            text = "\n".join(doc[p].get_text() for p in range(start, end))
            chapters.append(Chapter(index=i, title=title, text=text))
        return chapters


def _heading_for_speech(text: str) -> str:
    if text.rstrip().endswith(('.', '!', '?', '…')):
        return text
    return f'{text}.'


_CHAPTER_LABEL = re.compile(
    r'^(?:chapter|part|book|section)\s+'
    r'(?:\d+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten)'
    r'\b(?:\s*(?:[:.—–-]\s*|\s+).*)?$',
    re.IGNORECASE,
)
_STANDALONE_HEADING = re.compile(
    r'^(?:prologue|epilogue|preface|foreword|afterword|introduction)$',
    re.IGNORECASE,
)
_CHAPTER_NUMBER_ONLY_LABEL = re.compile(
    r'^(?:chapter|part|book|section)\s+'
    r'(?:\d+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten)$',
    re.IGNORECASE,
)


def _is_semantic_heading(tag) -> bool:
    if tag.name in {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}:
        return True
    role = str(tag.get('role', '')).lower()
    epub_type = str(tag.get('epub:type', '')).lower().split()
    classes = {
        re.sub(r'[^a-z0-9]+', '', name.lower())
        for name in tag.get('class', [])
    }
    return (
        'heading' in role
        or bool({'title', 'bridgehead'} & set(epub_type))
        or any(
            name in {
                'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'title', 'heading',
                'chaptertitle', 'chapterheading', 'sectiontitle',
            }
            for name in classes
        )
    )


def _looks_like_chapter_heading(text: str) -> bool:
    return bool(_CHAPTER_LABEL.fullmatch(text) or _STANDALONE_HEADING.fullmatch(text))


def _speech_heading_from_blocks(blocks: list[str]) -> str:
    if len(blocks) > 1 and _CHAPTER_NUMBER_ONLY_LABEL.fullmatch(blocks[0]):
        return f'{blocks[0]}. {_heading_for_speech(" ".join(blocks[1:]))}'
    if (
        len(blocks) > 2
        and re.fullmatch(r'(?:chapter|part|book|section)', blocks[0], re.IGNORECASE)
        and re.fullmatch(r'(?:\d+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten)', blocks[1], re.IGNORECASE)
    ):
        return (
            f'{blocks[0]} {blocks[1]}. '
            f'{_heading_for_speech(" ".join(blocks[2:]))}'
        )
    return _heading_for_speech(' '.join(blocks))


def _epub_toc_links(entries):
    for entry in entries:
        if isinstance(entry, tuple):
            section, children = entry
            if hasattr(section, 'href'):
                yield section
            yield from _epub_toc_links(children)
        elif hasattr(entry, 'href'):
            yield entry


def _epub_toc_boundaries(book, item, soup, block_entries):
    item_path = posixpath.normpath(unquote(item.get_name())).lstrip('./')
    block_indices = {block: index for index, (block, _) in enumerate(block_entries)}
    boundaries = {}
    for link in _epub_toc_links(book.toc):
        parsed_href = urlsplit(link.href)
        if not parsed_href.fragment:
            continue
        link_path = posixpath.normpath(unquote(parsed_href.path)).lstrip('./')
        if link_path != item_path:
            if posixpath.basename(link_path) != posixpath.basename(item_path):
                continue
        fragment = unquote(parsed_href.fragment)
        target = soup.find(id=fragment)
        if target is None:
            target = soup.find(attrs={'name': fragment})
        while target is not None and target not in block_indices:
            target = target.parent
        if target is None:
            continue
        block_index = block_indices[target]
        title = str(link.title or '').strip()
        if not block_entries[block_index][1] or not title:
            continue
        boundaries.setdefault(block_index, title)
    return boundaries


def _epub_document_blocks(soup):
    body = soup.body or soup
    block_names = [
        'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li',
        'blockquote', 'pre', 'dt', 'dd',
    ]
    candidate_nodes = [
        node for node in body.find_all(True)
        if node.name in block_names or _is_semantic_heading(node)
    ]
    candidate_set = set(candidate_nodes)
    block_entries = []
    for block in candidate_nodes:
        parent = block.parent
        nested_candidate = False
        while parent is not None and parent is not body:
            if parent in candidate_set:
                nested_candidate = True
                break
            parent = parent.parent
        if nested_candidate:
            continue
        text = block.get_text(' ', strip=True)
        if text:
            block_entries.append((block, text))
    return body, block_entries


def extract_chapters_from_epub(path: str) -> list[Chapter]:
    book = epub.read_epub(path)
    chapters = []
    documents = []
    for spine_entry in book.spine:
        item_id = spine_entry[0] if isinstance(spine_entry, tuple) else spine_entry
        item = book.get_item_with_id(item_id)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        if len(spine_entry) > 1 and spine_entry[1] == 'no':
            continue

        soup = BeautifulSoup(item.get_content(), 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'title']):
            tag.decompose()
        for tag in soup.find_all(True):
            role = tag.get('role', '').lower()
            epub_type = tag.get('epub:type', '').lower()
            classes = ' '.join(tag.get('class', [])).lower()
            if (
                role in {'doc-footnote', 'doc-endnote', 'doc-noteref'}
                or epub_type in {'footnote', 'endnote', 'noteref'}
                or any(note_type in classes for note_type in ('footnote', 'endnote'))
            ):
                tag.decompose()

        body, block_entries = _epub_document_blocks(soup)
        boundaries = _epub_toc_boundaries(book, item, soup, block_entries)
        documents.append((body, block_entries, boundaries))

    if any(boundaries for _, _, boundaries in documents):
        current_title = None
        current_body = []

        def finish_toc_chapter():
            if current_title is None:
                return
            speech_heading = _heading_for_speech(current_title)
            speech_body = '\n\n'.join(current_body)
            text = (
                f'{speech_heading}\n\n{speech_body}'
                if speech_body
                else speech_heading
            )
            chapters.append(Chapter(
                index=len(chapters),
                title=current_title,
                text=text,
                speech_heading=speech_heading,
                speech_body=speech_body,
            ))

        for _, block_entries, boundaries in documents:
            for block_index, (_, text) in enumerate(block_entries):
                if block_index in boundaries:
                    finish_toc_chapter()
                    current_title = boundaries[block_index]
                    current_body = []
                elif current_title is not None:
                    current_body.append(text)
        finish_toc_chapter()
        if not chapters:
            raise ValueError("EPUB table of contents has no readable chapter text.")
        return chapters

    for body, block_entries, _ in documents:
        text_blocks = []
        heading_blocks = []
        first_is_chapter_label = bool(
            block_entries and _looks_like_chapter_heading(block_entries[0][1])
        )
        heading_finished = False
        for block_index, (block, text) in enumerate(block_entries):
            is_heading = (
                _is_semantic_heading(block)
                or (block_index == 0 and first_is_chapter_label)
                or (
                    block_index == 1
                    and first_is_chapter_label
                    and len(text) <= 120
                    and not text.endswith(('.', '!', '?', '…'))
                )
            )
            if not heading_finished and is_heading:
                heading_blocks.append(text)
            else:
                heading_finished = True
                text_blocks.append(text)
        if not text_blocks:
            text = body.get_text(' ', strip=True)
            if text:
                text_blocks.append(text)

        speech_heading = _speech_heading_from_blocks(heading_blocks) if heading_blocks else None
        speech_body = '\n\n'.join(text_blocks) if heading_blocks else None
        if speech_heading and speech_body:
            text = f'{speech_heading}\n\n{speech_body}'
        elif speech_heading:
            text = speech_heading
        heading = next(
            (block for block, _ in block_entries if _is_semantic_heading(block)),
            body.find(['h1', 'h2', 'h3']),
        )
        title = (
            ' '.join(heading_blocks)
            if heading_blocks
            else heading.get_text(' ', strip=True) if heading
            else f"Section {len(chapters) + 1}"
        )
        if not heading_blocks:
            text = '\n\n'.join(text_blocks)
        if text:
            chapters.append(Chapter(
                index=len(chapters),
                title=title,
                text=text,
                speech_heading=speech_heading,
                speech_body=speech_body,
            ))
    if not book.spine:
        raise ValueError("EPUB has no reading-order spine.")
    return chapters


def extract_epub_title(path: str, fallback: str = '') -> str:
    book = epub.read_epub(path)
    title_metadata = book.get_metadata('DC', 'title')
    if title_metadata:
        title = title_metadata[0][0].strip()
        if title:
            return title
    return fallback.strip() or Path(path).stem


def clean_text(text: str) -> str:
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    paragraphs = (
        re.sub(r'\s+', ' ', paragraph).strip()
        for paragraph in re.split(r'\n\s*\n', text)
    )
    return '\n\n'.join(paragraph for paragraph in paragraphs if paragraph)


def split_text(text: str, max_length: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    if max_length < 1:
        raise ValueError("max_length must be greater than zero.")

    chunks, current = [], ''
    for paragraph_index, paragraph in enumerate(re.split(r'\n\s*\n', text)):
        sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())
        for sentence_index, sentence in enumerate(sentences):
            parts = []
            while len(sentence) > max_length:
                split_at = sentence.rfind(' ', 0, max_length + 1)
                if split_at < 1:
                    split_at = max_length
                parts.append(sentence[:split_at].strip())
                sentence = sentence[split_at:].strip()
            if sentence:
                parts.append(sentence)

            for part_index, part in enumerate(parts):
                if not current:
                    current = part
                    continue
                separator = (
                    '\n\n'
                    if paragraph_index > 0 and sentence_index == 0 and part_index == 0
                    else ' '
                )
                candidate = f"{current}{separator}{part}"
                if len(candidate) <= max_length:
                    current = candidate
                else:
                    chunks.append(current)
                    current = part
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

def blend_voice_styles(
    kokoro: Kokoro,
    primary_voice: str,
    secondary_voice: str,
    secondary_percent: float,
) -> np.ndarray:
    if not 0 <= secondary_percent <= 100:
        raise ValueError("The secondary voice percentage must be between 0 and 100.")
    primary_style = kokoro.get_voice_style(primary_voice)
    secondary_style = kokoro.get_voice_style(secondary_voice)
    ratio = secondary_percent / 100
    return np.asarray(
        primary_style * (1 - ratio) + secondary_style * ratio,
        dtype=np.float32,
    )


def apply_pronunciation_replacements(
    text: str,
    replacements: list[tuple[str, str]] | tuple[tuple[str, str], ...],
) -> str:
    for original, spoken_form in replacements:
        text = re.sub(
            rf'(?<!\w){re.escape(original)}(?!\w)',
            lambda _match, replacement=spoken_form: replacement,
            text,
            flags=re.IGNORECASE,
        )
    return text


def synthesize(
    kokoro: Kokoro,
    text: str,
    voice: str | np.ndarray,
    speed: float,
    lang: str,
    sentence_pause: float = DEFAULT_SENTENCE_PAUSE_SECONDS,
) -> tuple[np.ndarray, int]:
    samples, sample_rate = kokoro.create(
        text,
        voice=voice,
        speed=speed,
        lang=lang,
        sentence_pause=sentence_pause,
    )
    return samples, sample_rate


def play_audio(audio: np.ndarray, sample_rate: int, device: int | None = None):
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples[:, np.newaxis]
    elif samples.ndim != 2:
        raise ValueError("Audio must be a mono or multi-channel sample array.")
    if samples.shape[0] == 0:
        return

    interactive = sys.stdin.isatty()
    terminal_settings = None
    terminal_fd = None
    if interactive and os.name != 'nt':
        import termios
        import tty

        terminal_fd = sys.stdin.fileno()
        terminal_settings = termios.tcgetattr(terminal_fd)
        tty.setcbreak(terminal_fd)

    block_size = max(1, sample_rate // 25)
    silence = np.zeros((block_size, samples.shape[1]), dtype=np.float32)
    position = 0
    paused = False

    try:
        if interactive:
            print("[PLAYBACK] Press SPACE to pause or continue.")
        with sd.OutputStream(
            samplerate=sample_rate,
            channels=samples.shape[1],
            device=device,
        ) as stream:
            while position < len(samples):
                if interactive:
                    if os.name == 'nt':
                        import msvcrt

                        key = msvcrt.getwch() if msvcrt.kbhit() else None
                    else:
                        ready, _, _ = select.select([sys.stdin], [], [], 0)
                        key = sys.stdin.read(1) if ready else None
                    if key == ' ':
                        paused = not paused
                        print(
                            "[PLAYBACK] Paused. Press SPACE to continue."
                            if paused else "[PLAYBACK] Continuing."
                        )

                if paused:
                    stream.write(silence)
                else:
                    end = min(position + block_size, len(samples))
                    stream.write(samples[position:end])
                    position = end
    finally:
        if terminal_settings is not None and terminal_fd is not None:
            import termios

            termios.tcsetattr(terminal_fd, termios.TCSADRAIN, terminal_settings)


# ---------------------------------------------------------------------------
# Resume / progress helpers
# ---------------------------------------------------------------------------

def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()[:16]


def _sanitize(name: str) -> str:
    return re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')[:40]


def chapter_wav_path(save_base: str, chapter: Chapter, single: bool) -> Path:
    if single:
        return Path(save_base)
    base, ext = os.path.splitext(save_base)
    return Path(f"{base}_ch{chapter.index + 1:03d}_{_sanitize(chapter.title)}{ext or '.wav'}")


def normalize_wav_save_path(save_path: str) -> str:
    path = Path(save_path)
    if path.suffix and path.suffix.lower() != '.wav':
        raise ValueError("Audio export currently supports WAV files; use a .wav output path.")
    if not path.suffix:
        path = path.with_suffix('.wav')
    return str(path)


def chunk_cache_path(progress_dir: Path, chapter_idx: int, chunk_idx: int) -> Path:
    return progress_dir / f'ch{chapter_idx:03d}_chunk{chunk_idx:04d}.wav'


def generate_audio_files(
    chapters: list[Chapter],
    kokoro: Kokoro,
    voice: str | np.ndarray,
    speed: float,
    lang: str,
    output_base: str | None,
    progress_dir: Path,
    *,
    resume: bool = True,
    play: bool = False,
    device: int | None = None,
    limit: int | None = None,
    should_stop=lambda: False,
    wait_if_paused=lambda: True,
    on_progress=None,
    single_chapter_output: bool | None = None,
    heading_pause_seconds: float = HEADING_PAUSE_SECONDS,
    sentence_pause: float = DEFAULT_SENTENCE_PAUSE_SECONDS,
    pronunciation_replacements: list[tuple[str, str]] | tuple[tuple[str, str], ...] = (),
) -> list[Path]:
    chapter_texts = [
        apply_pronunciation_replacements(clean_text(chapter.text), pronunciation_replacements)
        for chapter in chapters
    ]
    chapter_chunks = []
    chapter_headings = []
    for chapter, text in zip(chapters, chapter_texts):
        speech_heading = (
            apply_pronunciation_replacements(
                chapter.speech_heading,
                pronunciation_replacements,
            )
            if chapter.speech_heading
            else None
        )
        chapter_headings.append(speech_heading)
        if speech_heading and chapter.speech_body is not None:
            body = clean_text(apply_pronunciation_replacements(
                chapter.speech_body,
                pronunciation_replacements,
            ))
            chunks = [speech_heading]
            if body:
                chunks.extend(split_text(body))
            chapter_chunks.append(chunks)
        elif speech_heading and text.startswith(speech_heading):
            body = text[len(speech_heading):].strip()
            chunks = [speech_heading]
            if body:
                chunks.extend(split_text(body))
            chapter_chunks.append(chunks)
        else:
            chapter_chunks.append(split_text(text) if text else [])
    total_chunks = sum(len(chunks) for chunks in chapter_chunks)
    completed_chunks = 0
    outputs: list[Path] = []
    single_chapter = (
        len([chunks for chunks in chapter_chunks if chunks]) == 1
        if single_chapter_output is None
        else single_chapter_output
    )

    def report(
        chapter: Chapter | None,
        chunk_index: int = 0,
        chapter_chunk_count: int = 0,
    ) -> None:
        if on_progress is not None:
            on_progress({
                'completed_chunks': completed_chunks,
                'total_chunks': total_chunks,
                'chapter_index': chapter.index + 1 if chapter else 0,
                'chapter_title': chapter.title if chapter else '',
                'chapter_chunk': chunk_index,
                'chapter_chunks': chapter_chunk_count,
            })

    if output_base:
        progress_dir.mkdir(parents=True, exist_ok=True)

    report(None)
    for chapter, text, speech_heading, all_chunks in zip(
        chapters,
        chapter_texts,
        chapter_headings,
        chapter_chunks,
    ):
        if not text:
            continue
        out_path = (
            chapter_wav_path(output_base, chapter, single_chapter)
            if output_base else None
        )
        if resume and out_path and out_path.exists():
            outputs.append(out_path)
            completed_chunks += len(all_chunks)
            report(chapter, len(all_chunks), len(all_chunks))
            continue

        chunks = all_chunks
        if limit is not None:
            chunks = chunks[:max(0, limit - completed_chunks)]
        if not chunks:
            if limit is not None and completed_chunks >= limit:
                break
            continue

        chapter_audio: list[np.ndarray] = []
        sample_rate: int | None = None
        for chunk_index, chunk in enumerate(chunks):
            if should_stop() or not wait_if_paused():
                return outputs
            cache_path = chunk_cache_path(progress_dir, chapter.index, chunk_index)
            if resume and cache_path.exists():
                audio, current_sample_rate = sf.read(str(cache_path), dtype='float32')
            else:
                try:
                    audio, current_sample_rate = synthesize(
                        kokoro, chunk, voice, speed, lang, sentence_pause
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Synthesis failed in chapter {chapter.index + 1}, "
                        f"chunk {chunk_index + 1}."
                    ) from exc
                if audio is None or current_sample_rate is None:
                    raise RuntimeError(
                        f"Synthesis returned no audio in chapter {chapter.index + 1}, "
                        f"chunk {chunk_index + 1}."
                    )
                if output_base:
                    sf.write(str(cache_path), audio, current_sample_rate)
                if play:
                    play_audio(audio, current_sample_rate, device=device)

            if sample_rate is not None and current_sample_rate != sample_rate:
                raise RuntimeError("Audio sample rate changed between chunks.")
            sample_rate = current_sample_rate
            audio_samples = np.asarray(audio, dtype=np.float32)
            if (
                chunk_index == 0
                and speech_heading
                and chunk == speech_heading
            ):
                silence_shape = (round(current_sample_rate * heading_pause_seconds),)
                if audio_samples.ndim > 1:
                    silence_shape += (audio_samples.shape[1],)
                pause = np.zeros(silence_shape, dtype=np.float32)
                audio_samples = np.concatenate((audio_samples, pause))
            chapter_audio.append(audio_samples)
            completed_chunks += 1
            report(chapter, chunk_index + 1, len(all_chunks))

        chapter_is_complete = (
            len(chunks) == len(all_chunks)
            and len(chapter_audio) == len(all_chunks)
        )
        if out_path and chapter_audio and chapter_is_complete:
            sf.write(str(out_path), np.concatenate(chapter_audio), sample_rate)
            outputs.append(out_path)
            for chunk_index in range(len(all_chunks)):
                cache_path = chunk_cache_path(
                    progress_dir, chapter.index, chunk_index
                )
                if cache_path.exists():
                    cache_path.unlink()

        if limit is not None and completed_chunks >= limit:
            break
    return outputs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EPUB/PDF TTS reader using Kokoro-ONNX")
    parser.add_argument('file', nargs='?', help='Input .epub or .pdf file')
    parser.add_argument('--list-voices', action='store_true',
                        help='List voice IDs available in the Kokoro model and exit')
    parser.add_argument('--voice', default=DEFAULT_VOICE,
                        help='Voice ID (default: af_bella). '
                             'Use --list-voices to see available options.')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Speech speed multiplier (default: 1.0)')
    parser.add_argument('--save', metavar='OUTPUT.wav',
                        help='Save WAV audio. One file per chapter: OUTPUT_ch001_Title.wav, etc.')
    parser.add_argument('--no-play', action='store_true',
                        help='Skip live playback')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last interrupted position (requires --save)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Stop after this many total chunks (for testing)')
    parser.add_argument('--device', type=int, default=None,
                        help='Output device index (see sounddevice.query_devices())')
    parser.add_argument(
        '--lang',
        choices=SUPPORTED_ENGLISH_LANGUAGES,
        default='en-us',
        help='English pronunciation: en-us or en-gb (default: en-us)',
    )
    args = parser.parse_args()

    if args.speed <= 0:
        parser.error("--speed must be greater than zero.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be greater than zero.")
    if args.save:
        try:
            args.save = normalize_wav_save_path(args.save)
        except ValueError as exc:
            parser.error(str(exc))

    if args.list_voices:
        kokoro = load_model()
        print("Available English Kokoro voices:")
        for voice in english_voices(kokoro.get_voices()):
            print(f"  {voice}")
        return

    if not args.file:
        parser.error("an input .epub or .pdf file is required unless --list-voices is used")

    if not os.path.isfile(args.file):
        print(f"[ERROR] File not found: {args.file}")
        sys.exit(1)

    if args.resume and not args.save:
        print("[ERROR] --resume requires --save to know where chapter files are.")
        sys.exit(1)

    file_ext = os.path.splitext(args.file)[1].lower()
    if file_ext == '.pdf':
        chapters = extract_chapters_from_pdf(args.file)
    elif file_ext == '.epub':
        chapters = extract_chapters_from_epub(args.file)
    else:
        print("[ERROR] Unsupported file type. Use .pdf or .epub.")
        sys.exit(1)

    if not chapters:
        print("[ERROR] No content extracted.")
        sys.exit(1)

    print(f"[INFO] Found {len(chapters)} chapter(s)")

    settings_key = hashlib.sha256(
        f"{TTS_PIPELINE_VERSION}\0{args.voice}\0{args.speed:g}\0{args.lang}".encode()
    ).hexdigest()[:12]
    progress_dir = _PROGRESS_ROOT / file_hash(args.file) / settings_key

    kokoro = load_model()
    if args.voice not in english_voices(kokoro.get_voices()):
        parser.error("--voice must be an available American or British English voice.")

    def report_progress(progress: dict) -> None:
        if progress['chapter_chunk']:
            print(
                f"[INFO] Chapter {progress['chapter_index']}: "
                f"chunk {progress['chapter_chunk']}/{progress['chapter_chunks']} "
                f"({progress['completed_chunks']}/{progress['total_chunks']} total)"
            )

    outputs = generate_audio_files(
        chapters,
        kokoro,
        args.voice,
        args.speed,
        args.lang,
        args.save,
        progress_dir,
        resume=args.resume,
        play=not args.no_play,
        device=args.device,
        limit=args.limit,
        on_progress=report_progress,
    )
    for output_path in outputs:
        print(f"[INFO] Saved: {output_path}")


if __name__ == '__main__':
    main()
