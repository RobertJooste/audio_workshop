#!/usr/bin/env python3
"""Local web interface for generating EPUB audiobooks with Kokoro."""

import argparse
import hashlib
import json
import logging
import math
import re
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_from_directory
import numpy as np
from werkzeug.exceptions import NotFound

import epub_pdf_tts_reader as reader

logger = logging.getLogger(__name__)
_ACTIVE_STATES = {'queued', 'loading_model', 'running', 'paused', 'stopping'}
_MAX_UPLOAD_BYTES = 100 * 1024 * 1024
_PREVIEW_TTL_SECONDS = 60 * 60
_JOB_TTL_SECONDS = 24 * 60 * 60
_VOICE_LANGUAGES = {
    'a': 'American English',
    'b': 'British English',
    'j': 'Japanese',
    'z': 'Mandarin Chinese',
    'e': 'Spanish',
    'f': 'French',
    'h': 'Hindi',
    'i': 'Italian',
    'p': 'Brazilian Portuguese',
}
_VOICE_TYPES = {'f': 'feminine voice', 'm': 'masculine voice'}
_ENGLISH_LANGUAGE_TAGS = {'a': 'en-us', 'b': 'en-gb'}


def english_voices(voices: list[str]) -> list[str]:
    return reader.english_voices(voices)


def language_for_voice(voice_id: str) -> str:
    try:
        return _ENGLISH_LANGUAGE_TAGS[voice_id[0]]
    except (IndexError, KeyError) as exc:
        raise ValueError('Only American and British English voices are supported.') from exc


def _validated_number(value: object, label: str, minimum: float, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f'{label} must be between {minimum:g} and {maximum:g}.')
    return float(value)


def _validated_replacements(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list) or len(value) > 50:
        raise ValueError('Provide no more than 50 pronunciation replacements.')
    replacements = []
    seen = set()
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError('Each pronunciation replacement needs a word and spoken form.')
        original = entry.get('find')
        spoken_form = entry.get('replace')
        if not isinstance(original, str) or not isinstance(spoken_form, str):
            raise ValueError('Each pronunciation replacement needs a word and spoken form.')
        original = original.strip()
        spoken_form = spoken_form.strip()
        if (
            not original
            or not spoken_form
            or len(original) > 200
            or len(spoken_form) > 200
        ):
            raise ValueError('Replacement words must be 1–200 characters long.')
        key = original.casefold()
        if key in seen:
            raise ValueError(f'Duplicate pronunciation replacement: {original}.')
        seen.add(key)
        replacements.append((original, spoken_form))
    return tuple(replacements)


def describe_voice(voice_id: str) -> dict[str, str]:
    language = _VOICE_LANGUAGES.get(voice_id[:1])
    voice_type = _VOICE_TYPES.get(voice_id[1:2])
    name = voice_id[3:].replace('_', ' ').title() if len(voice_id) > 2 else ''
    details = [detail for detail in (language, voice_type) if detail]
    if name:
        label = f"{name} — {' · '.join(details)}" if details else name
    else:
        label = ' · '.join(details) if details else voice_id
    return {
        'id': voice_id,
        'name': name,
        'language': language or 'Language unspecified',
        'voice_type': voice_type or 'Voice type unspecified',
        'label': label,
    }


def folder_component(value: str, fallback: str) -> str:
    component = re.sub(r'[^\w .()-]+', '_', value, flags=re.UNICODE)
    component = re.sub(r'\s+', ' ', component).strip(' ._-')
    return component or fallback


def voice_folder_name(voice_id: str) -> str:
    voice = describe_voice(voice_id)
    if voice['name']:
        return folder_component(f"{voice['name']} - {voice_id}", voice_id)
    return folder_component(f"{voice['label']} ({voice_id})", voice_id)


@dataclass
class Job:
    job_id: str
    upload_path: Path
    output_dir: Path
    voice: str
    book_title: str
    chapter_indices: tuple[int, ...]
    chapters_total: int
    settings_key: str
    source_hash: str
    secondary_voice: str | None = None
    secondary_percent: float = 50
    speed: float = 1.0
    heading_pause_seconds: float = reader.HEADING_PAUSE_SECONDS
    sentence_pause: float = reader.DEFAULT_SENTENCE_PAUSE_SECONDS
    pronunciation_replacements: tuple[tuple[str, str], ...] = ()
    state: str = 'queued'
    phase: str = 'Queued'
    chapter_index: int = 0
    chapter_title: str = ''
    completed_chunks: int = 0
    total_chunks: int = 0
    outputs: list[str] = field(default_factory=list)
    error: str | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    resume_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.resume_event.set()


@dataclass
class BookPreview:
    upload_path: Path
    title: str
    chapters: list[reader.Chapter]
    created_at: float = field(default_factory=time.time)


class JobManager:
    def __init__(
        self,
        storage_root: Path,
        output_root: Path,
        model=None,
    ):
        self.storage_root = storage_root
        self.jobs_root = storage_root / 'jobs'
        self.progress_root = storage_root / 'progress'
        self.uploads_root = storage_root / 'uploads'
        self.output_root = output_root
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.progress_root.mkdir(parents=True, exist_ok=True)
        self.uploads_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.lock = threading.RLock()
        self.model_lock = threading.Lock()
        self.jobs: dict[str, Job] = {}
        self.books: dict[str, BookPreview] = {}
        self._last_cleanup: float = 0.0

    @staticmethod
    def _remove_upload(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove expired EPUB upload %s", path, exc_info=True)

    def _remove_upload_if_unused(self, path: Path) -> None:
        with self.lock:
            preview_uses_upload = any(
                book.upload_path == path for book in self.books.values()
            )
            job_needs_upload = any(
                job.upload_path == path and job.state != 'completed'
                for job in self.jobs.values()
            )
            if not preview_uses_upload and not job_needs_upload:
                self._remove_upload(path)

    def cleanup_expired(self, now: float | None = None) -> None:
        current_time = time.time() if now is None else now
        if now is None and current_time - self._last_cleanup < 60:
            return
        self._last_cleanup = current_time
        expired_uploads = []
        with self.lock:
            expired_book_ids = [
                book_id
                for book_id, book in self.books.items()
                if current_time - book.created_at > _PREVIEW_TTL_SECONDS
            ]
            for book_id in expired_book_ids:
                expired_uploads.append(self.books.pop(book_id).upload_path)

            for job in self.jobs.values():
                if (
                    job.state == 'paused'
                    and current_time - job.updated_at > _JOB_TTL_SECONDS
                ):
                    job.stop_event.set()
                    job.resume_event.set()
                    job.state = 'stopping'
                    job.phase = 'Expired paused job; stopping'
                    job.updated_at = current_time

            expired_job_ids = [
                job_id
                for job_id, job in self.jobs.items()
                if (
                    job.state not in _ACTIVE_STATES
                    and current_time - job.updated_at > _JOB_TTL_SECONDS
                )
            ]
            for job_id in expired_job_ids:
                expired_uploads.append(self.jobs.pop(job_id).upload_path)
        for upload_path in expired_uploads:
            self._remove_upload_if_unused(upload_path)

    def get_model(self):
        with self.model_lock:
            if self.model is None:
                self.model = reader.load_model()
            return self.model

    def preview_book(
        self,
        upload,
        original_filename: str,
    ) -> tuple[str, str, list[reader.Chapter]]:
        book_id = uuid.uuid4().hex
        upload_path = self.uploads_root / f'{book_id}.epub'
        upload.save(upload_path)
        if upload_path.stat().st_size == 0:
            upload_path.unlink()
            raise ValueError("The uploaded EPUB is empty.")
        with upload_path.open('rb') as f:
            if f.read(4) != b'PK\x03\x04':
                upload_path.unlink()
                raise ValueError("The uploaded file is not a valid EPUB.")
        try:
            title = reader.extract_epub_title(
                str(upload_path),
                fallback=Path(original_filename).stem,
            )
            chapters = reader.extract_chapters_from_epub(str(upload_path))
        except Exception as exc:
            upload_path.unlink(missing_ok=True)
            raise ValueError("Could not read this EPUB. Check that the file is valid.") from exc
        readable = [chapter for chapter in chapters if reader.clean_text(chapter.text)]
        if not readable:
            upload_path.unlink(missing_ok=True)
            raise ValueError("No readable chapters were found in the uploaded EPUB.")
        with self.lock:
            self.books[book_id] = BookPreview(upload_path, title, readable)
        return book_id, title, readable

    def create_job(
        self,
        book_id: str,
        voice: str,
        selected_chapter_indices: list[int],
        *,
        secondary_voice: str | None = None,
        secondary_percent: float = 50,
        speed: float = 1.0,
        heading_pause_seconds: float = reader.HEADING_PAUSE_SECONDS,
        sentence_pause: float = reader.DEFAULT_SENTENCE_PAUSE_SECONDS,
        pronunciation_replacements: tuple[tuple[str, str], ...] = (),
    ) -> Job:
        with self.lock:
            if any(job.state in _ACTIVE_STATES for job in self.jobs.values()):
                raise ValueError("A book is already being processed.")
            book = self.books.get(book_id)
            if book is None:
                raise ValueError("Upload and preview the EPUB before starting.")
            upload_path, title, chapters = (
                book.upload_path,
                book.title,
                book.chapters,
            )
            available_indices = {chapter.index for chapter in chapters}
            if (
                not selected_chapter_indices
                or any(type(index) is not int for index in selected_chapter_indices)
                or len(set(selected_chapter_indices)) != len(selected_chapter_indices)
                or not set(selected_chapter_indices).issubset(available_indices)
            ):
                raise ValueError("Select one or more valid chapters.")
            if secondary_voice and secondary_voice == voice:
                raise ValueError("The blend voice must be different from the primary voice.")

            job_id = uuid.uuid4().hex
            source_hash = reader.file_hash(str(upload_path))
            book_dir_name = folder_component(title, 'Untitled Book')
            voice_dir_name = voice_folder_name(voice)
            if secondary_voice:
                primary_percent = 100 - secondary_percent
                primary_name = describe_voice(voice)['name'] or voice
                secondary_name = describe_voice(secondary_voice)['name'] or secondary_voice
                voice_dir_name = folder_component(
                    f'{primary_name} {primary_percent:g} percent and '
                    f'{secondary_name} {secondary_percent:g} percent',
                    voice,
                )
            settings = {
                'pipeline': reader.TTS_PIPELINE_VERSION,
                'voice': voice,
                'secondary_voice': secondary_voice,
                'secondary_percent': secondary_percent if secondary_voice else 0,
                'speed': speed,
                'heading_pause_seconds': heading_pause_seconds,
                'sentence_pause': sentence_pause,
                'pronunciation_replacements': pronunciation_replacements,
            }
            settings_key = hashlib.sha256(
                json.dumps(settings, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()[:12]
            output_dir = (
                self.output_root
                / book_dir_name
                / voice_dir_name
                / f'Book {source_hash}'
                / f'Settings {settings_key}'
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            job = Job(
                job_id=job_id,
                upload_path=upload_path,
                output_dir=output_dir,
                voice=voice,
                book_title=title,
                chapter_indices=tuple(selected_chapter_indices),
                chapters_total=len(chapters),
                settings_key=settings_key,
                source_hash=source_hash,
                secondary_voice=secondary_voice,
                secondary_percent=secondary_percent,
                speed=speed,
                heading_pause_seconds=heading_pause_seconds,
                sentence_pause=sentence_pause,
                pronunciation_replacements=pronunciation_replacements,
            )
            self.jobs[job_id] = job
            self._start_locked(job)
            return job

    def _start_locked(self, job: Job) -> None:
        job.stop_event.clear()
        job.resume_event.set()
        job.state = 'queued'
        job.phase = 'Queued'
        job.error = None
        job.outputs = []
        job.updated_at = time.time()
        job.thread = threading.Thread(
            target=self._run_job,
            args=(job,),
            name=f'epub-job-{job.job_id[:8]}',
            daemon=True,
        )
        job.thread.start()

    def _run_job(self, job: Job) -> None:
        try:
            with self.lock:
                job.state = 'loading_model'
                job.phase = 'Loading speech model'
            model = self.get_model()
            available_voices = english_voices(model.get_voices())
            if job.voice not in available_voices:
                raise ValueError("The selected voice is not an available English voice.")
            if job.secondary_voice and job.secondary_voice not in available_voices:
                raise ValueError("The selected blend voice is not available in the Kokoro model.")

            chapters = reader.extract_chapters_from_epub(str(job.upload_path))
            chapters = [
                chapter for chapter in chapters
                if chapter.index in job.chapter_indices
            ]
            progress_dir = (
                self.progress_root
                / job.source_hash
                / job.settings_key
            )
            voice_style: str | np.ndarray = job.voice
            if job.secondary_voice:
                voice_style = reader.blend_voice_styles(
                    model,
                    job.voice,
                    job.secondary_voice,
                    job.secondary_percent,
                )

            def wait_if_paused() -> bool:
                while not job.stop_event.is_set():
                    if job.resume_event.wait(timeout=0.2):
                        return not job.stop_event.is_set()
                return False

            def on_progress(progress: dict) -> None:
                with self.lock:
                    job.completed_chunks = progress['completed_chunks']
                    job.total_chunks = progress['total_chunks']
                    job.chapter_index = progress['chapter_index']
                    job.chapter_title = progress['chapter_title']
                    if job.state not in {'paused', 'stopping'}:
                        job.state = 'running'
                        job.phase = 'Generating audio'

            with self.lock:
                if job.stop_event.is_set():
                    job.state = 'stopped'
                    job.phase = 'Stopped'
                    return
                job.state = 'running'
                job.phase = 'Generating audio'

            outputs = reader.generate_audio_files(
                chapters,
                model,
                voice_style,
                job.speed,
                language_for_voice(job.voice),
                str(job.output_dir / 'audiobook.wav'),
                progress_dir,
                resume=True,
                play=False,
                should_stop=job.stop_event.is_set,
                wait_if_paused=wait_if_paused,
                on_progress=on_progress,
                single_chapter_output=job.chapters_total == 1,
                heading_pause_seconds=job.heading_pause_seconds,
                sentence_pause=job.sentence_pause,
                pronunciation_replacements=job.pronunciation_replacements,
            )
            with self.lock:
                job.outputs = [path.name for path in outputs]
                if job.stop_event.is_set():
                    job.state = 'stopped'
                    job.phase = 'Stopped; generated chunks are saved for resume'
                else:
                    job.state = 'completed'
                    job.phase = 'Complete'
                job.updated_at = time.time()
            if job.state == 'completed':
                self._remove_upload_if_unused(job.upload_path)
        except Exception as exc:
            logger.exception("EPUB generation job %s failed", job.job_id)
            with self.lock:
                job.state = 'failed'
                job.phase = 'Failed'
                job.error = str(exc)
                job.updated_at = time.time()

    def pause(self, job: Job) -> None:
        with self.lock:
            if job.state != 'running':
                raise ValueError("Only a running job can be paused.")
            job.resume_event.clear()
            job.state = 'paused'
            job.phase = 'Paused'
            job.updated_at = time.time()

    def resume(self, job: Job) -> None:
        with self.lock:
            if job.state == 'paused':
                job.resume_event.set()
                job.state = 'running'
                job.phase = 'Generating audio'
                job.updated_at = time.time()
            elif job.state in {'stopped', 'failed'}:
                self._start_locked(job)
            else:
                raise ValueError("This job cannot be resumed in its current state.")

    def stop(self, job: Job) -> None:
        with self.lock:
            if job.state not in _ACTIVE_STATES:
                raise ValueError("This job is not currently running.")
            job.stop_event.set()
            job.resume_event.set()
            job.state = 'stopping'
            job.phase = 'Stopping after the current chunk'
            job.updated_at = time.time()

    def get_job(self, job_id: str) -> Job | None:
        with self.lock:
            return self.jobs.get(job_id)

    def serialize(self, job: Job) -> dict:
        with self.lock:
            percentage = (
                round(100 * job.completed_chunks / job.total_chunks)
                if job.total_chunks else 0
            )
            if job.state == 'completed':
                percentage = 100
            return {
                'id': job.job_id,
                'state': job.state,
                'phase': job.phase,
                'voice': job.voice,
                'secondary_voice': job.secondary_voice,
                'secondary_percent': job.secondary_percent,
                'speed': job.speed,
                'heading_pause_seconds': job.heading_pause_seconds,
                'sentence_pause': job.sentence_pause,
                'pronunciation_replacements': [
                    {'find': original, 'replace': spoken}
                    for original, spoken in job.pronunciation_replacements
                ],
                'book_title': job.book_title,
                'output_directory': str(job.output_dir),
                'selected_chapters': list(job.chapter_indices),
                'chapter_index': job.chapter_index,
                'chapter_title': job.chapter_title,
                'chapters_total': job.chapters_total,
                'completed_chunks': job.completed_chunks,
                'total_chunks': job.total_chunks,
                'percentage': percentage,
                'outputs': job.outputs,
                'error': job.error,
            }


def create_app(
    model=None,
    storage_root: Path | None = None,
    output_root: Path | None = None,
    desktop_shutdown_token: str | None = None,
) -> Flask:
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = _MAX_UPLOAD_BYTES
    app.config['DESKTOP_SHUTDOWN_TOKEN'] = desktop_shutdown_token
    effective_storage_root = storage_root or Path.home() / '.cache' / 'epub-tts-web'
    manager = JobManager(
        effective_storage_root,
        output_root or (
            Path(__file__).resolve().parent / 'processed'
            if storage_root is None
            else effective_storage_root / 'processed'
        ),
        model=model,
    )
    app.extensions['job_manager'] = manager

    @app.before_request
    def cleanup_expired_uploads() -> None:
        manager.cleanup_expired()

    @app.get('/')
    def index():
        return render_template(
            'index.html',
            desktop_shutdown_token=desktop_shutdown_token,
        )

    @app.get('/api/voices')
    def list_voices():
        try:
            voices = english_voices(manager.get_model().get_voices())
        except Exception:
            logger.exception("Could not load Kokoro voices")
            return jsonify({'error': 'Could not load the Kokoro voice model.'}), 500
        return jsonify({
            'voices': voices,
            'voice_options': [describe_voice(voice) for voice in voices],
            'default': reader.DEFAULT_VOICE,
        })

    @app.post('/api/jobs')
    def create_job():
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({'error': 'Invalid audiobook settings.'}), 400
        book_id = data.get('book_id')
        voice = data.get('voice', '')
        secondary_voice = data.get('secondary_voice', '')
        chapter_indices = data.get('chapter_indices')
        if not isinstance(book_id, str):
            return jsonify({'error': 'Upload and preview an EPUB first.'}), 400
        if not isinstance(voice, str):
            return jsonify({'error': 'Choose a voice from the available voice list.'}), 400
        if not isinstance(chapter_indices, list):
            return jsonify({'error': 'Select one or more chapters.'}), 400
        try:
            speed = _validated_number(data.get('speed', 1.0), 'Speed', 0.7, 1.3)
            heading_pause_seconds = _validated_number(
                data.get('heading_pause_seconds', reader.HEADING_PAUSE_SECONDS),
                'Heading pause',
                0,
                3,
            )
            sentence_pause = _validated_number(
                data.get('sentence_pause', reader.DEFAULT_SENTENCE_PAUSE_SECONDS),
                'Sentence pause',
                0,
                1,
            )
            secondary_percent = _validated_number(
                data.get('secondary_percent', 50),
                'Blend percentage',
                0,
                100,
            )
            if not isinstance(secondary_voice, str):
                raise ValueError('Choose a valid blend voice.')
            replacements = _validated_replacements(
                data.get('pronunciation_replacements', [])
            )
            model = manager.get_model()
            available_voices = english_voices(model.get_voices())
            if voice not in available_voices:
                return jsonify({'error': 'Choose an available English voice.'}), 400
            if secondary_voice and secondary_voice not in available_voices:
                return jsonify({'error': 'Choose a different available voice to blend.'}), 400
            job = manager.create_job(
                book_id,
                voice,
                chapter_indices,
                secondary_voice=secondary_voice or None,
                secondary_percent=secondary_percent,
                speed=speed,
                heading_pause_seconds=heading_pause_seconds,
                sentence_pause=sentence_pause,
                pronunciation_replacements=replacements,
            )
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400
        except Exception:
            logger.exception("Could not start EPUB generation")
            return jsonify({'error': 'Could not start the audiobook job.'}), 500
        return jsonify(manager.serialize(job)), 202

    @app.post('/api/books')
    def preview_book():
        upload = request.files.get('book')
        if upload is None or not upload.filename:
            return jsonify({'error': 'Choose an EPUB file to upload.'}), 400
        if Path(upload.filename).suffix.lower() != '.epub':
            return jsonify({'error': 'Only .epub books are supported by the web app.'}), 400
        try:
            book_id, title, chapters = manager.preview_book(upload, upload.filename)
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400
        return jsonify({
            'book_id': book_id,
            'title': title,
            'chapters': [
                {'index': chapter.index, 'title': chapter.title}
                for chapter in chapters
            ],
        })

    def find_job(job_id: str) -> Job:
        job = manager.get_job(job_id)
        if job is None:
            raise NotFound()
        return job

    @app.get('/api/jobs/<job_id>')
    def get_job_status(job_id: str):
        return jsonify(manager.serialize(find_job(job_id)))

    @app.post('/api/jobs/<job_id>/pause')
    def pause_job(job_id: str):
        job = find_job(job_id)
        try:
            manager.pause(job)
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 409
        return jsonify(manager.serialize(job))

    @app.post('/api/jobs/<job_id>/resume')
    def resume_job(job_id: str):
        job = find_job(job_id)
        try:
            manager.resume(job)
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 409
        return jsonify(manager.serialize(job))

    @app.post('/api/jobs/<job_id>/stop')
    def stop_job(job_id: str):
        job = find_job(job_id)
        try:
            manager.stop(job)
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 409
        return jsonify(manager.serialize(job))

    @app.get('/api/jobs/<job_id>/files/<filename>')
    def download_file(job_id: str, filename: str):
        job = find_job(job_id)
        if Path(filename).name != filename or filename not in job.outputs:
            abort(404)
        return send_from_directory(job.output_dir, filename, as_attachment=True)

    @app.post('/api/shutdown')
    def shutdown_desktop_app():
        expected_token = app.config['DESKTOP_SHUTDOWN_TOKEN']
        if expected_token is None:
            abort(404)
        provided_token = request.headers.get('X-Desktop-Shutdown-Token', '')
        if not secrets.compare_digest(provided_token, expected_token):
            abort(403)
        shutdown = app.config.get('DESKTOP_SHUTDOWN_CALLBACK')
        if not callable(shutdown):
            return jsonify({'error': 'The desktop app cannot be shut down right now.'}), 503
        threading.Thread(target=shutdown, daemon=True).start()
        return jsonify({'message': 'The desktop app is shutting down.'})

    @app.errorhandler(413)
    def upload_too_large(error):
        return jsonify({'error': 'The EPUB exceeds the 100 MB upload limit.'}), int(error.code or 413)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Local EPUB audiobook web app")
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    print("Loading speech model...")
    model = reader.load_model()
    app = create_app(model=model)
    print(f"Open http://127.0.0.1:{args.port} in your browser.")
    app.run(host='127.0.0.1', port=args.port, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()
