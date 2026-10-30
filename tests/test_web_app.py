import io
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from ebooklib import epub
import numpy as np
import soundfile as sf

import web_app
import epub_pdf_tts_reader as reader


class FakeModel:
    def get_voices(self):
        return ['af_bella', 'am_adam', 'bf_emma', 'jf_alpha']


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.epub_path = self.root / 'sample.epub'
        self._create_epub(self.epub_path)
        self.app = web_app.create_app(
            model=FakeModel(),
            storage_root=self.root / 'app-data',
            output_root=self.root / 'Audiobooks',
        )
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    @staticmethod
    def _create_epub(path: Path) -> None:
        book = epub.EpubBook()
        book.set_identifier('web-test-book')
        book.set_title('Web Test Book')
        book.set_language('en')
        chapter = epub.EpubHtml(uid='chapter', file_name='chapter.xhtml', title='Chapter', lang='en')
        text = ' '.join(['This is a complete sentence for testing audio generation.'] * 80)
        chapter.set_content(f'<html><body><h1>Chapter</h1><p>{text}</p></body></html>')
        book.add_item(chapter)
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.toc = (chapter,)
        book.spine = ['nav', chapter]
        epub.write_epub(str(path), book)

    @staticmethod
    def _create_multi_chapter_epub(path: Path) -> None:
        book = epub.EpubBook()
        book.set_identifier('multi-chapter-web-test')
        book.set_title('Multi-Chapter Test')
        book.set_language('en')
        chapters = []
        for index in range(1, 4):
            chapter = epub.EpubHtml(
                uid=f'chapter-{index}',
                file_name=f'chapter-{index}.xhtml',
                title=f'Chapter {index}',
                lang='en',
            )
            chapter.set_content(
                f'<html><body><h1>Chapter {index}</h1>'
                f'<p>Text for chapter {index}. This chapter is selected.</p>'
                '</body></html>'
            )
            chapters.append(chapter)
            book.add_item(chapter)
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.toc = tuple(chapters)
        book.spine = ['nav', *chapters]
        epub.write_epub(str(path), book)

    def _preview(self, path=None, filename='sample.epub'):
        selected_path = path or self.epub_path
        return self.client.post(
            '/api/books',
            data={'book': (io.BytesIO(selected_path.read_bytes()), filename)},
            content_type='multipart/form-data',
        )

    def _upload(self):
        preview = self._preview().get_json()
        return self.client.post(
            '/api/jobs',
            json={
                'book_id': preview['book_id'],
                'voice': 'af_bella',
                'chapter_indices': [chapter['index'] for chapter in preview['chapters']],
            },
        )

    def _wait_for_state(self, job_id, expected_state):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            response = self.client.get(f'/api/jobs/{job_id}')
            job = response.get_json()
            if job['state'] == expected_state:
                return job
            if job['state'] == 'failed':
                self.fail(f"Job failed: {job['error']}")
            time.sleep(0.02)
        self.fail(f"Timed out waiting for job state {expected_state}.")

    def test_voice_list_and_upload_validation(self):
        voices = self.client.get('/api/voices')
        self.assertEqual(voices.get_json()['voices'], ['af_bella', 'am_adam', 'bf_emma'])
        self.assertEqual(
            [voice['label'] for voice in voices.get_json()['voice_options']],
            [
                'Bella — American English · feminine voice',
                'Adam — American English · masculine voice',
                'Emma — British English · feminine voice',
            ],
        )
        self.assertNotIn('jf_alpha', voices.get_json()['voices'])
        self.assertEqual(web_app.describe_voice('bf_emma')['label'],
                         'Emma — British English · feminine voice')
        self.assertEqual(web_app.describe_voice('af_heart')['label'],
                         'Heart — American English · feminine voice')
        self.assertEqual(web_app.describe_voice('af')['label'],
                         'American English · feminine voice')

        invalid = self.client.post(
            '/api/books',
            data={
                'book': (io.BytesIO(b'not an epub'), 'book.pdf'),
            },
            content_type='multipart/form-data',
        )
        self.assertEqual(invalid.status_code, 400)

    def test_selected_chapters_only_are_synthesized_and_downloaded(self):
        multi_epub_path = self.root / 'multi.epub'
        self._create_multi_chapter_epub(multi_epub_path)
        preview_response = self._preview(multi_epub_path, 'multi.epub')
        self.assertEqual(preview_response.status_code, 200)
        preview = preview_response.get_json()
        self.assertEqual(preview['title'], 'Multi-Chapter Test')
        self.assertEqual(
            [chapter['title'] for chapter in preview['chapters']],
            ['Chapter 1', 'Chapter 2', 'Chapter 3'],
        )

        synthesis_calls = []

        def fake_synthesize(_model, text, *_args):
            synthesis_calls.append(text)
            return np.ones(12, dtype=np.float32), 24000

        with patch.object(reader, 'synthesize', side_effect=fake_synthesize):
            response = self.client.post(
                '/api/jobs',
                json={
                    'book_id': preview['book_id'],
                    'voice': 'af_bella',
                    'chapter_indices': [0, 2],
                },
            )
            self.assertEqual(response.status_code, 202)
            job_id = response.get_json()['id']
            completed = self._wait_for_state(job_id, 'completed')

        self.assertEqual(completed['selected_chapters'], [0, 2])
        self.assertEqual(completed['chapters_total'], 3)
        self.assertEqual(len(completed['outputs']), 2)
        self.assertTrue(any('Chapter_1' in name for name in completed['outputs']))
        self.assertTrue(any('Chapter_3' in name for name in completed['outputs']))
        self.assertTrue(any('Text for chapter 1' in text for text in synthesis_calls))
        self.assertTrue(any('Text for chapter 3' in text for text in synthesis_calls))
        self.assertFalse(any('Text for chapter 2' in text for text in synthesis_calls))
        expected_output_dir = (
            self.root
            / 'Audiobooks'
            / 'Multi-Chapter Test'
            / 'Bella - af_bella'
            / f"Book {self.app.extensions['job_manager'].jobs[job_id].source_hash}"
            / f"Settings {self.app.extensions['job_manager'].jobs[job_id].settings_key}"
        )
        self.assertEqual(Path(completed['output_directory']), expected_output_dir)
        self.assertTrue(all((expected_output_dir / name).is_file()
                            for name in completed['outputs']))
        completed_job = self.app.extensions['job_manager'].jobs[job_id]
        self.assertTrue(completed_job.upload_path.exists())
        self.assertIn(preview['book_id'], self.app.extensions['job_manager'].books)

        with patch.object(reader, 'synthesize', side_effect=fake_synthesize):
            single_chapter_response = self.client.post(
                '/api/jobs',
                json={
                    'book_id': preview['book_id'],
                    'voice': 'af_bella',
                    'chapter_indices': [1],
                },
            )
            self.assertEqual(single_chapter_response.status_code, 202)
            single_chapter = self._wait_for_state(
                single_chapter_response.get_json()['id'],
                'completed',
            )
        self.assertEqual(single_chapter['outputs'], ['audiobook_ch002_Chapter_2.wav'])
        self.assertTrue((expected_output_dir / single_chapter['outputs'][0]).is_file())

    def test_same_title_books_get_distinct_output_directories(self):
        second_path = self.root / 'second.epub'
        book = epub.EpubBook()
        book.set_identifier('different-content-same-title')
        book.set_title('Web Test Book')
        book.set_language('en')
        chapter = epub.EpubHtml(uid='chapter', file_name='chapter.xhtml', title='Chapter', lang='en')
        chapter.set_content('<html><body><h1>Chapter</h1><p>Different source content.</p></body></html>')
        book.add_item(chapter)
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.toc = (chapter,)
        book.spine = ['nav', chapter]
        epub.write_epub(str(second_path), book)

        synthesis_calls = []

        def fake_synthesize(_model, text, *_args):
            synthesis_calls.append(text)
            return np.ones(12, dtype=np.float32), 24000

        with patch.object(reader, 'synthesize', side_effect=fake_synthesize):
            first = self._upload()
            self.assertEqual(first.status_code, 202)
            first_job = self._wait_for_state(first.get_json()['id'], 'completed')
            second_preview = self._preview(second_path, 'second.epub').get_json()
            second = self.client.post(
                '/api/jobs',
                json={
                    'book_id': second_preview['book_id'],
                    'voice': 'af_bella',
                    'chapter_indices': [0],
                },
            )
            self.assertEqual(second.status_code, 202)
            second_job = self._wait_for_state(second.get_json()['id'], 'completed')

        self.assertNotEqual(first_job['output_directory'], second_job['output_directory'])
        self.assertTrue(any('Different source content' in text for text in synthesis_calls))

    def test_job_rejects_empty_or_out_of_range_chapter_selection(self):
        preview = self._preview().get_json()
        for indices in ([], [10], [0, 0], ['0']):
            with self.subTest(indices=indices):
                response = self.client.post(
                    '/api/jobs',
                    json={
                        'book_id': preview['book_id'],
                        'voice': 'af_bella',
                        'chapter_indices': indices,
                    },
                )
                self.assertEqual(response.status_code, 400)

    def test_custom_voice_speed_pauses_and_pronunciations_reach_synthesis(self):
        preview = self._preview().get_json()
        blend_style = np.full((2, 1, 3), 0.5, dtype=np.float32)
        synthesis_calls = []

        def fake_synthesize(_model, text, voice, speed, lang, sentence_pause):
            synthesis_calls.append((text, voice, speed, lang, sentence_pause))
            return np.ones(12, dtype=np.float32), 24000

        with (
            patch.object(reader, 'blend_voice_styles', return_value=blend_style) as blend,
            patch.object(reader, 'synthesize', side_effect=fake_synthesize),
        ):
            response = self.client.post(
                '/api/jobs',
                json={
                    'book_id': preview['book_id'],
                    'voice': 'af_bella',
                    'secondary_voice': 'am_adam',
                    'secondary_percent': 35,
                    'speed': 0.9,
                    'heading_pause_seconds': 0.8,
                    'sentence_pause': 0.4,
                    'pronunciation_replacements': [
                        {'find': 'testing', 'replace': 'checking'},
                    ],
                    'chapter_indices': [0],
                },
            )
            self.assertEqual(response.status_code, 202)
            job_id = response.get_json()['id']
            completed = self._wait_for_state(job_id, 'completed')

        blend.assert_called_once_with(
            self.app.extensions['job_manager'].model,
            'af_bella',
            'am_adam',
            35.0,
        )
        self.assertTrue(synthesis_calls)
        self.assertTrue(all(call[1] is blend_style for call in synthesis_calls))
        self.assertTrue(all(call[2:] == (0.9, 'en-us', 0.4) for call in synthesis_calls))
        self.assertTrue(any('checking' in call[0] for call in synthesis_calls))
        self.assertEqual(completed['secondary_voice'], 'am_adam')
        self.assertEqual(completed['secondary_percent'], 35)
        self.assertEqual(completed['speed'], 0.9)
        self.assertEqual(completed['heading_pause_seconds'], 0.8)
        self.assertEqual(completed['sentence_pause'], 0.4)
        self.assertEqual(
            completed['pronunciation_replacements'],
            [{'find': 'testing', 'replace': 'checking'}],
        )
        self.assertIn('Settings ', completed['output_directory'])
        output_path = Path(completed['output_directory']) / completed['outputs'][0]
        self.assertEqual(
            sf.info(str(output_path)).frames,
            12 * len(synthesis_calls) + round(24000 * 0.8),
        )

    def test_job_rejects_invalid_customization_settings(self):
        preview = self._preview().get_json()
        invalid_settings = (
            {'speed': 1.4},
            {'heading_pause_seconds': -0.1},
            {'sentence_pause': 1.1},
            {'secondary_voice': 'af_bella'},
            {'pronunciation_replacements': [{'find': 'word', 'replace': ''}]},
            {'pronunciation_replacements': [
                {'find': 'word', 'replace': 'one'},
                {'find': 'WORD', 'replace': 'two'},
            ]},
        )
        for settings in invalid_settings:
            with self.subTest(settings=settings):
                response = self.client.post(
                    '/api/jobs',
                    json={
                        'book_id': preview['book_id'],
                        'voice': 'af_bella',
                        'chapter_indices': [0],
                        **settings,
                    },
                )
                self.assertEqual(response.status_code, 400)

    def test_job_rejects_non_english_voices_and_uses_british_english_for_british_voice(self):
        preview = self._preview().get_json()
        rejected = self.client.post(
            '/api/jobs',
            json={
                'book_id': preview['book_id'],
                'voice': 'jf_alpha',
                'chapter_indices': [0],
            },
        )
        self.assertEqual(rejected.status_code, 400)

        synthesis_calls = []

        def fake_synthesize(_model, text, voice, speed, lang, sentence_pause):
            synthesis_calls.append(lang)
            return np.ones(12, dtype=np.float32), 24000

        with patch.object(reader, 'synthesize', side_effect=fake_synthesize):
            response = self.client.post(
                '/api/jobs',
                json={
                    'book_id': preview['book_id'],
                    'voice': 'bf_emma',
                    'chapter_indices': [0],
                },
            )
            self.assertEqual(response.status_code, 202)
            self._wait_for_state(response.get_json()['id'], 'completed')

        self.assertTrue(synthesis_calls)
        self.assertEqual(set(synthesis_calls), {'en-gb'})

    def test_expired_preview_uploads_are_removed(self):
        preview = self._preview().get_json()
        manager = self.app.extensions['job_manager']
        book = manager.books[preview['book_id']]
        self.assertTrue(book.upload_path.is_file())

        manager.cleanup_expired(now=book.created_at + web_app._PREVIEW_TTL_SECONDS + 1)

        self.assertNotIn(preview['book_id'], manager.books)
        self.assertFalse(book.upload_path.exists())

    def test_stop_keeps_chunks_and_resume_completes_downloadable_wav(self):
        first_synthesis_started = threading.Event()
        release_first_synthesis = threading.Event()
        call_count = 0
        call_lock = threading.Lock()

        def fake_synthesize(*_args):
            nonlocal call_count
            with call_lock:
                call_count += 1
                current_call = call_count
            if current_call == 1:
                first_synthesis_started.set()
                if not release_first_synthesis.wait(timeout=5):
                    raise TimeoutError("Test did not release the first synthesis.")
            return np.ones(12, dtype=np.float32), 24000

        with patch.object(reader, 'synthesize', side_effect=fake_synthesize):
            response = self._upload()
            self.assertEqual(response.status_code, 202)
            job_id = response.get_json()['id']
            self.assertTrue(first_synthesis_started.wait(timeout=5))

            stop_response = self.client.post(f'/api/jobs/{job_id}/stop')
            self.assertEqual(stop_response.status_code, 200)
            release_first_synthesis.set()
            stopped = self._wait_for_state(job_id, 'stopped')
            self.assertEqual(stopped['completed_chunks'], 1)
            self.assertEqual(stopped['outputs'], [])

            resumed_response = self.client.post(f'/api/jobs/{job_id}/resume')
            self.assertEqual(resumed_response.status_code, 200)
            completed = self._wait_for_state(job_id, 'completed')

        self.assertEqual(completed['percentage'], 100)
        self.assertEqual(len(completed['outputs']), 1)
        self.assertEqual(call_count, completed['total_chunks'])

        download = self.client.get(
            f"/api/jobs/{job_id}/files/{completed['outputs'][0]}"
        )
        self.assertEqual(download.status_code, 200)
        download.close()
        output_path = (
            Path(completed['output_directory']) / completed['outputs'][0]
        )
        self.assertEqual(
            sf.info(str(output_path)).frames,
            12 * completed['total_chunks']
            + round(24000 * reader.HEADING_PAUSE_SECONDS),
        )

    def test_pause_and_continue_controls_generation(self):
        first_synthesis_started = threading.Event()
        release_first_synthesis = threading.Event()
        call_count = 0

        def fake_synthesize(*_args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                first_synthesis_started.set()
                if not release_first_synthesis.wait(timeout=5):
                    raise TimeoutError("Test did not release the first synthesis.")
            return np.ones(12, dtype=np.float32), 24000

        with patch.object(reader, 'synthesize', side_effect=fake_synthesize):
            response = self._upload()
            self.assertEqual(response.status_code, 202)
            job_id = response.get_json()['id']
            self.assertTrue(first_synthesis_started.wait(timeout=5))

            paused_response = self.client.post(f'/api/jobs/{job_id}/pause')
            self.assertEqual(paused_response.status_code, 200)
            release_first_synthesis.set()
            deadline = time.monotonic() + 10
            paused = None
            while time.monotonic() < deadline:
                status = self.client.get(f'/api/jobs/{job_id}').get_json()
                if status['state'] == 'failed':
                    self.fail(f"Job failed: {status['error']}")
                if status['state'] == 'paused' and status['completed_chunks'] == 1:
                    paused = status
                    break
                time.sleep(0.02)
            self.assertIsNotNone(paused, "Job did not pause after its current chunk.")
            assert paused is not None
            self.assertEqual(paused['completed_chunks'], 1)

            continued_response = self.client.post(f'/api/jobs/{job_id}/resume')
            self.assertEqual(continued_response.status_code, 200)
            completed = self._wait_for_state(job_id, 'completed')

        self.assertEqual(completed['percentage'], 100)


if __name__ == '__main__':
    unittest.main()
