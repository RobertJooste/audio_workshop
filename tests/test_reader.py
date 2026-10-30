from pathlib import Path
from tempfile import TemporaryDirectory
import io
import unittest
from unittest.mock import MagicMock, patch

import epub_pdf_tts_reader as reader
from ebooklib import epub
import numpy as np
import soundfile as sf
import termios
import tty


class EpubExtractionTests(unittest.TestCase):
    def test_extracts_spine_order_and_omits_navigation_and_footnotes(self):
        with TemporaryDirectory() as directory:
            book = epub.EpubBook()
            book.set_identifier('test-book')
            book.set_title('Test Book')
            book.set_language('en')
            first = epub.EpubHtml(uid='first', file_name='first.xhtml', title='First', lang='en')
            first.set_content(
                '<html><head><title>Metadata title</title></head><body>'
                '<nav>Contents page.</nav><h1>First</h1>'
                '<p>First paragraph.</p><p>Second paragraph.</p>'
                '<script>Not book text.</script></body></html>'
            )
            later = epub.EpubHtml(uid='later', file_name='later.xhtml', title='Later', lang='en')
            later.set_content(
                '<html><body><h1>Later</h1><p>Second chapter.</p>'
                '<aside>Sidebar text.</aside>'
                '<p role="doc-footnote">Footnote text.</p></body></html>'
            )
            book.add_item(first)
            book.add_item(later)
            book.add_item(epub.EpubNcx())
            book.add_item(epub.EpubNav())
            book.toc = (first, later)
            book.spine = ['nav', later, first]
            epub_path = Path(directory) / 'book.epub'
            epub.write_epub(str(epub_path), book)

            chapters = reader.extract_chapters_from_epub(str(epub_path))

        self.assertEqual([chapter.title for chapter in chapters], ['Later', 'First'])
        self.assertEqual(
            chapters[1].text,
            'First.\n\nFirst paragraph.\n\nSecond paragraph.',
        )
        self.assertEqual(chapters[1].speech_heading, 'First.')
        self.assertEqual(chapters[0].speech_heading, 'Later.')
        self.assertNotIn('Contents page', chapters[1].text)
        self.assertNotIn('Footnote text', chapters[0].text)
        self.assertNotIn('Sidebar text', chapters[0].text)

    def test_unpunctuated_heading_is_given_a_sentence_boundary(self):
        with TemporaryDirectory() as directory:
            book = epub.EpubBook()
            book.set_identifier('heading-pause-test')
            book.set_title('Heading Pause Test')
            book.set_language('en')
            chapter = epub.EpubHtml(
                uid='chapter',
                file_name='chapter.xhtml',
                title='An Empty World',
                lang='en',
            )
            chapter.set_content(
                '<html><body><h1>Chapter 1. An empty world</h1>'
                '<p>The Wheel of Time turns.</p></body></html>'
            )
            book.add_item(chapter)
            book.add_item(epub.EpubNcx())
            book.add_item(epub.EpubNav())
            book.toc = (chapter,)
            book.spine = ['nav', chapter]
            epub_path = Path(directory) / 'heading.epub'
            epub.write_epub(str(epub_path), book)

            [extracted] = reader.extract_chapters_from_epub(str(epub_path))

        self.assertEqual(
            extracted.text,
            'Chapter 1. An empty world.\n\nThe Wheel of Time turns.',
        )
        self.assertEqual(
            extracted.speech_heading,
            'Chapter 1. An empty world.',
        )
        self.assertEqual(
            reader.split_text(reader.clean_text(extracted.text)),
            ['Chapter 1. An empty world.\n\nThe Wheel of Time turns.'],
        )

    def test_extracts_plain_chapter_label_and_title_paragraphs_as_heading(self):
        with TemporaryDirectory() as directory:
            book = epub.EpubBook()
            book.set_identifier('plain-heading-test')
            book.set_title('Plain Heading Test')
            book.set_language('en')
            chapter = epub.EpubHtml(
                uid='chapter',
                file_name='chapter.xhtml',
                title='Chapter',
                lang='en',
            )
            chapter.set_content(
                '<html><body><p>Chapter 1</p><p>The Road</p>'
                '<p>When the sun came up, the road was empty.</p></body></html>'
            )
            book.add_item(chapter)
            book.add_item(epub.EpubNcx())
            book.add_item(epub.EpubNav())
            book.toc = (chapter,)
            book.spine = ['nav', chapter]
            epub_path = Path(directory) / 'plain-heading.epub'
            epub.write_epub(str(epub_path), book)

            [extracted] = reader.extract_chapters_from_epub(str(epub_path))

        self.assertEqual(extracted.speech_heading, 'Chapter 1. The Road.')
        self.assertEqual(
            extracted.speech_body,
            'When the sun came up, the road was empty.',
        )

    def test_extracts_styled_heading_paragraph_sequence_separately_from_body(self):
        with TemporaryDirectory() as directory:
            book = epub.EpubBook()
            book.set_identifier('styled-heading-test')
            book.set_title('Styled Heading Test')
            book.set_language('en')
            chapter = epub.EpubHtml(
                uid='chapter',
                file_name='chapter.xhtml',
                title='Chapter',
                lang='en',
            )
            chapter.set_content(
                '<html><body>'
                '<p class="h4">CHAPTER</p>'
                '<p class="h4">11</p>'
                '<p class="h3">The Road to Taren Ferry</p>'
                '<p class="tx16">O n the hard-packed road, when dawn came, they left.</p>'
                '</body></html>'
            )
            book.add_item(chapter)
            book.add_item(epub.EpubNcx())
            book.add_item(epub.EpubNav())
            book.toc = (chapter,)
            book.spine = ['nav', chapter]
            epub_path = Path(directory) / 'styled-heading.epub'
            epub.write_epub(str(epub_path), book)

            [extracted] = reader.extract_chapters_from_epub(str(epub_path))

        self.assertEqual(
            extracted.speech_heading,
            'CHAPTER 11. The Road to Taren Ferry.',
        )
        self.assertEqual(
            extracted.speech_body,
            'O n the hard-packed road, when dawn came, they left.',
        )
        self.assertNotIn('when dawn', extracted.speech_heading.lower())

        synthesis_calls = []

        def fake_synthesize(_model, text, *_args):
            synthesis_calls.append(text)
            return np.ones(8, dtype=np.float32), 24000

        with TemporaryDirectory() as directory:
            output_path = Path(directory) / 'chapter.wav'
            with patch.object(reader, 'synthesize', side_effect=fake_synthesize):
                reader.generate_audio_files(
                    [extracted],
                    object(),
                    'af_bella',
                    1.0,
                    'en-us',
                    str(output_path),
                    Path(directory) / 'progress',
                    resume=False,
                )
        self.assertEqual(
            synthesis_calls,
            [
                'CHAPTER 11. The Road to Taren Ferry.',
                'O n the hard-packed road, when dawn came, they left.',
            ],
        )

    def test_splits_toc_linked_pov_chapters_across_spine_documents(self):
        with TemporaryDirectory() as directory:
            book = epub.EpubBook()
            book.set_identifier('toc-pov-chapters-test')
            book.set_title('TOC POV Chapters Test')
            book.set_language('en')
            first = epub.EpubHtml(
                uid='part-one',
                file_name='part-one.xhtml',
                title='Part One',
                lang='en',
            )
            first.set_content(
                '<html><body><p>Front matter must be skipped.</p>'
                '<p><a id="arya"></a><b>ARYA</b></p>'
                '<p>The first chapter begins here.</p></body></html>'
            )
            second = epub.EpubHtml(
                uid='part-two',
                file_name='part-two.xhtml',
                title='Part Two',
                lang='en',
            )
            second.set_content(
                '<html><body><p>The first chapter continues here.</p>'
                '<p><a id="jon"></a><b>JON</b></p>'
                '<p>The second chapter begins here.</p></body></html>'
            )
            book.add_item(first)
            book.add_item(second)
            book.add_item(epub.EpubNcx())
            book.add_item(epub.EpubNav())
            book.toc = (
                epub.Link('part-one.xhtml#arya', 'Arya', 'arya'),
                epub.Link('part-two.xhtml#jon', 'Jon', 'jon'),
            )
            book.spine = ['nav', first, second]
            epub_path = Path(directory) / 'toc-pov.epub'
            epub.write_epub(str(epub_path), book)

            chapters = reader.extract_chapters_from_epub(str(epub_path))

        self.assertEqual([chapter.title for chapter in chapters], ['Arya', 'Jon'])
        self.assertEqual(
            [chapter.speech_heading for chapter in chapters],
            ['Arya.', 'Jon.'],
        )
        self.assertEqual(
            [chapter.speech_body for chapter in chapters],
            [
                'The first chapter begins here.\n\n'
                'The first chapter continues here.',
                'The second chapter begins here.',
            ],
        )
        self.assertNotIn('Front matter', chapters[0].text)


class TextProcessingTests(unittest.TestCase):
    def test_clean_text_preserves_paragraph_breaks(self):
        self.assertEqual(
            reader.clean_text('First  paragraph.\n\n  Second\nparagraph.'),
            'First paragraph.\n\nSecond paragraph.',
        )

    def test_split_text_preserves_paragraph_breaks(self):
        self.assertEqual(
            reader.split_text('First paragraph.\n\nSecond paragraph.', max_length=100),
            ['First paragraph.\n\nSecond paragraph.'],
        )

    def test_split_text_caps_chunks_even_for_long_sentences(self):
        chunks = reader.split_text('A sentence. ' + 'word ' * 14, max_length=20)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(chunk) <= 20 for chunk in chunks))
        expected = ('A sentence. ' + 'word ' * 14).strip()
        self.assertEqual(' '.join(chunks), expected)

    def test_split_text_rejects_nonpositive_limits(self):
        with self.assertRaises(ValueError):
            reader.split_text('Text.', max_length=0)

    def test_blends_voice_styles_using_secondary_percentage(self):
        primary = np.zeros((2, 1, 3), dtype=np.float32)
        secondary = np.ones((2, 1, 3), dtype=np.float32)
        model = MagicMock()
        model.get_voice_style.side_effect = [primary, secondary]

        blended = reader.blend_voice_styles(model, 'af_bella', 'af_sarah', 30)

        np.testing.assert_allclose(blended, np.full((2, 1, 3), 0.3))
        self.assertEqual(blended.dtype, np.float32)
        with self.assertRaises(ValueError):
            reader.blend_voice_styles(model, 'af_bella', 'af_sarah', 101)

    def test_pronunciation_replacements_are_case_insensitive_and_word_bounded(self):
        self.assertEqual(
            reader.apply_pronunciation_replacements(
                'The WHEEL and Wheelhouse.',
                [('wheel', 'weel')],
            ),
            'The weel and Wheelhouse.',
        )


class AudioOutputTests(unittest.TestCase):
    def test_heading_and_first_sentence_are_synthesized_separately_with_audible_gap(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            chapter = reader.Chapter(
                index=0,
                title='The Road',
                text='Chapter 1. The Road.\n\nWhen morning came, we left.',
                speech_heading='Chapter 1. The Road.',
                speech_body='When morning came, we left.',
            )
            synthesized_text = []

            def fake_synthesize(_model, text, *_args):
                synthesized_text.append(text)
                return np.ones(8, dtype=np.float32), 24000

            output_path = root / 'chapter.wav'
            with patch.object(reader, 'synthesize', side_effect=fake_synthesize):
                outputs = reader.generate_audio_files(
                    [chapter],
                    object(),
                    'af_bella',
                    1.0,
                    'en-us',
                    str(output_path),
                    root / 'progress',
                    resume=False,
                    heading_pause_seconds=1.0,
                )

            self.assertEqual(outputs, [output_path])
            self.assertEqual(
                synthesized_text,
                ['Chapter 1. The Road.', 'When morning came, we left.'],
            )
            samples, _ = sf.read(str(output_path), dtype='float32')
            pause_start = 8
            pause_end = pause_start + 24000
            self.assertEqual(len(samples), pause_end + 8)
            self.assertTrue(np.all(samples[pause_start:pause_end] == 0))
            self.assertAlmostEqual(samples[pause_end], 1, places=4)

    def test_space_pauses_and_continues_audio_playback(self):
        samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        input_stream = MagicMock()
        input_stream.isatty.return_value = True
        input_stream.fileno.return_value = 10
        input_stream.read.side_effect = [' ', ' ', '']
        ready_states = [[input_stream], [input_stream], [], []]
        writes = []
        output_stream = MagicMock()
        output_stream.__enter__.return_value = output_stream
        output_stream.write.side_effect = lambda block: writes.append(block.copy())

        with (
            patch('sys.stdin', input_stream),
            patch.object(reader.sd, 'OutputStream', return_value=output_stream),
            patch.object(reader.select, 'select', side_effect=[
                (ready, [], []) for ready in ready_states
            ]),
            patch.object(termios, 'tcgetattr', return_value=[]),
            patch.object(termios, 'tcsetattr'),
            patch.object(tty, 'setcbreak'),
            patch('sys.stdout', new_callable=io.StringIO),
        ):
            reader.play_audio(samples, sample_rate=25)

        np.testing.assert_array_equal(writes[0], [[0.0]])
        np.testing.assert_array_equal(np.concatenate(writes[1:]).ravel(), samples)

    def test_list_voices_does_not_require_a_book_file(self):
        model = MagicMock()
        model.get_voices.return_value = ['af_bella', 'am_adam', 'jf_alpha']

        with (
            patch.object(reader, 'load_model', return_value=model),
            patch('sys.argv', ['epub_pdf_tts_reader.py', '--list-voices']),
            patch('sys.stdout', new_callable=io.StringIO) as output,
        ):
            reader.main()

        self.assertIn('af_bella', output.getvalue())
        self.assertIn('am_adam', output.getvalue())
        self.assertNotIn('jf_alpha', output.getvalue())

    def test_partial_export_is_not_marked_complete_and_resume_finishes_it(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            epub_path = root / 'book.epub'
            output_path = root / 'book.wav'
            progress_root = root / 'progress'

            book = epub.EpubBook()
            book.set_identifier('audio-test')
            book.set_title('Audio Test')
            book.set_language('en')
            chapter = epub.EpubHtml(uid='chapter', file_name='chapter.xhtml', title='Chapter', lang='en')
            paragraph = ' '.join(['This is a full sentence for audio output.'] * 80)
            chapter.set_content(f'<html><body><h1>Chapter</h1><p>{paragraph}</p></body></html>')
            book.add_item(chapter)
            book.add_item(epub.EpubNcx())
            book.add_item(epub.EpubNav())
            book.toc = (chapter,)
            book.spine = ['nav', chapter]
            epub.write_epub(str(epub_path), book)

            def fake_synthesize(*_args):
                return np.ones(8, dtype=np.float32), 24000

            model = MagicMock()
            model.get_voices.return_value = ['af_bella']
            with (
                patch.object(reader, '_PROGRESS_ROOT', progress_root),
                patch.object(reader, 'load_model', return_value=model),
                patch.object(reader, 'synthesize', side_effect=fake_synthesize),
                patch('sys.argv', [
                    'epub_pdf_tts_reader.py', str(epub_path), '--save', str(output_path),
                    '--no-play', '--limit', '1',
                ]),
            ):
                reader.main()

            self.assertFalse(output_path.exists())
            self.assertEqual(len(list(progress_root.rglob('*.wav'))), 1)

            with (
                patch.object(reader, '_PROGRESS_ROOT', progress_root),
                patch.object(reader, 'load_model', return_value=model),
                patch.object(reader, 'synthesize', side_effect=fake_synthesize),
                patch('sys.argv', [
                    'epub_pdf_tts_reader.py', str(epub_path), '--save', str(output_path),
                    '--no-play', '--resume',
                ]),
            ):
                reader.main()

            expected_chunks = 1 + len(reader.split_text(reader.clean_text(paragraph)))
            self.assertTrue(output_path.exists())
            self.assertEqual(
                sf.info(str(output_path)).frames,
                expected_chunks * 8 + round(24000 * reader.HEADING_PAUSE_SECONDS),
            )
            samples, _ = sf.read(str(output_path))
            heading_pause_start = 8
            heading_pause_end = heading_pause_start + round(
                24000 * reader.HEADING_PAUSE_SECONDS
            )
            self.assertTrue(np.all(samples[heading_pause_start:heading_pause_end] == 0))
            self.assertAlmostEqual(samples[heading_pause_end], 1, places=4)
            self.assertEqual(list(progress_root.rglob('*.wav')), [])

    def test_save_path_is_always_a_supported_wav_path(self):
        self.assertEqual(reader.normalize_wav_save_path('book'), 'book.wav')
        with self.assertRaises(ValueError):
            reader.normalize_wav_save_path('book.mp3')


if __name__ == '__main__':
    unittest.main()
