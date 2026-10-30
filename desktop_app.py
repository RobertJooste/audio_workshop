"""Launch the local audiobook web app as a desktop application."""

import logging
import secrets
import sys
import webbrowser
from pathlib import Path

from platformdirs import user_data_path, user_log_path
from werkzeug.serving import make_server

import web_app


APP_NAME = 'EPUB Audiobook Maker'
logger = logging.getLogger(__name__)


def _show_message(title: str, message: str, *, error: bool = False) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        if error:
            messagebox.showerror(title, message, parent=root)
        else:
            messagebox.showinfo(title, message, parent=root)
        root.destroy()
    except Exception:
        logger.exception('Could not show the desktop notification.')


def _configure_logging() -> Path:
    log_dir = Path(user_log_path(APP_NAME, appauthor=False))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / 'app.log'
    log_stream = log_path.open('a', encoding='utf-8', buffering=1)
    sys.stdout = log_stream
    sys.stderr = log_stream
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        handlers=[logging.StreamHandler(log_stream)],
        force=True,
    )
    return log_path


def main() -> None:
    log_path = _configure_logging()
    server = None
    try:
        data_dir = Path(user_data_path(APP_NAME, appauthor=False))
        app = web_app.create_app(
            storage_root=data_dir / 'state',
            output_root=Path.home() / 'Audiobooks',
            desktop_shutdown_token=secrets.token_urlsafe(32),
        )
        server = make_server(
            '127.0.0.1',
            0,
            app,
            threaded=True,
        )

        def stop_app() -> None:
            manager = app.extensions['job_manager']
            with manager.lock:
                jobs = list(manager.jobs.values())
            for job in jobs:
                try:
                    manager.stop(job)
                except ValueError:
                    continue
            for job in jobs:
                if job.thread is not None:
                    job.thread.join()
            server.shutdown()

        app.config['DESKTOP_SHUTDOWN_CALLBACK'] = stop_app
        url = f'http://127.0.0.1:{server.server_port}'
        logger.info('Starting local app at %s', url)
        if not webbrowser.open_new(url):
            _show_message(
                APP_NAME,
                f'Open this address in your browser:\n\n{url}',
            )
        server.serve_forever()
    except Exception as exc:
        logger.exception('Could not start the desktop app.')
        _show_message(
            f'{APP_NAME} could not start',
            f'{exc}\n\nMore details are in the log file:\n{log_path}',
            error=True,
        )
    finally:
        if server is not None:
            server.server_close()


if __name__ == '__main__':
    main()
