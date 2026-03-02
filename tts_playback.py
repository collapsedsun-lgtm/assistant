"""Playback manager for non-blocking, queued audio playback with optional volume control.

Uses `ffplay` when available (supports volume filter and stdin piping) and falls
back to `aplay` for raw WAV stdin playback. Runs a background worker thread to
play queued items; supports `enqueue_bytes`, `enqueue_file`, `play_now`, and
`stop` (which can clear the queue).

This intentionally keeps dependencies to the stdlib and shell players to avoid
adding audio libraries.
"""
from __future__ import annotations

import threading
import queue
import subprocess
import shutil
from typing import Optional


class PlaybackManager:
    def __init__(self, preferred_player: Optional[str] = None, default_volume: float = 1.0, queue_enabled: bool = True):
        self._player = self._select_player(preferred_player)
        self._default_volume = float(default_volume)
        self._queue_enabled = queue_enabled
        self._q: "queue.Queue[tuple]" = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._current_proc: Optional[subprocess.Popen] = None
        self._stop_event = threading.Event()
        self._worker.start()

    def _select_player(self, preferred: Optional[str]) -> str:
        if preferred and shutil.which(preferred):
            return preferred
        # Prefer ffplay for volume control and stdin support
        if shutil.which("ffplay"):
            return "ffplay"
        if shutil.which("aplay"):
            return "aplay"
        if shutil.which("paplay"):
            return "paplay"
        # last resort: use xdg-open (requires files)
        if shutil.which("xdg-open"):
            return "xdg-open"
        return "none"

    def enqueue_bytes(self, audio_bytes: bytes, volume: Optional[float] = None):
        self._q.put(("bytes", audio_bytes, float(volume) if volume is not None else None))

    def enqueue_file(self, path: str, volume: Optional[float] = None):
        self._q.put(("file", path, float(volume) if volume is not None else None))

    def play_now_bytes(self, audio_bytes: bytes, volume: Optional[float] = None):
        # Stop current and put item at front by stopping then enqueueing
        self.stop()
        self.enqueue_bytes(audio_bytes, volume)

    def play_now_file(self, path: str, volume: Optional[float] = None):
        self.stop()
        self.enqueue_file(path, volume)

    def stop(self, clear_queue: bool = False):
        try:
            if self._current_proc and self._current_proc.poll() is None:
                try:
                    self._current_proc.terminate()
                except Exception:
                    try:
                        self._current_proc.kill()
                    except Exception:
                        pass
        finally:
            if clear_queue:
                with self._q.mutex:
                    self._q.queue.clear()

    def _run(self):
        while True:
            try:
                item_type, payload, vol = self._q.get()
            except Exception:
                continue

            if vol is None:
                vol = self._default_volume

            try:
                if item_type == "bytes":
                    self._play_bytes(payload, vol)
                elif item_type == "file":
                    self._play_file(payload, vol)
            except Exception:
                # swallow playback exceptions to keep worker alive
                pass

    def _play_bytes(self, audio_bytes: bytes, volume: float):
        if self._player == "ffplay":
            cmd = [
                "ffplay",
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "quiet",
                "-af",
                f"volume={volume}",
                "-i",
                "-",
            ]
            self._current_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            try:
                self._current_proc.stdin.write(audio_bytes)
                self._current_proc.stdin.close()
                self._current_proc.wait()
            finally:
                self._current_proc = None
            return

        if self._player == "aplay":
            # aplay supports wav stdin
            cmd = ["aplay", "-t", "wav", "-"]
            self._current_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            try:
                self._current_proc.stdin.write(audio_bytes)
                self._current_proc.stdin.close()
                self._current_proc.wait()
            finally:
                self._current_proc = None
            return

        # Fallback: write to a temporary file and play
        import tempfile, os

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            with open(path, "wb") as fh:
                fh.write(audio_bytes)
            self._play_file(path, volume)
        finally:
            try:
                os.remove(path)
            except Exception:
                pass

    def _play_file(self, path: str, volume: float):
        if self._player == "ffplay":
            cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-af", f"volume={volume}", path]
            self._current_proc = subprocess.Popen(cmd)
            try:
                self._current_proc.wait()
            finally:
                self._current_proc = None
            return

        if self._player == "aplay":
            cmd = ["aplay", path]
            self._current_proc = subprocess.Popen(cmd)
            try:
                self._current_proc.wait()
            finally:
                self._current_proc = None
            return

        if self._player == "xdg-open":
            subprocess.Popen(["xdg-open", path])
            return

        # No player: no-op
        return


__all__ = ["PlaybackManager"]
