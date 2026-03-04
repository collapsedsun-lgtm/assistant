"""Simple Piper TTS wrapper supporting HTTP or subprocess modes.

Usage examples:
- HTTP mode (recommended if you run a Piper server):
    p = PiperTTS(mode="http", server_url="http://localhost:5002/synthesize")
    p.synthesize("hello world", "out.wav")

- Subprocess mode (generic): provide a command template that writes audio to {out}:
    cmd = ["piper", "--model", "myvoice", "--text", "{text}", "-o", "{out}"]
    p = PiperTTS(mode="binary", binary_cmd=cmd)
    p.synthesize("hello", "out.wav")

Note: this module does not assume specific Piper CLI flags or HTTP endpoints —
provide the server URL or a command template appropriate for your Piper install.
"""
from __future__ import annotations

import shlex
import subprocess
from typing import List, Optional, Union, Generator

import requests


class PiperTTS:
    def __init__(
        self,
        mode: str = "http",
        server_url: Optional[str] = None,
        binary_cmd: Optional[Union[str, List[str]]] = None,
        headers: Optional[dict] = None,
        timeout: int = 60,
    ) -> None:
        """Create a PiperTTS instance.

        - mode: 'http' or 'binary'
        - server_url: full URL to POST text to (HTTP mode)
        - binary_cmd: string or list template with '{text}' and '{out}' placeholders (binary mode)
        - headers: optional HTTP headers for the request
        - timeout: request timeout seconds
        """
        self.mode = mode
        self.server_url = server_url
        self.binary_cmd = binary_cmd
        self.headers = headers or {}
        self.timeout = timeout

    def synthesize(self, text: str, out_path: Optional[str] = None) -> Union[str, bytes]:
        """Synthesize `text` and either write audio to `out_path` or return bytes.

        - If `out_path` is provided, write audio to that path and return the path.
        - If `out_path` is None, return the raw audio bytes (HTTP mode returns WAV bytes).

        Raises exceptions on failure.
        """
        if self.mode == "http":
            if not self.server_url:
                raise ValueError("server_url must be set for HTTP mode")

            payload = {"text": text}
            # Many Piper HTTP servers accept raw POST returning audio bytes; adapt headers/payload as needed.
            resp = requests.post(self.server_url, json=payload, headers=self.headers, timeout=self.timeout)
            resp.raise_for_status()
            if out_path:
                with open(out_path, "wb") as fh:
                    fh.write(resp.content)
                return out_path
            else:
                return resp.content

        elif self.mode == "binary":
            if not self.binary_cmd:
                raise ValueError("binary_cmd must be provided for binary mode")

            # Accept either a single string (shell) or a list of args.
            if isinstance(self.binary_cmd, str):
                cmd = self.binary_cmd.format(text=shlex.quote(text), out=shlex.quote(out_path) if out_path else "")
                if out_path:
                    subprocess.run(cmd, shell=True, check=True)
                    return out_path
                else:
                    proc = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE)
                    return proc.stdout
            else:
                cmd = [arg.format(text=text, out=out_path or "") for arg in self.binary_cmd]
                if out_path:
                    subprocess.run(cmd, check=True)
                    return out_path
                else:
                    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE)
                    return proc.stdout

        else:
            raise ValueError("mode must be 'http' or 'binary'")

    def stream_synthesize(self, text: str, chunk_size: int = 4096) -> Generator[bytes, None, None]:
        """Yield audio bytes as they are produced by Piper.

        - For HTTP mode, performs a streaming POST (if the server supports chunked responses).
        - For binary mode, runs the CLI and yields stdout chunks as they arrive.

        Note: caller is responsible for consuming the generator promptly to avoid
        blocking the underlying process.
        """
        if self.mode == "http":
            if not self.server_url:
                raise ValueError("server_url must be set for HTTP mode")
            payload = {"text": text}
            # Request with streaming; server must support streaming responses for low latency
            resp = requests.post(self.server_url, json=payload, headers=self.headers, timeout=self.timeout, stream=True)
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    yield chunk

        elif self.mode == "binary":
            if not self.binary_cmd:
                raise ValueError("binary_cmd must be provided for binary mode")

            # For binary mode, invoke the command and stream stdout. We assume the
            # provided command writes WAV bytes to stdout when {out} is empty.
            if isinstance(self.binary_cmd, str):
                # If string, format and run in shell
                cmd = self.binary_cmd.format(text=shlex.quote(text), out="")
                proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            else:
                cmd = [arg.format(text=text, out="") for arg in self.binary_cmd]
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

            try:
                assert proc.stdout is not None
                while True:
                    data = proc.stdout.read(chunk_size)
                    if not data:
                        break
                    yield data
                proc.wait()
            finally:
                try:
                    if proc.stdout:
                        proc.stdout.close()
                except Exception:
                    pass

        else:
            raise ValueError("mode must be 'http' or 'binary'")


__all__ = ["PiperTTS"]
