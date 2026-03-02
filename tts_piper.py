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
from typing import List, Optional, Union

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


__all__ = ["PiperTTS"]
