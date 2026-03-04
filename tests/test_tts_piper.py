import io
import types
import sys
import os
import pytest

# Ensure project root is on sys.path so imports work when running pytest from
# the workspace root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tts_piper import PiperTTS


class DummyResp:
    def __init__(self, chunks):
        self._chunks = chunks
        self.status_code = 200
        self.content = b"".join(chunks)

    def raise_for_status(self):
        if getattr(self, "status_code", 200) != 200:
            raise Exception("bad status")

    def iter_content(self, chunk_size=4096):
        for c in self._chunks:
            yield c


def test_http_stream_synthesize(monkeypatch):
    chunks = [b"hello", b" ", b"world"]

    def fake_post(url, json=None, headers=None, timeout=None, stream=False):
        return DummyResp(chunks)

    monkeypatch.setattr("tts_piper.requests.post", fake_post)

    p = PiperTTS(mode="http", server_url="http://example.local/synthesize")
    gen = p.stream_synthesize("hi")
    out = b"".join(list(gen))
    assert out == b"hello world"


def test_binary_stream_synthesize_stdout():
    # Use a short Python one-liner to write two chunks to stdout
    code = "import sys; sys.stdout.buffer.write(b'chunkA'); sys.stdout.buffer.write(b'chunkB')"
    p = PiperTTS(mode="binary", binary_cmd=[sys.executable, "-u", "-c", code])
    gen = p.stream_synthesize("ignored")
    data = b"".join(list(gen))
    assert data == b"chunkAchunkB"
