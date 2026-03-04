KNOWN BUGS
===========

This file tracks known, reproducible issues in the assistant codebase and
workarounds or short-term mitigation steps.

1) `aplay` prints `read_header:2912: read error` when `tts.streaming_output` is enabled
- Symptom: When `tts.streaming_output` is true and the worker pipes audio chunks to `aplay -t wav -`, aplay logs repeated "read_header:... read error" messages and playback fails for some sentences. You may still hear audio for other sentences.
- Cause: `aplay` expects a complete, valid WAV header up front. If the TTS stream sends raw PCM, sends the header later, or the header arrives split across chunks, `aplay` fails to parse it and prints this error. The pipe closing between short utterances can also trigger the message.
- Workarounds:
  - Use `ffplay` when available (preferred) — it's more tolerant of chunked/partial streams and will play streamed WAV/PCM with lower latency.
  - If you must use `aplay`, buffer incoming chunks until a valid WAV header (`RIFF....WAVE`) is seen, then start `aplay` and stream the rest; otherwise write the bytes to a temp WAV file and play that.
  - Disable `tts.streaming_output` (use `tts.streaming` without `streaming_output`) so the worker synthesizes full sentences before playback (less latency but reliable playback).

2) Duplicate console printing / duplicate TTS enqueues (resolved)
- Symptom (prior behavior): when sentence streaming was enabled the console sometimes printed partial sentences and then printed the final reply again; TTS could be enqueued twice (sentences + full reply).
- Status: Fixed. The CLI now only suppresses the final printed assistant reply when progressive printing actually occurred, and it avoids calling full-text `_maybe_speak()` when sentence-level streaming is active.

3) ONNX/Runtime GPU discovery warning
- Symptom: WARNING logs like:
  `GPU device discovery failed: ReadFileContents Failed to open file: "/sys/class/drm/card0/device/vendor"`
- Meaning: ONNX Runtime attempted to detect GPU/DRM devices and couldn't read system device files. This is a discovery log, not a fatal error — the runtime falls back to CPU execution provider.
- Mitigation: To force CPU-only behavior, set `CUDA_VISIBLE_DEVICES=""` when launching the assistant or add `--device cpu` to the Piper CLI (if supported).

4) Misc syntax/indentation fixes during development
- Several temporary edits were applied to `cli.py` to support streaming and to make the synth worker testable; those were validated with `py_compile` and unit tests. If you see unexpected behavior, re-run the CLI with `--debug` and file an issue with the transcript.

How to report new bugs
- Reproduce with `python3 main.py --debug` and capture the console transcript.
- Note your `settings.json` `tts` and `llm` flags and whether `aplay` or `ffplay` is present on PATH.
- Open an issue or paste the transcript into a message and the maintainer will triage.
