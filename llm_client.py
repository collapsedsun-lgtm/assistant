import aiohttp
import asyncio
import inspect
import json
import os
import time
from typing import List, Optional

from llm_config import MODEL, OLLAMA_URL, LLM_OPTIONS
from llm_config import LLM_STREAM, PARTIAL_TTL, FINAL_TTL, PARTIAL_SAVE_THRESHOLD, LLM_TIMEOUT
import actions
import memory_summarizer
from kv_store import get_default_kv, response_cache_key


def _extract_texts(obj):
    out = []
    if obj is None:
        return out
    if isinstance(obj, str):
        out.append(obj)
        return out
    if isinstance(obj, dict):
        message = obj.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content:
                out.append(content)

        response = obj.get("response")
        if isinstance(response, str) and response:
            out.append(response)

        delta = obj.get("delta")
        if isinstance(delta, dict):
            dcontent = delta.get("content")
            if isinstance(dcontent, str) and dcontent:
                out.append(dcontent)

        text = obj.get("text")
        if isinstance(text, str) and text:
            out.append(text)

        content = obj.get("content")
        if isinstance(content, str) and content:
            out.append(content)

        if out:
            return out

        for v in obj.values():
            out.extend(_extract_texts(v))
        return out
    if isinstance(obj, list):
        for it in obj:
            out.extend(_extract_texts(it))
        return out
    return out


def load_system_prompt() -> str:
    path = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_settings() -> dict:
    path = os.path.join(os.path.dirname(__file__), "settings.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


async def call_llm(user_input: str, history: List[dict], mock: bool = False, debug: bool = False, rolling_window: int = 5, summarize_memory: bool = False, pre_send_hooks: List[callable] | None = None) -> tuple:
    if mock:
        if "time" in user_input.lower():
            return json.dumps({"tool": "get_time", "args": {}}), []
        return "I would do that if I could.", []

    system_prompt = load_system_prompt()
    actions_list = actions.load_actions()
    actions_desc = actions.actions_description(actions_list)

    settings = load_settings()
    is_chat_endpoint = OLLAMA_URL.rstrip("/").endswith("/api/chat")
    use_session = settings.get("ollama_use_session", False) and (not is_chat_endpoint)

    if use_session:
        system_prompt_to_send = ""
        actions_desc_to_send = ""
    else:
        system_prompt_to_send = system_prompt
        actions_desc_to_send = actions_desc

    prefetch_texts: List[str] = []
    if pre_send_hooks:
        for hook in pre_send_hooks:
            try:
                maybe = None
                try:
                    sig = inspect.signature(hook)
                    kwargs = {}
                    params = sig.parameters
                    if "debug" in params:
                        kwargs["debug"] = debug
                    if "settings" in params or "config" in params:
                        kwargs["settings"] = settings
                    maybe = await hook(user_input, history, **kwargs)
                except Exception:
                    try:
                        maybe = await hook(user_input, history)
                    except TypeError:
                        try:
                            maybe = await hook(user_input)
                        except TypeError:
                            maybe = None

                if maybe:
                    if isinstance(maybe, str):
                        prefetch_texts.append(maybe)
                    else:
                        try:
                            prefetch_texts.append(json.dumps(maybe))
                        except Exception:
                            prefetch_texts.append(str(maybe))
                if debug:
                    print(f"[debug] pre_send hook {getattr(hook, '__name__', repr(hook))} returned: {repr(maybe)[:400]}")
            except Exception:
                if debug:
                    import traceback
                    print("[debug] pre_send hook failed:\n", traceback.format_exc())

    if use_session:
        messages: List[dict] = []
        if prefetch_texts:
            combined = "\n\n".join(t for t in prefetch_texts if t)
            if combined:
                messages.append({"role": "system", "content": "Pre-fetched facts (sanitized):\n" + combined})

        tool_reply_history: List[dict] = []
        for item in reversed(history):
            if item.get("role") != "assistant":
                continue
            content = item.get("content")
            if not isinstance(content, str):
                continue
            try:
                parsed = json.loads(content)
            except Exception:
                continue
            if isinstance(parsed, dict) and "tool" in parsed and "args" in parsed:
                tool_reply_history.append({"role": "assistant", "content": content})
            if len(tool_reply_history) >= 4:
                break
        tool_reply_history.reverse()
        messages.extend(tool_reply_history)
        messages.append({"role": "user", "content": user_input})
    else:
        # Build messages locally to avoid importing assistant module
        from assistant_core import build_messages

        messages = build_messages(
            system_prompt_to_send,
            actions_desc_to_send,
            history,
            user_input,
            rolling_window,
            summarize_memory,
            prefetch_texts=prefetch_texts,
        )

    payload = {"model": MODEL, "messages": messages}
    payload.update(LLM_OPTIONS)
    if "keep_alive" not in payload:
        payload["keep_alive"] = settings.get("ollama_keep_alive", "30m")

    use_stream = LLM_STREAM or bool(LLM_OPTIONS.get("stream"))

    try:
        kv = await get_default_kv()
    except Exception:
        kv = None

    messages_json = None
    cache_key = None
    if kv is not None:
        try:
            messages_json = json.dumps(messages, sort_keys=True)
            cache_key = response_cache_key(MODEL, LLM_OPTIONS, messages_json)
            cached = await kv.get(cache_key)
            if cached is not None:
                return cached, prefetch_texts

            partial_key = cache_key + ":partial"
            partial = await kv.get(partial_key)
            if partial is not None:
                async def _background_refresh(k: "KVStore", ckey: str, payload_obj: dict):
                    timeout = aiohttp.ClientTimeout(total=LLM_TIMEOUT)
                    try:
                        async with aiohttp.ClientSession(timeout=timeout) as session:
                            async with session.post(OLLAMA_URL, json=payload_obj) as resp:
                                if resp.status != 200:
                                    text = await resp.text()
                                    return None
                                data = await resp.json()
                                if "message" in data and "content" in data["message"]:
                                    final_text = data["message"]["content"]
                                    try:
                                        await k.set(ckey, final_text, ex=FINAL_TTL)
                                        await k.delete(ckey + ":partial")
                                    except Exception:
                                        pass
                                    return final_text
                    except Exception:
                        return None

                try:
                    final_now = await _background_refresh(kv, cache_key, payload)
                    if final_now is not None:
                        return final_now, prefetch_texts
                except Exception:
                    pass
                try:
                    asyncio.create_task(_background_refresh(kv, cache_key, payload))
                except Exception:
                    pass
                return partial, prefetch_texts
        except Exception:
            pass

    if use_stream:
        async def _stream_gen():
            timeout = aiohttp.ClientTimeout(total=LLM_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                try:
                    async with session.post(OLLAMA_URL, json=payload) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            raise RuntimeError(f"LLM returned status {resp.status}: {text}")

                        buf = ""
                        assembled = []
                        last_saved_len = 0
                        partial_key = (cache_key + ":partial") if cache_key else None
                        save_threshold = PARTIAL_SAVE_THRESHOLD

                        async for raw_chunk in resp.content.iter_chunked(1024):
                            try:
                                s = raw_chunk.decode()
                            except Exception:
                                s = raw_chunk.decode(errors="ignore")
                            buf += s
                            while "\n\n" in buf:
                                event, buf = buf.split("\n\n", 1)
                                data_lines = []
                                for line in event.splitlines():
                                    if line.startswith("data:"):
                                        data_lines.append(line[5:].lstrip())
                                if not data_lines:
                                    continue
                                data_str = "\n".join(data_lines)
                                try:
                                    obj = json.loads(data_str)
                                except Exception:
                                    if data_str.strip():
                                        assembled.append(data_str)
                                        yield data_str
                                    continue

                                pieces = _extract_texts(obj)
                                if pieces:
                                    for piece in pieces:
                                        if piece:
                                            assembled.append(piece)
                                            yield piece

                            if partial_key and assembled and len("".join(assembled)) - last_saved_len >= save_threshold:
                                try:
                                    await kv.set(partial_key, "".join(assembled), ex=PARTIAL_TTL)
                                    last_saved_len = len("".join(assembled))
                                except Exception:
                                    pass

                        if buf.strip():
                            try:
                                obj = json.loads(buf)
                                pieces = _extract_texts(obj)
                                for piece in pieces:
                                    if piece:
                                        assembled.append(piece)
                                        yield piece
                            except Exception:
                                assembled.append(buf)
                                yield buf

                        final_text = "".join(assembled)
                        if kv is not None and cache_key:
                            try:
                                await kv.set(cache_key, final_text, ex=FINAL_TTL)
                                if partial_key:
                                    await kv.delete(partial_key)
                            except Exception:
                                pass
                except Exception as e:
                    import traceback
                    raise RuntimeError(f"LLM streaming request failed: {e!r}\n{traceback.format_exc()}")

        return _stream_gen(), prefetch_texts

    timeout = aiohttp.ClientTimeout(total=LLM_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(OLLAMA_URL, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"LLM returned status {resp.status}: {text}")
                data = await resp.json()
        except Exception as e:
            import traceback
            raise RuntimeError(f"LLM request failed: {e!r}\n{traceback.format_exc()}")

    if "message" not in data or "content" not in data["message"]:
        raise RuntimeError(f"Unexpected LLM response shape: {data}")

    final = data["message"]["content"]
    if kv is not None and cache_key:
        try:
            await kv.set(cache_key, final, ex=FINAL_TTL)
        except Exception:
            pass

    return final, prefetch_texts


async def check_model_endpoint(debug: bool = False):
    import traceback
    from urllib.parse import urlparse

    parsed = urlparse(OLLAMA_URL)
    if not parsed.scheme or not parsed.netloc:
        return False, f"invalid OLLAMA_URL: {OLLAMA_URL}"
    tags_url = f"{parsed.scheme}://{parsed.netloc}/api/tags"

    timeout = aiohttp.ClientTimeout(total=min(15, LLM_TIMEOUT))
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(tags_url) as resp:
                text = await resp.text()
                if resp.status != 200:
                    return False, f"status={resp.status} body={text[:500]}"
                try:
                    data = await resp.json()
                except Exception as e:
                    return False, f"invalid json response: {e} / body={text[:500]}"
    except Exception as e:
        tb = traceback.format_exc()
        return False, tb

    if not isinstance(data, dict) or "models" not in data:
        return False, f"unexpected tags response shape: {data}"

    return True, f"connected ({len(data.get('models', []))} models listed)"
