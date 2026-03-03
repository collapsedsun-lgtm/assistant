from datetime import datetime, timezone
import os
import json

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


def _load_tz_from_settings(settings: dict | None = None) -> str:
    if settings and settings.get("local_timezone"):
        return str(settings.get("local_timezone"))
    try:
        here = os.path.dirname(__file__)
        settings_path = os.path.abspath(os.path.join(here, "..", "settings.json"))
        if os.path.exists(settings_path):
            with open(settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            tz = data.get("local_timezone")
            if tz:
                return str(tz)
    except Exception:
        pass
    return "UTC"


async def _prefetch_time_context(user_input: str, history, debug: bool = False, settings: dict | None = None):
    if settings and settings.get("always_on_time_context", True) is False:
        return None

    tz_name = _load_tz_from_settings(settings)
    try:
        if ZoneInfo is not None:
            now = datetime.now(ZoneInfo(tz_name))
        else:
            now = datetime.now().astimezone()
            tz_name = now.tzname() or tz_name
    except Exception:
        now = datetime.now(timezone.utc)
        tz_name = "UTC"

    iso = now.isoformat()
    spoken = now.strftime("%H:%M")
    weekday = now.strftime("%A")
    date_str = now.strftime("%Y-%m-%d")
    return f"Time (sanitized): local_time={spoken}, weekday={weekday}, date={date_str}, timezone={tz_name}, local_iso={iso}"


def register():
    return {
        "actions": {},
        "pre_send": [_prefetch_time_context],
        "provider": "context_time",
    }
