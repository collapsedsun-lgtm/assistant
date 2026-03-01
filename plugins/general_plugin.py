from datetime import datetime, timezone
import os
import json
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


def _load_local_timezone():
    try:
        here = os.path.dirname(__file__)
        settings_path = os.path.abspath(os.path.join(here, "..", "settings.json"))
        if os.path.exists(settings_path):
            with open(settings_path, "r", encoding="utf-8") as f:
                s = json.load(f)
            tz = s.get("local_timezone")
            if tz:
                return tz
    except Exception:
        pass
    return None


def register():
    async def get_time(args: dict):
        # Prefer reporting local time according to settings.json `local_timezone`.
        local_tz = _load_local_timezone()
        now_dt = None
        tz_used = None
        try:
            if local_tz and ZoneInfo is not None:
                tzobj = ZoneInfo(local_tz)
                now_dt = datetime.now(tzobj)
                tz_used = local_tz
            else:
                # fallback to system local timezone-aware datetime
                now_dt = datetime.now().astimezone()
                tz_used = now_dt.tzname() or "local"
        except Exception:
            # final fallback to UTC
            now_dt = datetime.now(timezone.utc)
            tz_used = "UTC"

        iso = now_dt.isoformat()
        readable = now_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        spoken = now_dt.strftime("%H:%M %Z")

        result = {
            "status": "ok",
            "type": "get_time",
            "timezone": tz_used,
            "local_iso": iso,
            "local_readable": readable,
            "spoken": f"The current time is {spoken}."
        }
        print(f"[general_plugin] ACTION: get_time -> {result}")
        return result

    return {"get_time": get_time}
