from datetime import datetime, timezone


def register():
    async def get_time(args: dict):
        # Return current UTC ISO timestamp (consumer may convert/display local time).
        now = datetime.now(timezone.utc).isoformat()
        msg = now
        print(f"[general_plugin] ACTION: get_time -> {msg}")
        return {"status": "ok", "message": msg}

    return {"get_time": get_time}
