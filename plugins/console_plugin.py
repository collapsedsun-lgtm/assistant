def register():
    async def turn_on_light(args: dict):
        room = args.get("room", "unknown")
        msg = f"[console_plugin] ACTION: turn_on_light -> room={room}"
        print(msg)
        return {"status": "printed", "message": msg}

    async def turn_off_light(args: dict):
        room = args.get("room", "unknown")
        msg = f"[console_plugin] ACTION: turn_off_light -> room={room}"
        print(msg)
        return {"status": "printed", "message": msg}

    async def get_time(args: dict):
        msg = "[console_plugin] ACTION: get_time"
        print(msg)
        return {"status": "printed", "message": msg}

    return {
        "turn_on_light": turn_on_light,
        "turn_off_light": turn_off_light,
        "get_time": get_time,
    }
